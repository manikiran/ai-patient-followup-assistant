#!/bin/bash
# One-command deploy to EC2 (free-tier services + Claude). Idempotent: re-run
# to ship new code (it replaces the instance). Prints the app URL at the end.
#
#   PYTHON=.venv/bin/python bash deploy/ec2/deploy.sh
#
# Needs: AWS CLI credentials with deploy/ec2/deployer-policy.json, a Python
# env with requirements.txt installed, and ANTHROPIC_API_KEY in env or .env.
# Optional env: AWS_REGION (us-east-1), INSTANCE_TYPE (t3.small),
# ALLOWED_CIDR (your current public IP /32), S3_BUCKET.
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME=patient-followup
REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=$REGION
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${S3_BUCKET:-$NAME-$ACCOUNT}
INSTANCE_TYPE=${INSTANCE_TYPE:-t3.small}
ALLOWED_CIDR=${ALLOWED_CIDR:-$(curl -s https://checkip.amazonaws.com)/32}
PYTHON=${PYTHON:-python}
step() { echo; echo "==> $*"; }

step "S3 bucket s3://$BUCKET"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = us-east-1 ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

step "Synthetic patients + vector index -> S3"
S3_BUCKET=$BUCKET AWS_REGION=$REGION "$PYTHON" -m src.synthetic_data --count 100 --upload
S3_BUCKET=$BUCKET AWS_REGION=$REGION VECTOR_DIR=$(mktemp -d)/vectorstore "$PYTHON" -m src.rag build

step "Source bundle -> S3"
BUNDLE=$(mktemp -d)/source.tar.gz
tar czf "$BUNDLE" Dockerfile requirements.txt app.py src schema scripts data/patients
aws s3 cp "$BUNDLE" "s3://$BUCKET/deploy/source.tar.gz" >/dev/null

step "Claude API key -> SSM Parameter Store (SecureString, standard tier)"
KEY=${ANTHROPIC_API_KEY:-$(grep -E '^ANTHROPIC_API_KEY=' .env 2>/dev/null | cut -d= -f2- || true)}
[ -n "$KEY" ] || { echo "ANTHROPIC_API_KEY not found in env or .env" >&2; exit 1; }
PARAM_FILE=$(mktemp) && chmod 600 "$PARAM_FILE"
KEY="$KEY" "$PYTHON" -c 'import json, os; print(json.dumps({"Name": "/patient-followup/anthropic-api-key", "Type": "SecureString", "Tier": "Standard", "Overwrite": True, "Value": os.environ["KEY"]}))' > "$PARAM_FILE"
aws ssm put-parameter --cli-input-json "file://$PARAM_FILE" >/dev/null
rm -f "$PARAM_FILE"; unset KEY

step "IAM role + instance profile $NAME-ec2"
if ! aws iam get-role --role-name "$NAME-ec2" >/dev/null 2>&1; then
  aws iam create-role --role-name "$NAME-ec2" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
sed -e "s/YOUR_BUCKET/$BUCKET/g" -e "s/YOUR_ACCOUNT_ID/$ACCOUNT/g" -e "s/us-east-1/$REGION/g" \
  deploy/ec2/iam-policy.json > "$(dirname "$BUNDLE")/role-policy.json"
aws iam put-role-policy --role-name "$NAME-ec2" --policy-name "$NAME" \
  --policy-document "file://$(dirname "$BUNDLE")/role-policy.json"
# Session Manager shell access for troubleshooting, instead of SSH keys.
aws iam attach-role-policy --role-name "$NAME-ec2" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$NAME-ec2" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$NAME-ec2" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$NAME-ec2" --role-name "$NAME-ec2"
  sleep 15  # let the new instance profile propagate
fi

step "Security group: port 8501 from $ALLOWED_CIDR only"
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$NAME Name=vpc-id,Values="$VPC" \
  --query "SecurityGroups[0].GroupId" --output text)
if [ "$SG" = None ]; then
  SG=$(aws ec2 create-security-group --group-name $NAME --vpc-id "$VPC" \
    --description "Streamlit from known IPs" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$NAME}]" \
    --query GroupId --output text)
fi
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 8501 \
  --cidr "$ALLOWED_CIDR" >/dev/null 2>&1 || true  # already allowed

step "EC2 instance ($INSTANCE_TYPE)"
OLD=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$NAME \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query "Reservations[].Instances[].InstanceId" --output text)
if [ -n "$OLD" ]; then
  echo "Replacing previous instance(s): $OLD"
  aws ec2 terminate-instances --instance-ids $OLD >/dev/null
fi
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-kernel-*-x86_64" Name=state,Values=available \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text)
USER_DATA=$(dirname "$BUNDLE")/user-data.sh
sed -e "s/__S3_BUCKET__/$BUCKET/" -e "s/__AWS_REGION__/$REGION/" deploy/ec2/user-data.sh > "$USER_DATA"
INSTANCE=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --security-group-ids "$SG" --iam-instance-profile Name=$NAME-ec2 \
  --metadata-options HttpTokens=required \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
  --user-data "file://$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query "Instances[0].InstanceId" --output text)
aws ec2 wait instance-running --instance-ids "$INSTANCE"
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
URL="http://$IP:8501"

step "Waiting for the app to build and start at $URL (usually 5-10 min)"
for _ in $(seq 1 90); do
  if curl -sf --max-time 5 "$URL/_stcore/health" >/dev/null; then
    echo; echo "Deployed: $URL   (instance $INSTANCE, reachable from $ALLOWED_CIDR)"
    exit 0
  fi
  sleep 10; printf .
done
echo; echo "Instance $INSTANCE is up but the app isn't answering yet. Check progress with:" >&2
echo "  aws ssm start-session --target $INSTANCE   then: sudo tail -f /var/log/cloud-init-output.log" >&2
exit 1
