#!/bin/bash
# Removes everything deploy.sh created. Pass --delete-bucket to also delete the
# S3 bucket and its data (patients, memory, logs).
set -euo pipefail
NAME=patient-followup
export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${S3_BUCKET:-$NAME-$ACCOUNT}

IDS=$(aws ec2 describe-instances --filters Name=tag:Name,Values=$NAME \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query "Reservations[].Instances[].InstanceId" --output text)
if [ -n "$IDS" ]; then
  aws ec2 terminate-instances --instance-ids $IDS >/dev/null
  aws ec2 wait instance-terminated --instance-ids $IDS
fi
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$NAME --query "SecurityGroups[0].GroupId" --output text)
[ "$SG" != None ] && aws ec2 delete-security-group --group-id "$SG"

aws iam remove-role-from-instance-profile --instance-profile-name $NAME-ec2 --role-name $NAME-ec2 2>/dev/null || true
aws iam delete-instance-profile --instance-profile-name $NAME-ec2 2>/dev/null || true
aws iam detach-role-policy --role-name $NAME-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
aws iam delete-role-policy --role-name $NAME-ec2 --policy-name $NAME 2>/dev/null || true
aws iam delete-role --role-name $NAME-ec2 2>/dev/null || true
aws ssm delete-parameter --name /patient-followup/anthropic-api-key 2>/dev/null || true

if [ "${1:-}" = --delete-bucket ]; then
  aws s3 rm "s3://$BUCKET" --recursive >/dev/null
  aws s3api delete-bucket --bucket "$BUCKET"
fi
echo "Teardown complete."
