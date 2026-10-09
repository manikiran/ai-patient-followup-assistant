#!/bin/bash
# EC2 user data (Amazon Linux 2023), rendered by deploy.sh (__S3_BUCKET__ and
# __AWS_REGION__ are replaced). Builds the app image on the instance from the
# source bundle in S3 and runs it with Claude.
# No secrets in this file: the API key is read from SSM Parameter Store
# (standard tier = free) via the instance's IAM role. (No `set -x`, so the
# key never lands in /var/log/cloud-init-output.log.)
set -euo pipefail

S3_BUCKET="__S3_BUCKET__"
AWS_REGION="__AWS_REGION__"

# 2 GB swap: headroom for the image build and Streamlit + ChromaDB on small instances.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo "/swapfile swap swap defaults 0 0" >> /etc/fstab
fi

dnf install -y docker
systemctl enable --now docker

mkdir -p /opt/followup && cd /opt/followup
aws s3 cp "s3://$S3_BUCKET/deploy/source.tar.gz" source.tar.gz --region "$AWS_REGION"
rm -rf src && mkdir src && tar xzf source.tar.gz -C src
docker build -t followup src

ANTHROPIC_API_KEY=$(aws ssm get-parameter --region "$AWS_REGION" --with-decryption \
  --name /patient-followup/anthropic-api-key --query Parameter.Value --output text)

docker rm -f followup 2>/dev/null || true
docker run -d --name followup --restart unless-stopped \
  -p 8501:8501 \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e S3_BUCKET="$S3_BUCKET" \
  -e AWS_REGION="$AWS_REGION" \
  followup
echo "patient-followup app started on port 8501"
