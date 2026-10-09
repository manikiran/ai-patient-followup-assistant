# Deploying to AWS EC2 (plan M6), free-tier AWS + Claude

All AWS pieces stay within the free tier on an account that still has free-tier
eligibility. The only paid item is Claude usage on your own Anthropic API key.
**Check your own account's free-tier status before launching.** Free-tier
terms changed for accounts created after July 15, 2025: newer accounts get
credits that free-tier-eligible usage draws down, and usage beyond the limits
is billed.

| Piece | Option used |
|---|---|
| Compute | **t3.small** (free-tier eligible on the current AWS free plan; 2 GB RAM) + 2 GB swap. On older accounts where only t2/t3.micro is free, run with `INSTANCE_TYPE=t3.micro`. The architecture doc's t3.medium is *not* free-tier eligible. |
| LLM | **Claude** via your commercial `ANTHROPIC_API_KEY` (billed by Anthropic per token, not by AWS). |
| Embeddings / vector DB | Local MiniLM + ChromaDB inside the container. No cost. |
| Storage | **S3** free tier (5 GB). This app uses well under 50 MB. |
| API key storage | **SSM Parameter Store** standard SecureString (free). Secrets Manager charges per secret, so it isn't used. |
| Image | Built **on the instance** from a source bundle in S3. No registry needed. |
| Shell access | **SSM Session Manager** (free). No SSH keys, no port 22. |

## One-command deploy

```bash
PYTHON=.venv/bin/python bash deploy/ec2/deploy.sh
```

The script is idempotent, and re-running it ships new code. It:

1. creates the private bucket `s3://patient-followup-<account-id>`;
2. uploads the 100 synthetic patients and the vector-index snapshot (T1.5, T2.2);
3. uploads a source bundle for the image build;
4. stores `ANTHROPIC_API_KEY` (from env or `.env`) as a SecureString parameter (T6.2);
5. creates the IAM role `patient-followup-ec2`, scoped to that bucket and parameter;
6. creates a security group that opens **8501 to your current public IP only** (T6.1);
7. launches the instance with `user-data.sh`, which adds swap, installs Docker,
   builds the image and runs it (T6.3);
8. waits for the health check and prints `http://<public-ip>:8501`.

Options (env vars): `AWS_REGION` (default `us-east-1`), `INSTANCE_TYPE`
(`t3.small`), `ALLOWED_CIDR` (your IP `/32`; add more IPs later with
`aws ec2 authorize-security-group-ingress`), `S3_BUCKET`.

### Permissions the deploying AWS user needs

`deployer-policy.json` is a least-privilege policy limited to `patient-followup-*`
resources. Create it as a **customer-managed policy** and attach it to the CLI
user. Don't use an inline policy: all inline policies on a user share one
2048-character limit. As an admin, either use the console (IAM → Policies →
Create policy → JSON → paste → name it `patient-followup-deploy`, then
Users → your user → Add permissions → Attach policies directly), or run:

```bash
aws iam create-policy --policy-name patient-followup-deploy \
  --policy-document file://deploy/ec2/deployer-policy.json
aws iam attach-user-policy --user-name <cli-user> \
  --policy-arn arn:aws:iam::<account-id>:policy/patient-followup-deploy
```

The user also needs read-only IAM access (e.g. the `IAMReadOnlyAccess` managed policy).

## Troubleshooting

```bash
aws ssm start-session --target <instance-id>     # needs the Session Manager plugin
sudo tail -f /var/log/cloud-init-output.log      # first-boot build progress
sudo docker logs -f followup                     # app logs
```

If EC2 is blocked on demo day, the plan's fallback is to run locally:
`docker compose up` or `streamlit run app.py`.

## Tear down

```bash
bash deploy/ec2/teardown.sh                  # instance, security group, role, parameter
bash deploy/ec2/teardown.sh --delete-bucket  # ...and the S3 bucket with all data
```
