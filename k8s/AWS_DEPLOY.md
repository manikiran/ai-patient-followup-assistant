# Deploying to AWS (EKS)

This extends the local-cluster pattern from `ollama-FAPI/k8s` (built for
Minikube) to a real AWS EKS cluster. Unlike that reference project, the
agents here call the Anthropic (Claude) API directly instead of running a
local model server, so there's no Ollama deployment to manage — just the
agent-app and Open WebUI.

## 0. Prerequisites

- AWS CLI configured (`aws configure`)
- `eksctl`, `kubectl`, and Docker installed

## 1. Create an EKS cluster

```bash
eksctl create cluster \
  --name patient-followup \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

`t3.medium` is plenty here — the agent-app is a small FastAPI service with no
model to load.

## 2. Push the agent-app image to ECR

```bash
aws ecr create-repository --repository-name agent-app

docker build -t agent-app:latest .

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag agent-app:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/agent-app:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/agent-app:latest
```

Then in `k8s/agent-app-deployment.yaml`, replace:

```yaml
image: agent-app:latest
imagePullPolicy: Never
```

with:

```yaml
image: <account-id>.dkr.ecr.us-east-1.amazonaws.com/agent-app:latest
imagePullPolicy: Always
```

(`imagePullPolicy: Never` only works for images built directly into a local
Docker daemon like Minikube's — it will fail on EKS.)

## 3. Storage class

EKS's default StorageClass is EBS-backed (`gp2`/`gp3`), so `openwebui-pvc.yaml`
works unmodified — no `storageClassName` override needed unless you want to
pin `gp3` explicitly.

## 4. Create the Anthropic API key secret

The agent-app reads `ANTHROPIC_API_KEY` from a Kubernetes Secret
(`agent-app-secrets`) — never bake the key into a manifest or the image.
Create it imperatively so it never touches a file that could be committed:

```bash
kubectl create secret generic agent-app-secrets \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
```

## 5. Apply manifests

```bash
kubectl apply -f k8s/agent-app-deployment.yaml
kubectl apply -f k8s/agent-app-service.yaml
kubectl apply -f k8s/openwebui-pvc.yaml
kubectl apply -f k8s/openwebui-deployment.yaml
kubectl apply -f k8s/openwebui-service.yaml
```

## 6. Access Open WebUI

`openwebui-service.yaml` is `type: LoadBalancer`, so AWS provisions an ELB:

```bash
kubectl get svc open-webui
# EXTERNAL-IP column is your public endpoint, on port 8080
```

Open that address, create an account (first user becomes admin), and Open
WebUI is already pointed at the agent-app's OpenAI-compatible API via the
`OPENAI_API_BASE_URL` env var baked into the deployment. Chat "run followup
for P0001" to drive the pipeline.

## 7. Tear down

```bash
eksctl delete cluster --name patient-followup
```

This deletes the nodegroup and control plane; make sure you're done before
running it, since it takes any un-persisted state with it.
