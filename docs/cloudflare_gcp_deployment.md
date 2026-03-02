# Cloudflare + GCP Deployment Runbook

## Recommended production topology
- Keep `Streamlit` on `Cloud Run` (containerized) because Streamlit is a Python server process.
- Put `Cloudflare` in front as edge/WAF/CDN and TLS termination.
- Keep Vertex AI + Gemini calls on the server side only.

## Why not Cloudflare Pages/Workers for raw Streamlit
- Pages is static hosting.
- Workers are request handlers and do not run Streamlit's long-running Python app server model.

## Option A (recommended): Cloudflare in front of Cloud Run Streamlit
1. Build and deploy Streamlit container to Cloud Run.
2. Restrict Cloud Run ingress to `internal-and-cloud-load-balancing` or authenticated invocations.
3. Put HTTPS Load Balancer in front of Cloud Run.
4. Point Cloudflare DNS to the load balancer and enable proxy mode.
5. Use Cloudflare WAF/rate limiting for public edge controls.

## Option B (Pages/Workers frontend)
1. Build a static frontend (React/Vue) on Cloudflare Pages.
2. Use a Cloudflare Worker as BFF/API gateway.
3. Worker calls a private GCP backend endpoint (Cloud Run) that performs Gemini extraction + Vertex inference.

## Secure backend authentication

### Preferred: Workload Identity Federation (keyless)
1. Create a GCP Workload Identity Pool + Provider for Cloudflare OIDC issuer.
2. Map Cloudflare identity claims to GCP principal attributes.
3. Grant `roles/aiplatform.user` and `roles/monitoring.metricWriter` to the federated principal.
4. Worker exchanges OIDC subject token for short-lived Google access token (STS flow).
5. Worker calls backend/API with short-lived token.

### Fallback: Service account key in Cloudflare secrets (not preferred)
1. Create dedicated service account with least privilege.
2. Store JSON key only in Cloudflare encrypted secrets.
3. Rotate key automatically and alert on stale keys.
4. Restrict IAM and network egress tightly.

## Required IAM (minimum)
- `roles/aiplatform.user`
- `roles/monitoring.metricWriter`
- `roles/logging.logWriter`

## Network controls
- Deny direct internet access to predictor container except required Google APIs.
- Enforce request timeout and retries at backend only.
- Use Cloudflare rate limiting and bot management at edge.
