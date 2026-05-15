#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Gummaa-Atlas Cloud Run deployment helper
# Usage:  chmod +x deploy.sh && ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-multi-agent-lab01}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-gummaa-atlas-api}"
GCS_BUCKET="${GCS_BUCKET_NAME:-lance-data-adk-02}"
SA_NAME="gummaa-agent-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "═══════════════════════════════════════════════════"
echo "  Gummaa-Atlas · Cloud Run Deployment"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE_NAME}"
echo "  Bucket  : ${GCS_BUCKET}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Authenticate + set project ───────────────────────────────────────
echo ""
echo "▶ Step 1 — Configure GCP environment"
gcloud config set project "${PROJECT_ID}"

# ── Step 2: Enable APIs ───────────────────────────────────────────────────────
echo ""
echo "▶ Step 2 — Enabling required APIs"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    iam.googleapis.com

# ── Step 3: Create service account (idempotent) ──────────────────────────────
echo ""
echo "▶ Step 3 — Service account setup"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="Gummaa-Atlas Cloud Run Service Account"
    echo "  Created: ${SA_EMAIL}"
else
    echo "  Already exists: ${SA_EMAIL}"
fi

# Grant Vertex AI + GCS roles
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user" \
    --condition=None

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --condition=None

# ── Step 4: Ensure GCS bucket exists ─────────────────────────────────────────
echo ""
echo "▶ Step 4 — Verifying GCS bucket"
if ! gsutil ls -b "gs://${GCS_BUCKET}" &>/dev/null; then
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${GCS_BUCKET}"
    echo "  Created bucket: gs://${GCS_BUCKET}"
else
    echo "  Bucket exists: gs://${GCS_BUCKET}"
fi

# ── Step 5: Deploy to Cloud Run ───────────────────────────────────────────────
echo ""
echo "▶ Step 5 — Deploying to Cloud Run (source build)…"
gcloud run deploy "${SERVICE_NAME}" \
    --source . \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --service-account "${SA_EMAIL}" \
    --allow-unauthenticated \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${GCS_BUCKET},LANCEDB_PATH=/app/gummaa_workspace.lance,LOCATION=${REGION}" \
    --execution-environment=gen2 \
    --add-volume=name=lance-lake,type=cloud-storage,bucket="${GCS_BUCKET}" \
    --add-volume-mount=volume=lance-lake,mount-path=/app/gummaa_workspace.lance \
    --min-instances=0 \
    --max-instances=10 \
    --memory=2Gi \
    --cpu=2

echo ""
echo "✓ Deployment complete."
echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format="value(status.url)")
echo "  Service URL: ${SERVICE_URL}"
echo ""
echo "  ⊕ Open in browser: ${SERVICE_URL}"
