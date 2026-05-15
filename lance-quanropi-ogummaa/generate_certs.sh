#!/usr/bin/env bash
# scripts/generate_certs.sh
# Generate self-signed TLS certificates for DEVELOPMENT ONLY.
# In production, use certificates issued by your PKI or Let's Encrypt.
set -euo pipefail

CERT_DIR="./distribution-tier/certs/dev"
DAYS=365
HOSTNAME="${1:-localhost}"

echo "⚠️  Generating SELF-SIGNED certificates for DEVELOPMENT use only."
echo "    hostname=${HOSTNAME}  validity=${DAYS} days  output=${CERT_DIR}"
echo ""

mkdir -p "${CERT_DIR}"

# Generate CA key and certificate
openssl req -x509 -newkey rsa:4096 -days "${DAYS}" -nodes \
  -keyout "${CERT_DIR}/ca.key" \
  -out    "${CERT_DIR}/ca.crt" \
  -subj   "/CN=QiSpace-Dev-CA/O=Dev/C=US"

# Generate server key and CSR
openssl req -newkey rsa:4096 -nodes \
  -keyout "${CERT_DIR}/server.key" \
  -out    "${CERT_DIR}/server.csr" \
  -subj   "/CN=${HOSTNAME}/O=Dev/C=US"

# Create SAN extension file
cat > "${CERT_DIR}/ext.cnf" <<EOF
[SAN]
subjectAltName=DNS:${HOSTNAME},DNS:localhost,IP:127.0.0.1
EOF

# Sign the server cert with the CA
openssl x509 -req -days "${DAYS}" \
  -in     "${CERT_DIR}/server.csr" \
  -CA     "${CERT_DIR}/ca.crt" \
  -CAkey  "${CERT_DIR}/ca.key" \
  -CAcreateserial \
  -out    "${CERT_DIR}/server.crt" \
  -extfile "${CERT_DIR}/ext.cnf" \
  -extensions SAN

echo ""
echo "✅ Certificates generated:"
echo "   CA cert:     ${CERT_DIR}/ca.crt"
echo "   Server cert: ${CERT_DIR}/server.crt"
echo "   Server key:  ${CERT_DIR}/server.key"
echo ""
echo "Update distribution-tier/.env:"
echo "   TLS_CERT_PATH=${CERT_DIR}/server.crt"
echo "   TLS_KEY_PATH=${CERT_DIR}/server.key"
