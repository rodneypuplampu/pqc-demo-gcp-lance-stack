#!/usr/bin/env bash
# scripts/verify_tunnel.sh
# Post-deploy verification: SSH into PAN-OS and confirm PPK mixing is active.
set -euo pipefail

FW_IP="${1:?Usage: verify_tunnel.sh <fw_ip> <gateway_name> [username]}"
GATEWAY="${2:?}"
FW_USER="${3:-admin}"

echo "=== IKEv2 PPK Verification ==="
echo "    Firewall : ${FW_IP}"
echo "    Gateway  : ${GATEWAY}"
echo "    User     : ${FW_USER}"
echo ""

# Use sshpass if PANOS_ADMIN_PASS is set, otherwise prompt
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"

if [[ -n "${PANOS_ADMIN_PASS:-}" ]]; then
  which sshpass > /dev/null 2>&1 || { echo "Install sshpass for non-interactive auth"; exit 1; }
  SSH_CMD="sshpass -e ${SSH_CMD}"
fi

echo "[1] Checking IKEv2 SA for gateway '${GATEWAY}'..."
${SSH_CMD} "${FW_USER}@${FW_IP}" "show vpn ike-sa gateway ${GATEWAY}" || {
  echo "❌ Could not retrieve IKEv2 SA status"
  exit 1
}

echo ""
echo "[2] Checking PPK negotiation in system logs..."
${SSH_CMD} "${FW_USER}@${FW_IP}" \
  "show log system direction equal forward | match ike | match ppk" 2>/dev/null || \
  echo "   (no PPK log entries — SA may not have renegotiated yet)"

echo ""
echo "[3] Current PPK configuration (IDs only — values are masked)..."
${SSH_CMD} "${FW_USER}@${FW_IP}" \
  "show config running | xpath /config/devices/entry/network/ike/gateway/entry[@name='${GATEWAY}']/protocol/ikev2/pq-ppk/keys"

echo ""
echo "✅ Verification complete."
echo "   If PPK mixing shows 'inactive', trigger IKEv2 renegotiation:"
echo "   > test vpn ike-sa-rekey gateway ${GATEWAY}"
