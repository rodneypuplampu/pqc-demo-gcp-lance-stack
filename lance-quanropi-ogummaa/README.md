# Secure-Cloud-Encryption Powered by Quantropi QiSpace™

A highly secure, quantum-resistant application for encrypting files before they are uploaded to the cloud, as well as securely decrypting files shared within an encryption group. 

By integrating **Quantropi QiSpace™**, this application upgrades traditional cryptographic standards by utilizing a combination of **quantum-enhanced** symmetric and post-quantum asymmetric encryption. This ensures that files are completely secure against both modern classical attacks and future quantum computing threats (protecting against "Harvest Now, Decrypt Later" attacks). The architecture gracefully enables the adding and removing of users from an encryption group so that files can be safely shared between multiple people in a zero-trust environment.

### 🛡️ Why Quantropi QiSpace™?
Standard encryption is only as secure as the random numbers used to generate keys, and standard asymmetric algorithms (like RSA or ECC) are vulnerable to future quantum computers. By securing this application with the QiSpace platform, all cryptographic operations are elevated:
*   **SEQUR™ (Quantum Entropy):** Generates true quantum-random keys, removing the vulnerabilities of pseudo-random number generators.
*   **MASQ™ (Asymmetric PQC):** Replaces legacy public-key cryptography with NIST-compliant, post-quantum algorithms for securing user identities and key exchanges.
*   **QEEP™ (Symmetric):** Provides ultra-fast, quantum-secure symmetric encryption for the file payloads.

---

### 📋 Overview & Architecture

- A **quantum-secure asymmetric public/private key pair** (powered by QiSpace MASQ™) is generated for each new user account created.
- A user’s private key is stored strictly locally on their device. *(Note: If they lose it, they will no longer be able to decrypt the files).*
- A user’s post-quantum public key is stored securely in the database for anyone in the network to access (Firebase Realtime Database / Firestore).
- **Quantum-enhanced symmetric AES keys** (seeded via true quantum entropy from QiSpace SEQUR™) are generated for each user. To share a file, this symmetric key is encrypted with the recipient's post-quantum public key and stored in the **database** (Firebase).
- All encrypted files (ciphertexts) are stored in the cloud storage area (Firebase Storage). Because files are encrypted client-side using quantum-safe keys, the cloud provider has zero knowledge of the data.
- All user account passwords are automatically hashed by Firebase Auth using **Scrypt**, which is considered a highly secure, memory-hard hashing algorithm.
- The account password is used to symmetrically encrypt the locally stored private key so that the private key is protected at rest and can only be used when the user is actively logged in.

---

## 🐳 Deploying the QiSpace Container Application

To enable the quantum-secure cryptographic functions locally or in your backend architecture, this application requires a connection to a Quantropi QiSpace™ distribution node. Deploying the node as a containerized service ensures secure, scalable, and redundant operation within your infrastructure.

### Prerequisites
*   **Docker Engine** (v20.10+) installed and running.
*   **Docker Compose** (optional, but recommended).
*   **Quantropi QiSpace Credentials**:
    *   Registry access token (to pull the official image).
    *   API Key / Space ID (provided via your Quantropi Developer Portal).

### 1. Environment Configuration

To securely manage your credentials, never hardcode them into your codebase. Create a `.env` file in your root deployment directory:

```env
# Quantropi QiSpace Configuration
QISPACE_API_KEY=your_qispace_api_key_here
QISPACE_SPACE_ID=your_space_id_here
QISPACE_ENVIRONMENT=production # e.g., 'production' or 'sandbox'

# Application & Network Settings
PORT=8080
