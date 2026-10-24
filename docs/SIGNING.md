# Signing & notarization

BirdSorter builds and runs **unsigned** by default: a fresh clone, a fork, and a
PR all produce working installers with no certificates. Signing is opt-in and
turns on automatically when the right secrets/certificates are present.

- Windows: [§1](#1-windows-code-signing)
- macOS: [§2-macos](#2-macos-signing--notarization)
- Secret reference: [§3](#3-secret-reference)

> **Current status:** no certificate is configured yet, so released installers
> show a SmartScreen/Gatekeeper warning. Everything below is ready to switch on;
> the only missing piece is buying the certificate.

---

## 1. Windows code signing

Without a signature, Windows SmartScreen shows *"Windows protected your PC"* on
first run. Options, cheapest first:

| Option | Approx. cost | Notes |
| --- | --- | --- |
| **Azure Trusted Signing** | ~$10/month | Cheapest for individuals; no hardware token; signs in CI with `trusted-signing-cli`. Best choice for this project. |
| **OV certificate** (Sectigo/DigiCert/SSL.com) | ~$200–400/yr | Needs the cert on a hardware token/USB HSM, or a cloud HSM. Reputation builds over time. |
| **EV certificate** | ~$400–600/yr | Instant SmartScreen reputation, but still token/HSM-bound. |

A standard OV/EV certificate file is a `.pfx` (PKCS#12) once exported from the
token/HSM as **base64**. Azure Trusted Signing does not use a `.pfx` — see the
note at the end of this section.

### What the build does

`.github/workflows/release.yml` on `windows-latest`:

1. decodes `WINDOWS_CERTIFICATE` (base64 `.pfx`) and imports it into
   `Cert:\CurrentUser\My`;
2. writes `tauri.signing.json` with the certificate thumbprint, SHA-256 digest
   and a timestamp URL;
3. `tauri build --config tauri.signing.json` signs the **app exe** and the NSIS
   and MSI installers.

With no secret the step writes `{}` and the build is unsigned, so nothing breaks.

### Getting the `.pfx` into base64

```powershell
# From a .pfx you exported from your provider:
[Convert]::ToBase64String([IO.File]::ReadAllBytes("birdsorter.pfx")) |
  Set-Clipboard          # paste into the WINDOWS_CERTIFICATE secret
```

Set these repository secrets (Settings ▸ Secrets and variables ▸ Actions):

- `WINDOWS_CERTIFICATE` — base64 `.pfx`
- `WINDOWS_CERTIFICATE_PASSWORD` — its password

Then re-run the Release workflow (or `workflow_dispatch`).

### Local signed build

```powershell
# 1. Import the certificate
Import-PfxCertificate -FilePath birdsorter.pfx -CertStoreLocation Cert:\CurrentUser\My

# 2. Point Tauri at it (thumbprint from certmgr.msc, no spaces)
@'
{ "bundle": { "windows": {
    "certificateThumbprint": "ABCDEF...",
    "digestAlgorithm": "sha256",
    "timestampUrl": "http://timestamp.digicert.com"
} } }
'@ | Set-Content tauri.signing.local.json

# 3. Build (frontend first)
cd ui; npm run build; cd ..
cd crates/desktop
../../ui/node_modules/.bin/tauri build --config ../../tauri.signing.local.json
```

Verify:

```powershell
Get-AuthenticodeSignature "target\release\bundle\nsis\BirdSorter_0.1.0_x64-setup.exe" |
  Format-List Status, SignerCertificate, TimeStamperCertificate
```

`Status: Valid` means it worked. `tauri.signing.local.json` is gitignored.

### Azure Trusted Signing instead

Azure Trusted Signing signs via a service, not a file, so `certificateThumbprint`
does not apply. Install `trusted-signing-cli` on the runner and set Tauri's
custom sign command:

```json
{ "bundle": { "windows": {
  "signCommand": "trusted-signing-cli -e https://eus.codesigning.azure.net -a <account> -c <profile> -d BirdSorter %1"
} } }
```

Store the service principal values as secrets and export them as
`AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_CLIENT_SECRET` in the workflow.

---

## 2. macOS signing & notarization

Without a Developer ID signature and notarization, macOS Gatekeeper refuses to
open the app ("cannot be opened because the developer cannot be verified").

Requirements:

1. An **Apple Developer Program** membership ($99/yr).
2. A **Developer ID Application** certificate (create in Xcode ▸ Settings ▸
   Accounts ▸ Manage Certificates, or at developer.apple.com).
3. An app-specific password for notarization (appleid.apple.com ▸ Sign-In and
   Security ▸ App-Specific Passwords).

Export the certificate as a `.p12` and base64 it:

```bash
base64 -i DeveloperID.p12 | pbcopy   # -> APPLE_CERTIFICATE
```

Tauri imports the `.p12` itself and notarizes with `notarytool`; the workflow
only needs the secrets in §3. If `APPLE_CERTIFICATE` is empty, the macOS builds
stay unsigned (fine for local testing).

Local signed build:

```bash
export APPLE_CERTIFICATE=$(base64 -i DeveloperID.p12)
export APPLE_CERTIFICATE_PASSWORD="..."
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAMID"
cd ui && npm run build && cd ../crates/desktop
../../ui/node_modules/.bin/tauri build --target aarch64-apple-darwin
```

Verify:

```bash
spctl -a -vv "target/release/bundle/macos/BirdSorter.app"   # accepted
codesign -dv --verbose=4 "target/release/bundle/macos/BirdSorter.app"
```

Signed installers for every platform are built by the Tauri CLI; see the macOS
notes above for the notarization steps.

---

## 3. Secret reference

| Secret | Platform | Purpose |
| --- | --- | --- |
| `WINDOWS_CERTIFICATE` | Windows | base64 `.pfx` |
| `WINDOWS_CERTIFICATE_PASSWORD` | Windows | `.pfx` password |
| `APPLE_CERTIFICATE` | macOS | base64 `.p12` (Developer ID Application) |
| `APPLE_CERTIFICATE_PASSWORD` | macOS | `.p12` password |
| `APPLE_SIGNING_IDENTITY` | macOS | e.g. `Developer ID Application: Name (TEAMID)` |
| `APPLE_ID` | macOS | Apple account email |
| `APPLE_PASSWORD` | macOS | app-specific password |
| `APPLE_TEAM_ID` | macOS | 10-character team ID |

All are optional; missing values degrade to an unsigned build rather than a
failed one.

## 4. Auto-updater (not enabled yet)

Once releases are signed, `tauri-plugin-updater` can be added: generate a
keypair (`tauri signer generate`), store the private key as
`TAURI_SIGNING_PRIVATE_KEY`, put the public key in `tauri.conf.json` under
`plugins.updater.pubkey`, and upload the `latest.json` manifest. Do this
**after** signing works, otherwise updates are blocked by the OS anyway.
