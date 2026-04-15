# Security

This project is designed for self-hosted use on trusted networks, but includes built-in safeguards for safer shared operation.

## Current Security Controls

### API boundary protections

- Mutating API routes (`POST`, `PUT`, `DELETE` under `/api/`) are protected by a pre-request security guard.
- Optional API token authentication is supported with the `MEETING_SUMMARIZER_API_TOKEN` environment variable.
  - Clients send the token via the `X-API-Token` request header.
  - Token comparison uses constant-time `hmac.compare_digest`.
- Same-origin checks are enabled by default for mutating API routes.
  - Requests are validated against `Origin` / `Referer` when present.
  - Set `ENFORCE_ORIGIN_CHECK=0` only if you intentionally need to disable this behavior.

### SSRF risk reduction

- Server-side URL proxy endpoints validate submitted URLs.
- Only `http` / `https` schemes are allowed.
- Link-local metadata ranges are blocked (for example `169.254.0.0/16`, `fe80::/10`).
- URL validation is applied to:
  - Service health check endpoints
  - VRAM proxy endpoint
  - Server configuration create/update endpoints

### Input validation and request limits

- Global upload size cap is enforced by Flask `MAX_CONTENT_LENGTH` (500 MB).
- Audio uploads are restricted to an allowlist of expected audio file extensions.
- Backend value is allowlisted (`claude-api`, `ollama`, `claude-cli`, `transcript_only`).
- Job input validation includes:
  - Date format validation (`YYYY-MM-DD`)
  - Known organization and server ID checks
  - Length limits for free-text fields
- Previous-minutes payloads are capped at 2 MB (both uploaded file and pasted text).

### Browser hardening headers

Responses include security headers such as:

- `Content-Security-Policy`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: same-origin`
- `Cross-Origin-Resource-Policy: same-origin`
- `Permissions-Policy`

### Output/download safety and stability

- Download responses are generated in memory rather than temporary files.
- This avoids temporary-file race patterns and temp-file accumulation.

## Recommended Production Configuration

For shared access (family/team/LAN users), set:

- `MEETING_SUMMARIZER_API_TOKEN` to a strong random value.
- `ENFORCE_ORIGIN_CHECK=1` (default).

Also recommended:

- Run behind HTTPS (reverse proxy) when exposed outside a trusted LAN.
- Limit network exposure (bind to trusted interfaces, firewall rules).
- Keep container and Python dependencies updated.
- Avoid storing secrets in committed files.

## Security Model Notes

- This project does not currently implement per-user accounts/roles.
- API token auth is shared-secret based.
- If you need multi-tenant or internet-facing deployment, add:
  - Per-user authentication/authorization
  - Rate limiting and abuse controls
  - Centralized audit logging

## Reporting Security Issues

If you discover a vulnerability, please open a private security report where possible (or contact the maintainer directly before public disclosure), including:

- Impact summary
- Reproduction steps
- Affected version/commit
- Suggested remediation
