# Meeting Summarizer

Automatically transcribes meeting audio recordings and generates formatted meeting minutes using a self-hosted Whisper API server and an LLM of your choice.

## Features

- Supports any common audio format (m4a, mp3, wav, etc.) — converts automatically if needed
- Transcribes using distil-Whisper large-v3 (6× faster than large-v3, within 1% accuracy)
- Detects and retries hallucinated segments automatically
- Per-committee vocabulary hints improve transcription of names, places, and jargon
- Generates Robert's Rules of Order formatted minutes in Markdown
- **Three LLM backends for summarization** (see below)
- Unloads Whisper and local LLM models from memory when done
- **Web UI** — upload audio and receive minutes in a browser (no command line needed)
- Jobs and downloads persist across container/image restarts (when data volume is mounted)
- Delete finished jobs and their saved files from the web UI
- Upload multiple associated reference files (previous minutes, notes, roster exports)

---

## LLM Backends

| Backend | Flag | Requires | Best for |
|---|---|---|---|
| **Claude API** | `--backend claude-api` | `ANTHROPIC_API_KEY` | Best quality output |
| **Ollama** | `--backend ollama` | Ollama running locally | Free, fully local |
| **Claude Code CLI** | `--backend claude-cli` | `claude` CLI + Pro subscription | No API key needed |

The default is `claude-api`.

---

## Requirements

- Python 3.10+
- `ffmpeg` installed and on your PATH
- A running Whisper API server (e.g. [Speaches](https://speaches.ai) or any OpenAI-compatible `/v1/audio/transcriptions` endpoint)
- One of the following for minutes generation:
  - An Anthropic API key (Claude API)
  - [Ollama](https://ollama.com) running locally with a model pulled
  - The `claude` CLI installed and authenticated (Claude Code)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

The following environment variables can be set in your shell profile (e.g. `~/.bashrc`):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_URL` | `http://localhost:8000` | URL of your Whisper API server |
| `ANTHROPIC_API_KEY` | *(none)* | Required for `--backend claude-api` |
| `OLLAMA_URL` | `http://localhost:11434` | URL of your Ollama server |
| `WHISPER_MODEL` | `Systran/faster-distil-whisper-large-v3` | Whisper model to use |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for API backend |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Default Ollama model |
| `JOB_RETENTION_DAYS` | `30` | Keep completed/failed/cancelled jobs for this many days |
| `DATA_DIR` | `./data` | Base directory for org/server config and persisted job metadata |
| `UPLOAD_DIR` | `<DATA_DIR>/jobs` | Directory where per-job audio, transcript, minutes, and associated files are stored |

---

## Quick Start

```
$ python summarize_meeting.py "March Meeting.m4a"

Whisper API server URL [http://localhost:8000]:
Meeting name (used for output filename, or leave blank to use audio filename): 202603 Meeting Minutes
Attendee/vendor names for spelling hints [...]: Jane Smith, John Doe, Alice Johnson
Meeting context for the LLM [...]: Example Community Committee, March 18 2026, meeting starts at 6 PM
LLM backend [claude-api / ollama / claude-cli] [claude-api]:
  Config saved → config.json

[1/2] Transcribing March Meeting.m4a ...
  Sending March Meeting.m4a (28,432 KB) to Whisper API (Systran/faster-distil-whisper-large-v3)...
  No hallucinations detected.
  Transcript saved → 202603 Meeting Minutes.txt

[2/2] Generating meeting minutes (backend: claude-api) ...
  Sending transcript to Claude API (claude-sonnet-4-6) ...
  Meeting minutes saved → 202603 Meeting Minutes.md

[3/3] Cleaning up ...
  Whisper model unloaded from server memory.

Done.
```

Your answers are saved to `config.json` and pre-filled as defaults next time. Use `-y` to skip all prompts and accept saved defaults.

---

## Web UI

A browser-based interface is available for non-technical users:

```bash
python web.py
# Open http://localhost:8082
```

Features:
- Select a pre-configured committee or add your own
- Audio upload with optional associated reference files (multi-file)
- Live transcription/generation log
- Download minutes (`.md`) and transcript (`.txt`)
- Import committee templates from YAML files
- Active/recent jobs survive restarts and can be deleted (with associated files)

Associated files can include previous-month minutes, meeting notes, roster exports, and similar reference material (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`).

### Job Persistence & Cleanup

- Job metadata and outputs are saved on disk and reloaded on startup.
- Completed/failed/cancelled jobs are retained for `JOB_RETENTION_DAYS` (default 30).
- Queued/running jobs interrupted by a restart are marked as interrupted.
- Non-active jobs can be deleted from the UI, which removes job metadata and associated files.

### Running with Docker

```bash
docker compose up
```

Or pull the pre-built image:

```bash
docker run -p 8082:8082 \
  -e ANTHROPIC_API_KEY=sk-... \
  -v meeting-data:/data \
  ghcr.io/bradbrownjr/meeting-summarizer:latest
```

### Quick Hardening Checklist

For shared use, set these before exposing the app to other users:

1. Set a strong API token:

```bash
export MEETING_SUMMARIZER_API_TOKEN="replace-with-long-random-secret"
```

2. Keep origin checks enabled (default is enabled):

```bash
export ENFORCE_ORIGIN_CHECK=1
```

3. Prefer HTTPS behind a reverse proxy if accessed outside a trusted LAN.
4. Restrict network exposure with firewall rules and trusted interfaces only.

To preserve jobs and downloads across upgrades/restarts, keep `/data` mounted to persistent storage.

See [SECURITY.md](SECURITY.md) for full security details.

---

## Committee Templates (YAML)

Define reusable committee presets as YAML files and import them via the web UI:

```yaml
id: mycommittee
label: My Committee
short: MC
context_template: "My Committee, Town Hall, {date}, meeting called to order at 7 PM"
name_template: "My Committee Meeting – {month_year}"
vocabulary:
  - My Committee
  - Town Hall
  - Jane Smith
  - relevant jargon
```

- `{date}` is replaced with the formatted meeting date (e.g. `April 3, 2026`)
- `{month_year}` is replaced with the month and year (e.g. `April 2026`)
- `vocabulary` terms are passed to Whisper to improve transcription of names, abbreviations, and domain-specific language

See `orgs.json.example` for the full schema.

---

## Usage

### Basic

```bash
python summarize_meeting.py recording.m4a
```

### With vocabulary hints and meeting context

```bash
python summarize_meeting.py recording.m4a \
  --meeting-name "202603 Meeting Minutes" \
  --names "Jane Smith, John Doe, Alice Johnson" \
  --vocabulary "WOHD, Waterboro, Old Home Days" \
  --context "Example Community Committee, March 18 2026, meeting starts at 6 PM"
```

### Choose an LLM backend

```bash
# Claude API (default — requires ANTHROPIC_API_KEY)
python summarize_meeting.py recording.m4a --backend claude-api

# Local Ollama model
python summarize_meeting.py recording.m4a --backend ollama --ollama-model qwen3.5:9b

# Claude Code CLI (uses your Claude Pro subscription, no API key needed)
python summarize_meeting.py recording.m4a --backend claude-cli
```

### Re-generate minutes without re-transcribing

```bash
python summarize_meeting.py recording.m4a --skip-transcribe recording.txt
```

### Transcribe only (no LLM)

```bash
python summarize_meeting.py recording.m4a --transcript-only
```

---

## All Options

```
usage: summarize_meeting.py [-h] [--output OUTPUT] [--names NAMES]
                            [--context CONTEXT] [--vocabulary VOCABULARY]
                            [--backend {claude-api,ollama,claude-cli}]
                            [--ollama-model OLLAMA_MODEL]
                            [--whisper-url URL] [--whisper-model MODEL]
                            [--transcript-only] [--skip-transcribe TXT]
                            [--yes]
                            [input]

positional arguments:
  input                       Audio file (m4a, mp3, wav, …)

optional arguments:
  --output, -o OUTPUT         Output markdown file (default: <input>.md)
  --names, -n NAMES           Comma-separated attendee names for spelling
                              hints (passed to both Whisper and LLM)
  --context, -c CONTEXT       Free-text context for the LLM
                              (org name, date, start time, etc.)
  --vocabulary VOCABULARY     Comma-separated terms to improve Whisper
                              transcription accuracy
  --backend                   LLM backend: claude-api | ollama | claude-cli
  --ollama-model OLLAMA_MODEL Ollama model name (default: qwen3.5:9b)
  --whisper-url URL           Whisper API server URL
  --whisper-model MODEL       Whisper model ID
  --transcript-only           Transcribe only; skip minutes generation
  --skip-transcribe TXT       Skip transcription; use existing .txt file
  --yes, -y                   Skip all interactive prompts; use defaults
```

---

## How Hallucination Detection Works

Whisper can hallucinate — especially during silences — producing repetitive output. The script detects this automatically:

1. **Sliding window** — if 60% or more of any 20 consecutive segments share the same text, the range is flagged.
2. **Per-segment metrics** — checks Whisper's own quality scores (`compression_ratio`, `no_speech_prob`, `avg_logprob`).

When a bad range is found, the script extracts that portion of audio, anchors Whisper with the last clean sentence as a prompt, and re-transcribes.

---

## Notes

- The Whisper model (~1.5 GB for distil-large-v3) is downloaded automatically on first use.
- The model is unloaded from server memory when the script finishes.
- Distil-Whisper large-v3 is ~6× faster than large-v3 with less than 1% accuracy difference.
- For best results, record in a quiet environment close to the microphone.

---

## Security

For deployment hardening guidance and a full list of implemented protections, see [SECURITY.md](SECURITY.md).
