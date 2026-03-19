# Meeting Summarizer

Automatically transcribes meeting audio recordings and generates formatted meeting minutes using Whisper (via [Speaches](https://speaches.ai)) and an LLM of your choice.

## Features

- Supports any common audio format (m4a, mp3, wav, etc.) — converts automatically if needed
- Transcribes using OpenAI Whisper large-v3 via a self-hosted Speaches server
- Detects and retries hallucinated segments automatically (no manual intervention)
- Generates Robert's Rules of Order formatted minutes in Markdown
- **Three LLM backends for summarization** (see below)
- Unloads Whisper and local LLM models from memory when done
- Saves the raw transcript alongside the minutes for reference
- Interactive prompts for missing arguments; saves config for future runs

---

## LLM Backends

| Backend | Flag | Requires | Best for |
|---|---|---|---|
| **Claude API** | `--backend claude-api` | `ANTHROPIC_API_KEY` | Best quality output |
| **Ollama** | `--backend ollama` | Ollama running locally | Free, fully local |
| **Claude Code CLI** | `--backend claude-cli` | `claude` CLI + Pro subscription | No API key needed |

The default is `claude-api`. If no backend is specified on the command line, you will be prompted to choose one interactively.

---

## Requirements

- Python 3.10+
- `ffmpeg` installed and on your PATH
- A running [Speaches](https://speaches.ai) server (self-hosted Whisper)
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
| `SPEACHES_URL` | `http://localhost:8000` | URL of your Speaches server |
| `ANTHROPIC_API_KEY` | *(none)* | Required for `--backend claude-api` |
| `OLLAMA_URL` | `http://localhost:11434` | URL of your Ollama server |
| `WHISPER_MODEL` | `Systran/faster-whisper-large-v3` | Whisper model to use |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for API backend |
| `OLLAMA_MODEL` | `qwen2.5:latest` | Default Ollama model |

---

## Quick Start

Just run the script with your audio file — it will walk you through everything:

```
$ python summarize_meeting.py "March Meeting.m4a"

Speaches server URL [http://localhost:8000]:
Meeting name (used for output filename, or leave blank to use audio filename): 202603 Meeting Minutes
Attendee/vendor names for spelling hints [...]: Jane Smith, John Doe, Alice Johnson
Meeting context for the LLM [...]: Example Community Committee, March 18 2026, meeting starts at 6 PM
LLM backend [claude-api / ollama / claude-cli] [claude-api]:
  Config saved → config.json

[1/2] Transcribing March Meeting.m4a ...
  Sending March Meeting.m4a (28,432 KB) to Speaches ...
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

## Usage

### Basic

```bash
python summarize_meeting.py recording.m4a
```

Transcribes the audio and writes minutes to `recording.md`. Also saves the raw transcript to `recording.txt`.

### With name hints and meeting context

Providing names improves Whisper spelling accuracy and helps the LLM format the attendee list correctly.

```bash
python summarize_meeting.py recording.m4a \
  --meeting-name "202603 Meeting Minutes" \
  --names "Jane Smith, John Doe, Alice Johnson" \
  --context "Example Community Committee, March 18 2026, meeting starts at 6 PM"
```

### Choose an LLM backend

```bash
# Claude API (default — requires ANTHROPIC_API_KEY)
python summarize_meeting.py recording.m4a --backend claude-api

# Local Ollama model
python summarize_meeting.py recording.m4a --backend ollama --ollama-model qwen2.5:latest

# Claude Code CLI (uses your Claude Pro subscription, no API key needed)
python summarize_meeting.py recording.m4a --backend claude-cli
```

### Re-generate minutes without re-transcribing

Useful if you want to tweak the output without waiting for Whisper again.

```bash
python summarize_meeting.py recording.m4a --skip-transcribe recording.txt
```

### Transcribe only (no LLM)

```bash
python summarize_meeting.py recording.m4a --transcript-only
```

---

## All Options

If you run the script without arguments, it will interactively prompt you for the most important ones. Use `--yes` / `-y` to skip prompts and accept defaults.

```
usage: summarize_meeting.py [-h] [--output OUTPUT] [--names NAMES]
                            [--context CONTEXT] [--hotwords HOTWORDS]
                            [--backend {claude-api,ollama,claude-cli}]
                            [--ollama-model OLLAMA_MODEL]
                            [--speaches-url URL]
                            [--transcript-only]
                            [--skip-transcribe TXT]
                            [--yes]
                            [input]

positional arguments:
  input                       Audio file (m4a, mp3, wav, …)
                              Prompted interactively if not provided.

optional arguments:
  --output, -o OUTPUT         Output markdown file (default: <input>.md)
  --names, -n NAMES           Comma-separated attendee/vendor names for
                              spelling hints (passed to both Whisper and LLM)
  --context, -c CONTEXT       Free-text context for the LLM
                              (org name, date, start time, etc.)
  --hotwords HOTWORDS         Comma-separated hotwords for Whisper accuracy
                              (defaults to --names if not set)
  --backend                   LLM backend: claude-api | ollama | claude-cli
                              Prompted interactively if not provided.
  --ollama-model OLLAMA_MODEL Ollama model name (default: qwen2.5:latest)
  --speaches-url URL          Speaches server URL — overrides SPEACHES_URL
                              env var (default: http://localhost:8000)
  --transcript-only           Transcribe only; skip minutes generation
  --skip-transcribe TXT       Skip transcription; use existing .txt file
  --yes, -y                   Skip all interactive prompts; use defaults
```

---

## How Hallucination Detection Works

Whisper can hallucinate — especially during silences — producing repetitive output like *"Yeah. Yeah. Yeah."* for hundreds of segments. The script detects this automatically using two methods:

1. **Sliding window** — if 60% or more of any 20 consecutive segments share the same text, the range is flagged.
2. **Per-segment metrics** — checks Whisper's own quality scores (`compression_ratio`, `no_speech_prob`, `avg_logprob`).

When a bad range is found, the script extracts that portion of audio, anchors Whisper with the last clean sentence as a prompt, and re-transcribes. This usually recovers the real speech.

---

## Output

For an input file `recording.m4a`, the script produces:

- `recording.txt` — raw transcript (saved before LLM processing; useful for debugging or re-running)
- `recording.md` — formatted meeting minutes (or whatever you specify with `--output`)

---

## Notes

- The Whisper model (~3 GB for large-v3) is downloaded automatically on first use if not present on the server.
- The model is unloaded from server memory when the script finishes.
- Transcription of a 40-minute meeting takes roughly 15–20 minutes on a CPU-only server using large-v3 in float32 mode. Servers with GPU or int8 support will be significantly faster.
- For best results, record in a quiet environment close to the microphone.
