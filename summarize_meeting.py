#!/usr/bin/env python3
"""
Meeting Summarizer
Transcribes meeting recordings and generates formatted meeting minutes.

Usage:
  python summarize_meeting.py recording.m4a [options]

Examples:
  # Basic usage — will prompt for anything not supplied
  python summarize_meeting.py meeting.m4a

  # Fully specified on the command line
  python summarize_meeting.py meeting.m4a \\
    --meeting-name "202603 Meeting Minutes" \\
    --names "Jane Smith, John Doe, Alice Johnson" \\
    --context "Example Community Committee, March 18 2026, meeting starts at 6 PM"

  # Re-run just the minutes (skip re-transcribing)
  python summarize_meeting.py meeting.m4a --skip-transcribe meeting.txt

  # Transcribe only, no LLM
  python summarize_meeting.py meeting.m4a --transcript-only

  # Use a local Ollama model instead of Claude API
  python summarize_meeting.py meeting.m4a --backend ollama --ollama-model qwen2.5:latest

  # Use the Claude Code CLI (uses your Pro subscription, no API key needed)
  python summarize_meeting.py meeting.m4a --backend claude-cli

  # Skip all prompts and use saved config defaults
  python summarize_meeting.py meeting.m4a -y

Environment variables:
  ANTHROPIC_API_KEY   Required for --backend claude-api (default)
  WHISPER_URL         Whisper API server URL (default: http://localhost:8000)
  OLLAMA_URL          Ollama server URL (default: http://localhost:11434)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import requests
import anthropic


# ── Configuration ──────────────────────────────────────────────────────────────
# config.json lives next to this script and is gitignored.
CONFIG_PATH = Path(__file__).parent / "config.json"

WHISPER_URL  = os.getenv("WHISPER_URL", "http://localhost:8000")
OLLAMA_URL    = os.getenv("OLLAMA_URL",   "http://localhost:11434")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL",  "claude-sonnet-4-6")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "gemma4:e4b")

# Hallucination detection — sliding window across consecutive segments
WINDOW_SIZE        = 20    # segments to check at once
REPEAT_THRESHOLD   = 0.60  # fraction of window that must share the same text
# Per-segment fallbacks (faster-whisper defaults)
COMPRESSION_RATIO_THRESHOLD = 2.4
NO_SPEECH_PROB_THRESHOLD    = 0.60
AVG_LOGPROB_THRESHOLD       = -1.0

# How far before the first bad segment to start the retry clip (for context)
RETRY_OVERLAP_SECONDS = 30.0


SYSTEM_PROMPT = """\
You are a professional meeting secretary. You will be given a raw audio \
transcription of a committee meeting and must produce formal meeting minutes \
following Robert's Rules of Order.

Format the output as Markdown with the following sections \
(omit any section for which no content exists in the transcript):

1. Meeting header (organization, date, time called to order, location)
2. Members / Attendees Present
3. Treasurer's Report
4. Secretary's Report
5. Old Business
6. New Business
7. Membership Changes (new members, departing members)
8. Action Items — render as a Markdown table: | Item | Owner | Due |
9. Next Meeting
10. Meeting Adjournment (time adjourned)
11. Signature / approval lines (Submitted by, Approved on)

Guidelines:
- Exclude personal conversations unrelated to committee business \
(health, personal anecdotes, off-topic chat).
- The conversation may jump between topics — carefully extract and \
organize the important information.
- Capture all formal motions: who moved, who seconded, and the vote outcome.
- Use professional, neutral language throughout.
- When speaker identity is unclear, omit the name rather than guess.
- Do not invent information not present in the transcript.
"""


# ── Audio helpers ───────────────────────────────────────────────────────────────

MIME_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}

def mime_for(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def convert_to_mp3(input_path: Path, cleanup: list) -> Path:
    """Convert audio to MP3. Registers the output in cleanup for later deletion."""
    output_path = input_path.with_suffix(".mp3")
    print(f"  Converting {input_path.name} → {output_path.name} ...")
    result = subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-codec:a", "libmp3lame",
         "-q:a", "2", str(output_path), "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"ffmpeg error:\n{result.stderr}")
    cleanup.append(output_path)
    return output_path


def extract_from(source: Path, start_seconds: float, dest: Path) -> Path:
    """Extract audio from start_seconds to end of file."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(source), "-ss", str(start_seconds),
         "-c", "copy", str(dest), "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # -c copy can fail for some containers; re-encode as fallback
        subprocess.run(
            ["ffmpeg", "-i", str(source), "-ss", str(start_seconds),
             str(dest), "-y"],
            capture_output=True, check=True,
        )
    return dest


def safe_filename(name: str) -> str:
    """Strip characters that are invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


# ── Transcription ───────────────────────────────────────────────────────────────

def ensure_model_downloaded(model: str, whisper_url: str = None) -> None:
    url = whisper_url or WHISPER_URL
    encoded = model.replace("/", "%2F")
    print(f"  Pulling model {model} from Whisper API server ...")
    resp = requests.post(f"{url}/v1/models/{encoded}", timeout=600)
    resp.raise_for_status()
    print("  Model ready.")


def unload_whisper_model(model: str, whisper_url: str = None) -> None:
    """Unload the Whisper model from the API server memory."""
    url = whisper_url or WHISPER_URL
    encoded = model.replace("/", "%2F")
    try:
        resp = requests.delete(f"{url}/api/ps/{encoded}", timeout=30)
        if resp.status_code in (200, 204):
            print("  Whisper model unloaded from server memory.")
        else:
            print(f"  Note: Whisper unload returned {resp.status_code}.")
    except requests.RequestException as e:
        print(f"  Warning: could not unload Whisper model: {e}")


def unload_ollama_model(model: str, ollama_url: str = None) -> None:
    """Unload an Ollama model from memory by sending a keep_alive=0 request."""
    url = ollama_url or OLLAMA_URL
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        if resp.status_code == 200:
            print(f"  Ollama model '{model}' unloaded from memory.")
        else:
            print(f"  Note: Ollama unload returned {resp.status_code}.")
    except requests.RequestException as e:
        print(f"  Warning: could not unload Ollama model: {e}")


def transcribe(audio_path: Path, cleanup: list,
               prompt: str = "", hotwords: str = "",
               emit_callback=None,
               whisper_url: str = None,
               whisper_model: str = None) -> dict:
    """Send audio to the Whisper API server and return the verbose_json dict."""
    url   = whisper_url   or WHISPER_URL
    model = whisper_model or WHISPER_MODEL

    def _log(msg):
        print(f"  {msg}")
        if emit_callback:
            try:
                emit_callback("log", message=msg, level="info")
            except Exception:
                pass

    size_kb = audio_path.stat().st_size // 1024
    _log(f"Sending {audio_path.name} ({size_kb:,} KB) to Whisper API ({model})...")

    data = {
        "model":           model,
        "response_format": "verbose_json",
        "vad_filter":      "true",
    }
    if prompt:
        data["prompt"] = prompt
    if hotwords:
        data["hotwords"] = hotwords

    with open(audio_path, "rb") as fh:
        resp = requests.post(
            f"{url}/v1/audio/transcriptions",
            files={"file": (audio_path.name, fh, mime_for(audio_path))},
            data=data,
            timeout=3600,
        )

    if resp.status_code == 404 and "not installed" in resp.text.lower():
        ensure_model_downloaded(model, whisper_url=url)
        return transcribe(audio_path, cleanup, prompt, hotwords,
                          emit_callback, whisper_url=url, whisper_model=model)

    if resp.status_code == 500 and audio_path.suffix.lower() != ".mp3":
        _log("Whisper API error — retrying with MP3 conversion ...")
        return transcribe(convert_to_mp3(audio_path, cleanup), cleanup,
                          prompt, hotwords, emit_callback,
                          whisper_url=url, whisper_model=model)

    if resp.status_code != 200:
        raise RuntimeError(f"Transcription failed ({resp.status_code}):\n{resp.text}")

    return resp.json()


# ── Hallucination detection ─────────────────────────────────────────────────────

def _segment_is_bad(seg: dict) -> bool:
    """Per-segment quality checks using faster-whisper's own metrics."""
    if seg.get("no_speech_prob",   0) > NO_SPEECH_PROB_THRESHOLD:
        return True
    if seg.get("compression_ratio", 0) > COMPRESSION_RATIO_THRESHOLD:
        return True
    if seg.get("avg_logprob",       0) < AVG_LOGPROB_THRESHOLD:
        return True
    # Within-segment word repetition (e.g. "Rubik's Cubes" × 30)
    words = seg.get("text", "").lower().split()
    if len(words) >= 6:
        top_count = Counter(words).most_common(1)[0][1]
        if top_count / len(words) > 0.60:
            return True
    return False


def find_hallucinated_ranges(segments: list) -> list:
    """
    Return a list of (start_sec, end_sec) tuples covering hallucinated audio.

    Uses a sliding window: if REPEAT_THRESHOLD of any WINDOW_SIZE consecutive
    segments share the same normalised text, the whole window is flagged.
    Also flags individual segments that fail quality metrics.
    """
    bad_indices = set()

    # Sliding-window repetition check
    texts = [s.get("text", "").strip().lower() for s in segments]
    for i in range(len(segments) - WINDOW_SIZE + 1):
        window = texts[i : i + WINDOW_SIZE]
        top_text, top_count = Counter(window).most_common(1)[0]
        if top_count / WINDOW_SIZE >= REPEAT_THRESHOLD:
            bad_indices.update(range(i, i + WINDOW_SIZE))

    # Per-segment metric check
    for i, seg in enumerate(segments):
        if _segment_is_bad(seg):
            bad_indices.add(i)

    if not bad_indices:
        return []

    # Convert index set → sorted list of segments → merged time ranges
    bad_segs = [segments[i] for i in sorted(bad_indices)]
    ranges = [(bad_segs[0]["start"], bad_segs[0]["end"])]
    for seg in bad_segs[1:]:
        if seg["start"] <= ranges[-1][1] + 2.0:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], seg["end"]))
        else:
            ranges.append((seg["start"], seg["end"]))
    return ranges


def _anchor_text(segments: list, before_time: float) -> str:
    """Last ~200 chars of clean transcript text before before_time."""
    good = [s for s in segments
            if s["end"] <= before_time and not _segment_is_bad(s)]
    if not good:
        return ""
    tail = " ".join(s.get("text", "").strip() for s in good[-5:])
    return tail[-200:]


# ── Clean transcript assembly ───────────────────────────────────────────────────

def build_clean_transcript(audio_path: Path, result: dict,
                            cleanup: list, hotwords: str = "",
                            emit_callback=None,
                            whisper_url: str = None,
                            whisper_model: str = None) -> str:
    """
    Detect hallucinated ranges, retry them with anchor prompts, and return
    a single clean transcript string.
    """
    def _log(msg):
        print(f"  {msg}")
        if emit_callback:
            try:
                emit_callback("log", message=msg, level="info")
            except Exception:
                pass

    segments   = result.get("segments", [])
    bad_ranges = find_hallucinated_ranges(segments)

    if not bad_ranges:
        _log("No hallucinations detected.")
        return " ".join(s.get("text", "").strip() for s in segments)

    _log(f"Detected {len(bad_ranges)} hallucinated range(s) — retrying...")
    for s, e in bad_ranges:
        _log(f"  Bad range: {int(s)//60:02d}:{int(s)%60:02d} → "
             f"{int(e)//60:02d}:{int(e)%60:02d}")

    # Start with all clean segments
    clean = [s for s in segments
             if not any(r[0] <= s["start"] < r[1] for r in bad_ranges)]

    for bad_start, bad_end in bad_ranges:
        anchor         = _anchor_text(segments, bad_start)
        retry_from     = max(0.0, bad_start - RETRY_OVERLAP_SECONDS)
        anchor_preview = anchor[-80:].replace("\n", " ")
        _log(f"Retrying from {retry_from/60:.1f} min (anchor: \"{anchor_preview}\")")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            extract_from(audio_path, retry_from, tmp_path)
            retry_result = transcribe(tmp_path, cleanup, prompt=anchor,
                                      hotwords=hotwords,
                                      emit_callback=emit_callback,
                                      whisper_url=whisper_url,
                                      whisper_model=whisper_model)
            retry_segs   = retry_result.get("segments", [])

            # Shift timestamps back to the original timeline
            for seg in retry_segs:
                seg["start"] += retry_from
                seg["end"]   += retry_from

            # Accept only clean segments that fall in the bad range
            for seg in retry_segs:
                if (bad_start - RETRY_OVERLAP_SECONDS
                        <= seg["start"] < bad_end
                        and not _segment_is_bad(seg)):
                    clean.append(seg)
        finally:
            tmp_path.unlink(missing_ok=True)

    clean.sort(key=lambda s: s["start"])
    return " ".join(s.get("text", "").strip() for s in clean)


# ── Minutes generation ──────────────────────────────────────────────────────────

def _build_user_content(transcript: str, names: str, context: str,
                        prev_minutes: str = "", roster: str = "") -> str:
    parts = []
    if context:
        parts.append(f"Meeting context: {context}")
    if names:
        parts.append(f"Known attendee/vendor names (use for correct spelling): {names}")
    if roster:
        parts.append(
            "Club member roster (name and call sign \u2014 use for attendance "
            "identification and correct spelling of names and call signs):\n"
            + roster
        )
    if prev_minutes:
        parts.append(
            "Previous meeting minutes (use for context on old/new business, "
            "member names, and continuity — do not copy verbatim):\n"
            + prev_minutes
        )
    parts.append(transcript)
    return "\n\n".join(parts)


def generate_minutes_claude_api(transcript: str, names: str, context: str,
                                emit_callback=None, prev_minutes: str = "",
                                roster: str = "") -> str:
    """Generate minutes using the Anthropic API (requires ANTHROPIC_API_KEY)."""
    if emit_callback:
        try:
            emit_callback("log", message=f"Sending transcript to Claude API ({CLAUDE_MODEL})...", level="info")
        except Exception:
            pass
    client = anthropic.Anthropic()
    user_content = _build_user_content(transcript, names, context, prev_minutes, roster)
    print(f"  Sending transcript to Claude API ({CLAUDE_MODEL}) ...")
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text


def generate_minutes_ollama(transcript: str, names: str, context: str,
                            model: str, emit_callback=None,
                            ollama_url: str = None, prev_minutes: str = "",
                            roster: str = "", cancel_event=None) -> str:
    """Generate minutes using a local Ollama model (streaming for tok/s stats and cancellation)."""
    url = ollama_url or OLLAMA_URL
    if emit_callback:
        try:
            emit_callback("log", message=f"Sending transcript to Ollama ({model})...", level="info")
        except Exception:
            pass
    user_content = _build_user_content(transcript, names, context, prev_minutes, roster)
    print(f"  Sending transcript to Ollama ({model}) ...")
    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        },
        stream=True,
        timeout=3600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama request failed ({resp.status_code}):\n{resp.text[:500]}")

    chunks: list[str] = []
    eval_count = 0
    eval_duration_ns = 0

    for line in resp.iter_lines():
        if cancel_event and cancel_event.is_set():
            resp.close()
            return ""  # caller will check cancel_event and discard result
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except Exception:
            continue
        content = chunk.get("message", {}).get("content", "")
        if content:
            chunks.append(content)
        if chunk.get("done"):
            eval_count        = chunk.get("eval_count", 0)
            eval_duration_ns  = chunk.get("eval_duration", 0)
            break

    if eval_count and eval_duration_ns:
        tps = eval_count / (eval_duration_ns / 1e9)
        msg = f"Ollama: {eval_count:,} tokens at {tps:.1f} tok/s"
        print(f"  {msg}")
        if emit_callback:
            try:
                emit_callback("log", message=msg, level="info")
            except Exception:
                pass

    return "".join(chunks)


def generate_minutes_claude_cli(transcript: str, names: str,
                                context: str, prev_minutes: str = "",
                                roster: str = "") -> str:
    """Generate minutes by piping the transcript through the `claude` CLI.

    Uses your Claude Pro subscription \u2014 no API key required.
    Requires `claude` to be installed and authenticated.
    """
    user_content = _build_user_content(transcript, names, context, prev_minutes, roster)
    full_prompt  = f"{SYSTEM_PROMPT}\n\n{user_content}"
    print("  Sending transcript to claude CLI ...")
    result = subprocess.run(
        ["claude", "-p", full_prompt],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI error:\n{result.stderr}")
    return result.stdout


def generate_minutes(transcript: str, names: str = "", context: str = "",
                     backend: str = "claude-api",
                     ollama_model: str = "",
                     emit_callback=None,
                     ollama_url: str = None,
                     prev_minutes: str = "",
                     roster: str = "",
                     cancel_event=None) -> str:
    if backend == "claude-api":
        return generate_minutes_claude_api(transcript, names, context,
                                           emit_callback=emit_callback,
                                           prev_minutes=prev_minutes,
                                           roster=roster)
    elif backend == "ollama":
        return generate_minutes_ollama(transcript, names, context,
                                       ollama_model or OLLAMA_MODEL,
                                       emit_callback=emit_callback,
                                       ollama_url=ollama_url,
                                       prev_minutes=prev_minutes,
                                       roster=roster,
                                       cancel_event=cancel_event)
    elif backend == "claude-cli":
        return generate_minutes_claude_cli(transcript, names, context,
                                           prev_minutes=prev_minutes,
                                           roster=roster)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")


# ── Config file ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load saved config from config.json, return empty dict if not found."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(cfg: dict) -> None:
    """Persist config keys to config.json next to this script."""
    existing = load_config()
    existing.update({k: v for k, v in cfg.items() if v})
    CONFIG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  Config saved → {CONFIG_PATH.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting recording and generate minutes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, nargs="?",
                        help="Audio file (m4a, mp3, wav, …)")
    parser.add_argument("--meeting-name", "-m", default=None,
                        help="Meeting name used for output filenames, "
                             "e.g. '202604 OHD Minutes'")
    parser.add_argument("--output", "-o", type=Path,
                        help="Override output markdown path entirely")
    parser.add_argument("--names", "-n", default=None,
                        help="Comma-separated attendee/vendor names for "
                             "spelling hints passed to both Whisper and LLM")
    parser.add_argument("--context", "-c", default=None,
                        help="Free-text context for the LLM "
                             "(org name, date, start time, etc.)")
    parser.add_argument("--vocabulary", default="",
                        help="Comma-separated terms (names, jargon, place names) "
                             "to improve Whisper transcription accuracy")
    parser.add_argument("--transcript-only", action="store_true",
                        help="Transcribe only; skip minutes generation")
    parser.add_argument("--skip-transcribe", type=Path, metavar="TXT",
                        help="Use an existing .txt transcript and go straight "
                             "to minutes generation")
    parser.add_argument("--backend", default=None,
                        choices=["claude-api", "ollama", "claude-cli"],
                        help="LLM backend for minutes generation "
                             "(default: claude-api)")
    parser.add_argument("--ollama-model", default="",
                        help=f"Ollama model to use (default: {OLLAMA_MODEL}). "
                             "Only used with --backend ollama")
    parser.add_argument("--whisper-model", default=None,
                        help=f"Whisper model ID (default: {WHISPER_MODEL}). "
                             "e.g. Systran/faster-distil-whisper-large-v3")
    parser.add_argument("--whisper-url", default=None,
                        help=f"Whisper API server URL (default: {WHISPER_URL})")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip interactive prompts and use saved defaults")
    args = parser.parse_args()

    # ── Load saved config (used as defaults for prompts) ────────────────────
    cfg = load_config()

    def prompt(question: str, default: str = "") -> str:
        """Prompt the user, showing the default in brackets. Skip if -y."""
        if args.yes:
            return default
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        return answer or default

    # ── Interactive prompts for missing arguments ────────────────────────────

    # Whisper model
    global WHISPER_MODEL
    if args.whisper_model:
        WHISPER_MODEL = args.whisper_model
    elif not os.getenv("WHISPER_MODEL"):
        WHISPER_MODEL = cfg.get("whisper_model", WHISPER_MODEL)

    # Whisper API URL
    global WHISPER_URL
    if args.whisper_url:
        WHISPER_URL = args.whisper_url
    elif not os.getenv("WHISPER_URL"):
        WHISPER_URL = prompt("Whisper API server URL",
                             cfg.get("whisper_url", WHISPER_URL))

    # Input file
    if not args.skip_transcribe and args.input is None:
        raw = prompt("Audio file path").strip('"').strip("'")
        if not raw:
            sys.exit("Error: no input file provided.")
        args.input = Path(raw)

    if not args.skip_transcribe and not args.input.exists():
        sys.exit(f"Error: {args.input} not found.")

    # Meeting name
    if args.meeting_name is None:
        args.meeting_name = prompt(
            "Meeting name (used for output filename, or leave blank to use audio filename)",
            cfg.get("meeting_name", ""),
        )

    # Names
    if args.names is None:
        args.names = prompt(
            "Attendee/vendor names for spelling hints (comma-separated, or leave blank)",
            cfg.get("names", ""),
        )

    # Context
    if args.context is None:
        args.context = prompt(
            "Meeting context for the LLM (org, date, start time — or leave blank)",
            cfg.get("context", ""),
        )

    # Backend
    if args.backend is None and not args.transcript_only:
        backend_choice = prompt(
            "LLM backend [claude-api / ollama / claude-cli]",
            cfg.get("backend", "claude-api"),
        ).lower()
        if backend_choice not in ("claude-api", "ollama", "claude-cli"):
            print(f"  Unknown backend '{backend_choice}', defaulting to claude-api.")
            backend_choice = "claude-api"
        args.backend = backend_choice
    elif args.backend is None:
        args.backend = "claude-api"

    # ── Save config for next run ─────────────────────────────────────────────
    if not args.yes:
        save_config({
            "whisper_url": WHISPER_URL,
            "backend":      args.backend,
            "names":        args.names,
            "context":      args.context,
            "meeting_name": args.meeting_name,
            "ollama_model": args.ollama_model or OLLAMA_MODEL,
            "whisper_model": WHISPER_MODEL,
        })

    # ── Resolve output paths ─────────────────────────────────────────────────
    base_dir = args.input.parent if args.input else Path(".")
    if args.output:
        output_path     = args.output
        transcript_path = output_path.with_suffix(".txt")
    elif args.meeting_name:
        stem            = safe_filename(args.meeting_name)
        output_path     = base_dir / f"{stem}.md"
        transcript_path = base_dir / f"{stem}.txt"
    else:
        output_path     = args.input.with_suffix(".md")
        transcript_path = args.input.with_suffix(".txt")

    # Tracks temp files created during this run (converted MP3, etc.)
    cleanup: list[Path] = []
    hotwords = args.vocabulary or args.names

    try:
        # ── Step 1: Transcribe ──────────────────────────────────────────────
        if args.skip_transcribe:
            print(f"Using existing transcript: {args.skip_transcribe}")
            transcript = args.skip_transcribe.read_text(encoding="utf-8")
        else:
            print(f"\n[1/2] Transcribing {args.input.name} ...")
            result     = transcribe(args.input, cleanup, hotwords=hotwords)
            transcript = build_clean_transcript(args.input, result,
                                                cleanup, hotwords=hotwords)
            transcript_path.write_text(transcript, encoding="utf-8")
            print(f"  Transcript saved → {transcript_path.name}")

            if args.transcript_only:
                print("\nDone (transcript only).")
                return

        # ── Step 2: Generate minutes ────────────────────────────────────────
        print(f"\n[2/2] Generating meeting minutes (backend: {args.backend}) ...")
        minutes = generate_minutes(transcript,
                                   names=args.names,
                                   context=args.context,
                                   backend=args.backend,
                                   ollama_model=args.ollama_model)
        output_path.write_text(minutes, encoding="utf-8")
        print(f"  Meeting minutes saved → {output_path}")

    finally:
        # ── Step 3: Unload models & clean up temp files ─────────────────────
        print(f"\n[3/3] Cleaning up ...")
        if not args.skip_transcribe:
            unload_whisper_model(WHISPER_MODEL)
        if args.backend == "ollama" and not args.transcript_only:
            unload_ollama_model(args.ollama_model or OLLAMA_MODEL)
        for path in cleanup:
            if path.exists():
                path.unlink()
                print(f"  Deleted temp file: {path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
