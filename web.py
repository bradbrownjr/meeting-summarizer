"""Meeting Summarizer — Web UI.

Run:
    python web.py [--port 8082] [--host 0.0.0.0]
"""

import ipaddress
import hmac
import json
import os
import re
import requests
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request


# ── Uploaded-document text extraction ───────────────────────────────────────

def _extract_uploaded_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded .txt, .md, .csv, .pdf, or .docx file."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        return raw.decode("utf-8", errors="replace").strip()
    if suffix == ".csv":
        return raw.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        import io
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    if suffix == ".docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    # Fallback for unexpected types
    return raw.decode("utf-8", errors="replace").strip()


# ── Data directory (orgs.json lives here) ────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ORGS_FILE = DATA_DIR / "orgs.json"

TEMPLATES_DIR = Path(os.environ.get(
    "TEMPLATES_DIR",
    Path.home() / "meeting-templates",
))

_env_backend = os.environ.get("DEFAULT_BACKEND", "")
if not _env_backend:
    # Auto-detect: prefer ollama if no Claude API key is configured
    _env_backend = "claude-api" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"
DEFAULT_BACKEND      = _env_backend
DEFAULT_OLLAMA_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "qwen3.5:9b")

SERVERS_FILE = DATA_DIR / "servers.json"


def _default_servers() -> dict:
    """Seed one server from the container's env vars."""
    return {
        "default": {
            "label":         "Home Server",
            "whisper_url":   os.environ.get("WHISPER_URL",   "http://localhost:8000"),
            "whisper_model": os.environ.get("WHISPER_MODEL", "Systran/faster-whisper-large-v3"),
            "ollama_url":    os.environ.get("OLLAMA_URL",    "http://localhost:11434"),
        }
    }


def load_servers() -> dict:
    if SERVERS_FILE.exists():
        try:
            data = json.loads(SERVERS_FILE.read_text(encoding="utf-8"))
            if data:
                return data
        except Exception:
            pass
    servers = _default_servers()
    save_servers(servers)
    return servers


def save_servers(servers: dict):
    SERVERS_FILE.write_text(
        json.dumps(servers, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_orgs() -> dict:
    if ORGS_FILE.exists():
        try:
            return json.loads(ORGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_orgs(orgs: dict):
    ORGS_FILE.write_text(json.dumps(orgs, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def _make_org_id(label: str, existing: dict) -> str:
    """Derive a slug from label; ensure uniqueness."""
    base = re.sub(r"[^\w]", "", label.lower())[:20] or "org"
    slug = base
    i = 2
    while slug in existing:
        slug = f"{base}{i}"
        i += 1
    return slug


# ── SSRF guard ───────────────────────────────────────────────────────────────
# The health-check endpoints proxy requests on behalf of the browser (needed
# because direct browser → local-service calls are blocked by CORS/mixed-content).
# Private/LAN IPs are intentionally allowed — they are the target services.
# We only block link-local (cloud metadata) ranges and non-HTTP(S) schemes.

_BLOCKED_NETS = [
    ipaddress.ip_network("169.254.0.0/16"),   # AWS/GCP/Azure IMDS (link-local)
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]
_MAX_URL_LEN = 512
API_AUTH_TOKEN = os.environ.get("MEETING_SUMMARIZER_API_TOKEN", "").strip()
ENFORCE_ORIGIN_CHECK = os.environ.get("ENFORCE_ORIGIN_CHECK", "1").strip().lower() not in (
    "0", "false", "no"
)
_ALLOWED_AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".wma", ".opus", ".webm"
}


def _is_safe_service_url(url: str) -> tuple[bool, str]:
    """Return (True, "") if url is a well-formed http(s) URL whose hostname
    does not resolve to a cloud-metadata (link-local) address."""
    if not url:
        return False, "empty URL"
    if len(url) > _MAX_URL_LEN:
        return False, "URL too long"
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "could not parse URL"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed"
    hostname = parsed.hostname
    if not hostname:
        return False, "no hostname"
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(hostname))
    except Exception:
        return False, f"could not resolve '{hostname}'"
    for net in _BLOCKED_NETS:
        if addr in net:
            return False, f"address {addr} is blocked"
    return True, ""


def _is_same_origin_request() -> bool:
    """Accept requests with matching Origin/Referer. If neither header is
    present (non-browser clients), allow the request."""
    expected = request.host_url.rstrip("/")
    origin = request.headers.get("Origin", "").rstrip("/")
    if origin:
        return origin == expected
    referer = request.headers.get("Referer", "")
    if referer:
        parsed = urllib.parse.urlparse(referer)
        ref_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return ref_origin == expected
    return True


# ── Flask app ─────────────────────────────────────────────────────────────────

_template_dir = os.path.join(os.path.dirname(__file__), "templates")
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, template_folder=_template_dir, static_folder=_static_dir)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


@app.before_request
def _api_security_guard():
    """Basic API hardening for browser-facing endpoints."""
    if not request.path.startswith("/api/"):
        return None
    if request.method not in ("POST", "PUT", "DELETE"):
        return None
    if ENFORCE_ORIGIN_CHECK and not _is_same_origin_request():
        return jsonify({"error": "Origin check failed"}), 403
    if API_AUTH_TOKEN:
        provided = request.headers.get("X-API-Token", "")
        if not provided or not hmac.compare_digest(provided, API_AUTH_TOKEN):
            return jsonify({"error": "Invalid API token"}), 401
    return None


@app.after_request
def _set_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'",
    )
    return resp

JOBS_DIR = Path(os.environ.get("UPLOAD_DIR", DATA_DIR / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_RETENTION_DAYS = max(1, int(os.environ.get("JOB_RETENTION_DAYS", "30")))


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _job_metadata_path(job_id: str) -> Path:
    return _job_dir(job_id) / "metadata.json"


def _job_transcript_path(job_id: str) -> Path:
    return _job_dir(job_id) / "transcript.txt"


def _job_minutes_path(job_id: str) -> Path:
    return _job_dir(job_id) / "minutes.md"


def _job_log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.log"


def _job_audio_path(job_id: str, audio_name: str) -> Path:
    return _job_dir(job_id) / audio_name


def _job_associated_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "associated"


def _default_progress(status: str = "queued") -> dict:
    label = {
        "queued": "Waiting in queue",
        "transcribing": "Transcribing audio",
        "generating": "Generating minutes",
        "done": "Complete",
        "error": "Failed",
        "cancelled": "Cancelled",
    }.get(status, "Processing")
    percent = {
        "queued": 0,
        "transcribing": 8,
        "generating": 92,
        "done": 100,
        "error": 100,
        "cancelled": 100,
    }.get(status, 0)
    return {
        "phase": status,
        "percent": percent,
        "label": label,
        "detail": "",
    }


def _probe_audio_duration_seconds(path: Path) -> float | None:
    """Return audio duration in seconds using ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value = float((result.stdout or "").strip())
        if value > 0:
            return value
    except Exception:
        return None
    return None


def _serializable_job(job_id: str, job: dict) -> dict:
    return {
        "job_id": job_id,
        "status": job["status"],
        "logs": list(job["logs"]),
        "meeting_name": job.get("meeting_name", ""),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "audio_name": job.get("audio_name", ""),
        "minutes_file": job.get("minutes_file"),
        "transcript_file": job.get("transcript_file"),
        "associated_files": job.get("associated_files", []),
        "progress": job.get("progress", _default_progress(job.get("status", "queued"))),
        "audio_duration_sec": job.get("audio_duration_sec"),
    }


def _persist_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        payload = _serializable_job(job_id, job)
        logs = list(job["logs"])
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _job_metadata_path(job_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _job_log_path(job_id).write_text(
        "\n".join(logs),
        encoding="utf-8",
    )


def _read_job_artifact(job_id: str, artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    path = _job_dir(job_id) / artifact_name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _new_runtime_job(created_at: float | None = None) -> dict:
    return {
        "status": "queued",
        "logs": [],
        "result_md": None,
        "transcript": None,
        "meeting_name": "",
        "error": None,
        "subscribers": [],
        "created_at": created_at or time.time(),
        "cancel_event": threading.Event(),
        "audio_name": "",
        "minutes_file": None,
        "transcript_file": None,
        "associated_files": [],
        "audio_duration_sec": None,
        "progress": _default_progress("queued"),
    }


def _load_persisted_jobs():
    for meta_path in sorted(JOBS_DIR.glob("*/metadata.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        job_id = data.get("job_id") or meta_path.parent.name
        job = _new_runtime_job(created_at=data.get("created_at"))
        job.update({
            "status": data.get("status", "error"),
            "logs": data.get("logs", []),
            "meeting_name": data.get("meeting_name", ""),
            "error": data.get("error"),
            "audio_name": data.get("audio_name", ""),
            "minutes_file": data.get("minutes_file"),
            "transcript_file": data.get("transcript_file"),
            "associated_files": data.get("associated_files", []),
            "audio_duration_sec": data.get("audio_duration_sec"),
            "progress": data.get("progress") or _default_progress(data.get("status", "queued")),
        })
        if job["minutes_file"]:
            job["result_md"] = _read_job_artifact(job_id, job["minutes_file"])
        if job["transcript_file"]:
            job["transcript"] = _read_job_artifact(job_id, job["transcript_file"])
        if job["status"] in ("queued", "transcribing", "generating"):
            message = "Job interrupted by server restart or container update."
            job["status"] = "error"
            job["error"] = message
            job["logs"].append(message)
            job["progress"] = _default_progress("error")
        with _jobs_lock:
            _jobs[job_id] = job
        _persist_job(job_id)


def _delete_job(job_id: str):
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    with _jobs_lock:
        _jobs.pop(job_id, None)


# ── Job state ─────────────────────────────────────────────────────────────────

import queue as _queue

_jobs: dict = {}
_jobs_lock = threading.Lock()
_run_queue: _queue.Queue = _queue.Queue()  # sequential job execution


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = _new_runtime_job()
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _persist_job(job_id)
    return job_id


def _push(job_id: str, event: dict):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return
    etype = event.get("type")
    if etype == "log":
        message = event.get("data", {}).get("message", "")
        job["logs"].append(message)
        # Keep transcription progress moving based on common log milestones.
        if job.get("status") == "transcribing":
            p = dict(job.get("progress") or _default_progress("transcribing"))
            current = int(p.get("percent", 8) or 8)
            if message.startswith("Sending "):
                current = min(80, max(current + 6, 14))
            elif message.startswith("Detected ") and "hallucinated range" in message:
                current = max(current, 72)
            elif message.startswith("Retrying from "):
                current = min(88, current + 3)
            elif message.startswith("No hallucinations detected"):
                current = max(current, 88)
            elif message.startswith("Transcript complete"):
                current = max(current, 90)
            p.update({
                "phase": "transcribing",
                "percent": current,
                "label": "Transcribing audio",
            })
            job["progress"] = p
    elif etype == "status":
        status = event.get("data", {}).get("status", job["status"])
        job["status"] = status
        p = dict(job.get("progress") or _default_progress(status))
        if status == "transcribing":
            p.update({"phase": "transcribing", "percent": max(8, int(p.get("percent", 0))), "label": "Transcribing audio"})
        elif status == "generating":
            p.update({"phase": "generating", "percent": max(92, int(p.get("percent", 0))), "label": "Generating minutes"})
        elif status in ("done", "error", "cancelled"):
            p = _default_progress(status)
        job["progress"] = p
    elif etype == "progress":
        incoming = event.get("data", {})
        p = dict(job.get("progress") or _default_progress(job.get("status", "queued")))
        p.update({k: v for k, v in incoming.items() if v is not None})
        if p.get("percent") is not None:
            try:
                p["percent"] = max(0, min(100, int(p["percent"])))
            except Exception:
                p["percent"] = 0
        job["progress"] = p
    elif etype == "result":
        job["result_md"] = event["data"].get("minutes", "")
        job["transcript"] = event["data"].get("transcript", "")
        if job["result_md"]:
            minutes_path = _job_minutes_path(job_id)
            minutes_path.write_text(job["result_md"], encoding="utf-8")
            job["minutes_file"] = minutes_path.name
        if job["transcript"]:
            transcript_path = _job_transcript_path(job_id)
            transcript_path.write_text(job["transcript"], encoding="utf-8")
            job["transcript_file"] = transcript_path.name
        job["status"] = "done"
        job["progress"] = _default_progress("done")
    elif etype == "error":
        job["error"] = event["data"].get("message", "Unknown error")
        job["status"] = "error"
        job["progress"] = _default_progress("error")
    _persist_job(job_id)
    with _jobs_lock:
        subscribers = list(job["subscribers"])
    for q in subscribers:
        try:
            q.put_nowait(event)
        except _queue.Full:
            pass


def _emit(job_id: str, event_type: str, **data):
    _push(job_id, {"type": event_type, "data": data})


# ── Job runner ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, audio_path: Path, org_id: str,
             date_str: str, names: str, backend: str,
             server_id: str = "default", associated_context: str = ""):
    cancel_event = _jobs[job_id]["cancel_event"]
    from summarize_meeting import (
        transcribe, build_clean_transcript, generate_minutes,
        unload_whisper_model, unload_ollama_model,
        WHISPER_MODEL, OLLAMA_MODEL,
    )

    def emit_cb(event_type, **data):
        _emit(job_id, event_type, **data)

    def log(msg: str, level: str = "info"):
        print(f"  [{job_id[:8]}] {msg}")
        _emit(job_id, "log", message=msg, level=level)

    def progress(percent: int, label: str, detail: str = "", phase: str | None = None):
        _emit(
            job_id,
            "progress",
            percent=max(0, min(100, int(percent))),
            label=label,
            detail=detail,
            phase=phase or _jobs.get(job_id, {}).get("status", "queued"),
        )

    cleanup: list = []

    # Resolve server config
    server = load_servers().get(server_id) or load_servers().get("default") or {}
    _whisper_url   = server.get("whisper_url")   or None
    _whisper_model = server.get("whisper_model") or None
    _ollama_url    = server.get("ollama_url")    or None
    _ollama_model  = server.get("ollama_model")  or None

    try:
        _emit(job_id, "status", status="transcribing")

        org = load_orgs().get(org_id, {})
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            month_year = dt.strftime("%B %Y")
            formatted_date = dt.strftime("%B %d, %Y")
        except ValueError:
            month_year = date_str
            formatted_date = date_str

        try:
            context = org.get("context_template", "").format(date=formatted_date)
        except KeyError:
            context = org.get("context_template", "")
        try:
            meeting_name = org.get("name_template", "Meeting \u2013 {month_year}").format(
                month_year=month_year
            )
        except KeyError:
            meeting_name = org.get("name_template", "Meeting \u2013 {month_year}")
        with _jobs_lock:
            _jobs[job_id]["meeting_name"] = meeting_name

        audio_duration = _probe_audio_duration_seconds(audio_path)
        with _jobs_lock:
            _jobs[job_id]["audio_duration_sec"] = audio_duration
        _persist_job(job_id)

        # Merge org vocabulary + roster entries with any names entered at submit time
        org_vocab  = org.get("vocabulary", "")
        org_roster = org.get("roster", "")
        # Flatten roster lines into comma-separated hotword tokens for Whisper
        roster_vocab = ", ".join(
            token
            for line in org_roster.splitlines()
            for token in [line.strip()] if token
        )
        combined_vocab = ", ".join(filter(None, [org_vocab, roster_vocab, names]))

        log(f"Starting: {meeting_name}")
        log(f"Audio: {audio_path.name}  ({audio_path.stat().st_size // 1024:,} KB)")
        if audio_duration:
            mins = int(audio_duration) // 60
            secs = int(audio_duration) % 60
            progress(8, "Transcribing audio", detail=f"Audio length {mins:02d}:{secs:02d}", phase="transcribing")
        else:
            progress(8, "Transcribing audio", detail="Audio length unknown", phase="transcribing")
        log(f"Whisper model: {_whisper_model or WHISPER_MODEL}")
        if combined_vocab:
            log(f"Vocabulary hints: {combined_vocab[:120]}{'...' if len(combined_vocab) > 120 else ''}")

        result = transcribe(audio_path, cleanup, hotwords=combined_vocab,
                            emit_callback=emit_cb,
                            whisper_url=_whisper_url,
                            whisper_model=_whisper_model)
        progress(70, "Transcribing audio", detail="Initial pass complete", phase="transcribing")
        transcript = build_clean_transcript(audio_path, result, cleanup,
                                            hotwords=combined_vocab,
                                            emit_callback=emit_cb,
                                            whisper_url=_whisper_url,
                                            whisper_model=_whisper_model)
        progress(90, "Transcribing audio", detail="Transcript ready", phase="transcribing")

        word_count = len(transcript.split())
        log(f"Transcript complete \u2014 {word_count:,} words")

        if backend == "transcript_only":
            _emit(job_id, "result", minutes="", transcript=transcript, meeting_name=meeting_name)
            progress(100, "Complete", detail="Transcript generated", phase="done")
            log("Done (transcript only.)")
            return

        _emit(job_id, "status", status="generating")
        progress(93, "Generating minutes", detail="Drafting summary", phase="generating")
        log(f"Generating minutes (backend: {backend})\u2026")

        if associated_context:
            log(f"Associated reference material provided ({len(associated_context.split()):,} words) — including as context.")
        if org_roster:
            log(f"Roster: {len([l for l in org_roster.splitlines() if l.strip()])} members loaded.")

        # Check for cancellation before starting LLM generation
        if cancel_event.is_set():
            log("Job cancelled before minutes generation.")
            return

        minutes = generate_minutes(
            transcript,
            names=names,
            context=context,
            backend=backend,
            ollama_model=_ollama_model or OLLAMA_MODEL,
            emit_callback=emit_cb,
            ollama_url=_ollama_url,
            associated_context=associated_context,
            roster=org_roster,
            cancel_event=cancel_event,
        )

        # Check for cancellation after generation (covers Ollama mid-stream cancel)
        if cancel_event.is_set():
            log("Job cancelled.")
            return

        _emit(job_id, "result", minutes=minutes, transcript=transcript, meeting_name=meeting_name)
        progress(100, "Complete", detail="Minutes generated", phase="done")
        log("Done.")

    except Exception as exc:
        if cancel_event.is_set():
            log("Job cancelled.")
            _emit(job_id, "progress", **_default_progress("cancelled"))
        else:
            log(f"Error: {exc}", level="error")
            _emit(job_id, "error", message=str(exc))

    finally:
        try:
            unload_whisper_model(_whisper_model or WHISPER_MODEL, whisper_url=_whisper_url)
        except Exception:
            pass
        if backend == "ollama":
            try:
                unload_ollama_model(_ollama_model or OLLAMA_MODEL, ollama_url=_ollama_url)
            except Exception:
                pass
        for p in cleanup:
            if p.exists():
                p.unlink()
        # Signal SSE stream to close (no-op if cancel endpoint already sent eof)
        _emit(job_id, "eof")


# ── Job queue worker (sequential execution) ──────────────────────────────────

def _queue_worker():
    while True:
        job_id = _run_queue.get()
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                continue
            if job["cancel_event"].is_set():
                with _jobs_lock:
                    job["status"] = "cancelled"
                    job["error"]  = "Cancelled before starting"
                continue
            pending = job.pop("_pending_args", None)
            if pending is None:
                continue
            try:
                _run_job(job_id, *pending["args"], **pending["kwargs"])
            except Exception as exc:
                message = f"Worker crashed before job could complete: {exc}"
                print(f"  [{job_id[:8]}] {message}")
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job:
                        job["logs"].append(message)
                        job["error"] = message
                        job["status"] = "error"
                _emit(job_id, "log", message=message, level="error")
                _emit(job_id, "error", message=message)
                _emit(job_id, "eof")
        finally:
            _run_queue.task_done()


threading.Thread(target=_queue_worker, daemon=True, name="job-worker").start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        default_backend=DEFAULT_BACKEND,
        default_ollama_model=DEFAULT_OLLAMA_MODEL,
        initial_servers=load_servers(),
    )


# ── Org CRUD ──────────────────────────────────────────────────────────────────

@app.route("/api/orgs", methods=["GET"])
def api_orgs_list():
    return jsonify(load_orgs())


@app.route("/api/orgs", methods=["POST"])
def api_orgs_create():
    data = request.get_json(force=True)
    if not data or not data.get("label", "").strip():
        return jsonify({"error": "label is required"}), 400
    orgs = load_orgs()
    org_id = data.get("id") or _make_org_id(data["label"], orgs)
    orgs[org_id] = {
        "label":            data["label"].strip(),
        "short":            data.get("short", "").strip(),
        "context_template": data.get("context_template", "").strip(),
        "name_template":    data.get("name_template", "Meeting \u2013 {month_year}").strip(),
        "vocabulary":       data.get("vocabulary", "").strip(),
        "roster":           data.get("roster", "").strip(),
    }
    save_orgs(orgs)
    return jsonify({"ok": True, "id": org_id, "orgs": orgs})


@app.route("/api/orgs/<org_id>", methods=["PUT"])
def api_orgs_update(org_id: str):
    data = request.get_json(force=True)
    if not data or not data.get("label", "").strip():
        return jsonify({"error": "label is required"}), 400
    orgs = load_orgs()
    if org_id not in orgs:
        return jsonify({"error": "Not found"}), 404
    orgs[org_id] = {
        "label":            data["label"].strip(),
        "short":            data.get("short", "").strip(),
        "context_template": data.get("context_template", "").strip(),
        "name_template":    data.get("name_template", "Meeting \u2013 {month_year}").strip(),
        "vocabulary":       data.get("vocabulary", "").strip(),
        "roster":           data.get("roster", "").strip(),
    }
    save_orgs(orgs)
    return jsonify({"ok": True, "orgs": orgs})


@app.route("/api/orgs/<org_id>", methods=["DELETE"])
def api_orgs_delete(org_id: str):
    orgs = load_orgs()
    if org_id not in orgs:
        return jsonify({"error": "Not found"}), 404
    del orgs[org_id]
    save_orgs(orgs)
    return jsonify({"ok": True, "orgs": orgs})


# ── Server CRUD ───────────────────────────────────────────────────────────────

@app.route("/api/servers", methods=["GET"])
def api_servers_list():
    return jsonify(load_servers())


@app.route("/api/servers", methods=["POST"])
def api_servers_create():
    data = request.get_json(force=True)
    if not data or not data.get("label", "").strip():
        return jsonify({"error": "label is required"}), 400
    for key in ("whisper_url", "ollama_url"):
        url = data.get(key, "").strip()
        if url:
            safe, reason = _is_safe_service_url(url)
            if not safe:
                return jsonify({"error": f"invalid {key}: {reason}"}), 400
    servers = load_servers()
    server_id = data.get("id") or _make_org_id(data["label"], servers)
    servers[server_id] = {
        "label":         data["label"].strip(),
        "whisper_url":   data.get("whisper_url", "").strip(),
        "whisper_model": data.get("whisper_model", "").strip(),
        "ollama_url":    data.get("ollama_url", "").strip(),
        "ollama_model":  data.get("ollama_model", "").strip(),
    }
    save_servers(servers)
    return jsonify({"ok": True, "id": server_id, "servers": servers})


@app.route("/api/servers/<server_id>", methods=["PUT"])
def api_servers_update(server_id: str):
    data = request.get_json(force=True)
    if not data or not data.get("label", "").strip():
        return jsonify({"error": "label is required"}), 400
    for key in ("whisper_url", "ollama_url"):
        url = data.get(key, "").strip()
        if url:
            safe, reason = _is_safe_service_url(url)
            if not safe:
                return jsonify({"error": f"invalid {key}: {reason}"}), 400
    servers = load_servers()
    if server_id not in servers:
        return jsonify({"error": "Not found"}), 404
    servers[server_id] = {
        "label":         data["label"].strip(),
        "whisper_url":   data.get("whisper_url", "").strip(),
        "whisper_model": data.get("whisper_model", "").strip(),
        "ollama_url":    data.get("ollama_url", "").strip(),
        "ollama_model":  data.get("ollama_model", "").strip(),
    }
    save_servers(servers)
    return jsonify({"ok": True, "servers": servers})


@app.route("/api/servers/<server_id>", methods=["DELETE"])
def api_servers_delete(server_id: str):
    servers = load_servers()
    if server_id not in servers:
        return jsonify({"error": "Not found"}), 404
    if len(servers) == 1:
        return jsonify({"error": "Cannot delete the last server"}), 400
    del servers[server_id]
    save_servers(servers)
    return jsonify({"ok": True, "servers": servers})


# ── Org import (YAML) ─────────────────────────────────────────────────────────

def _parse_yaml_orgs(text: str) -> list[dict]:
    """Parse a YAML string into a list of org dicts.

    Supports two formats:
      - A single mapping (one org per file).
      - A list of mappings (multiple orgs in one file).
    """
    data = yaml.safe_load(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _apply_org_entries(entries: list[dict], orgs: dict) -> tuple[list, list]:
    """Merge parsed org entries into orgs dict. Returns (imported_names, errors)."""
    imported, errors = [], []
    for entry in entries:
        label = (entry.get("label") or "").strip()
        if not label:
            errors.append("Entry missing 'label' — skipped")
            continue
        org_id = (entry.get("id") or "").strip() or _make_org_id(label, orgs)
        vocab = entry.get("vocabulary") or ""
        if isinstance(vocab, list):
            vocab = ", ".join(str(v) for v in vocab if v)
        orgs[org_id] = {
            "label":            label,
            "short":            (entry.get("short") or "").strip(),
            "context_template": (entry.get("context_template") or "").strip(),
            "name_template":    (entry.get("name_template") or "Meeting \u2013 {month_year}").strip(),
            "vocabulary":       vocab.strip(),
            "roster":           (entry.get("roster") or "").strip(),
        }
        imported.append(label)
    return imported, errors


@app.route("/api/orgs/<org_id>/roster", methods=["POST"])
def api_orgs_roster_upload(org_id: str):
    """Replace an org's roster from an uploaded plain-text file (one member per line)."""
    orgs = load_orgs()
    if org_id not in orgs:
        return jsonify({"error": "Not found"}), 404
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    if Path(f.filename).suffix.lower() not in (".txt", ".csv"):
        return jsonify({"error": "File must be .txt or .csv"}), 400
    try:
        text = f.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return jsonify({"error": f"Could not read file: {exc}"}), 400
    # Normalise: strip blank lines, deduplicate while preserving order
    seen, lines = set(), []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    roster = "\n".join(lines)
    orgs[org_id]["roster"] = roster
    save_orgs(orgs)
    return jsonify({"ok": True, "count": len(lines), "orgs": orgs})


@app.route("/api/orgs/templates", methods=["GET"])
def api_orgs_templates():
    """List .yaml/.yml files available in TEMPLATES_DIR."""
    if not TEMPLATES_DIR.exists():
        return jsonify({"dir": str(TEMPLATES_DIR), "files": [], "exists": False})
    files = sorted(
        p.name for p in TEMPLATES_DIR.iterdir()
        if p.suffix.lower() in (".yaml", ".yml") and p.is_file()
    )
    return jsonify({"dir": str(TEMPLATES_DIR), "files": files, "exists": True})


@app.route("/api/orgs/import/server", methods=["POST"])
def api_orgs_import_server():
    """Import all YAML files from TEMPLATES_DIR."""
    if not TEMPLATES_DIR.exists():
        return jsonify({"error": f"Templates directory not found: {TEMPLATES_DIR}"}), 404

    orgs = load_orgs()
    all_imported, all_errors = [], []

    yaml_files = sorted(
        p for p in TEMPLATES_DIR.iterdir()
        if p.suffix.lower() in (".yaml", ".yml") and p.is_file()
    )
    if not yaml_files:
        return jsonify({"error": "No .yaml files found in templates directory"}), 404

    for path in yaml_files:
        try:
            entries = _parse_yaml_orgs(path.read_text(encoding="utf-8"))
            imported, errors = _apply_org_entries(entries, orgs)
            all_imported.extend(f"{path.name}: {n}" for n in imported)
            all_errors.extend(f"{path.name}: {e}" for e in errors)
        except Exception as exc:
            all_errors.append(f"{path.name}: {exc}")

    save_orgs(orgs)
    return jsonify({"ok": True, "imported": all_imported, "errors": all_errors, "orgs": orgs})


@app.route("/api/orgs/import/upload", methods=["POST"])
def api_orgs_import_upload():
    """Import orgs from an uploaded YAML file."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    if Path(f.filename).suffix.lower() not in (".yaml", ".yml"):
        return jsonify({"error": "File must be .yaml or .yml"}), 400

    try:
        entries = _parse_yaml_orgs(f.read().decode("utf-8"))
    except Exception as exc:
        return jsonify({"error": f"Failed to parse YAML: {exc}"}), 400

    if not entries:
        return jsonify({"error": "No valid org entries found in file"}), 400

    orgs = load_orgs()
    imported, errors = _apply_org_entries(entries, orgs)
    save_orgs(orgs)
    return jsonify({"ok": True, "imported": imported, "errors": errors, "orgs": orgs})


# ── Job routes ────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def api_run():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify({"ok": False, "error": "No audio file provided"}), 400
    audio_name = Path(audio.filename).name
    if Path(audio_name).suffix.lower() not in _ALLOWED_AUDIO_EXTS:
        return jsonify({"ok": False, "error": "Unsupported audio file type"}), 400

    org_id    = request.form.get("org", "")
    date_str  = request.form.get("date", datetime.today().strftime("%Y-%m-%d"))
    names     = request.form.get("names", "")
    backend   = request.form.get("backend", "claude-api")
    server_id = request.form.get("server_id", "default")

    _VALID_BACKENDS = {"claude-api", "ollama", "claude-cli", "transcript_only"}
    if backend not in _VALID_BACKENDS:
        return jsonify({"ok": False, "error": f"Invalid backend '{backend}'"}), 400
    if len(names) > 2000:
        return jsonify({"ok": False, "error": "names field is too long"}), 400
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid date format; expected YYYY-MM-DD"}), 400

    orgs = load_orgs()
    if org_id and org_id not in orgs:
        return jsonify({"ok": False, "error": "Unknown organization"}), 400
    servers = load_servers()
    if server_id not in servers:
        return jsonify({"ok": False, "error": "Unknown server_id"}), 400

    job_id = _new_job()
    dest = _job_audio_path(job_id, audio_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio.save(dest)

    _ASSOCIATED_MAX_BYTES = 2 * 1024 * 1024  # 2 MB per file
    associated_files = []
    associated_text_parts = []
    for assoc in request.files.getlist("associated_files"):
        if not assoc or not assoc.filename:
            continue
        assoc_name = Path(assoc.filename).name
        assoc_ext = Path(assoc_name).suffix.lower()
        if assoc_ext not in {".txt", ".md", ".pdf", ".docx", ".csv"}:
            return jsonify({"ok": False, "error": f"Unsupported associated file type: {assoc_name}"}), 400
        raw = assoc.read(_ASSOCIATED_MAX_BYTES + 1)
        if len(raw) > _ASSOCIATED_MAX_BYTES:
            return jsonify({"ok": False, "error": f"Associated file exceeds 2 MB limit: {assoc_name}"}), 400
        try:
            text = _extract_uploaded_text(assoc_name, raw)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Could not read associated file '{assoc_name}': {exc}"}), 400
        assoc_dir = _job_associated_dir(job_id)
        assoc_dir.mkdir(parents=True, exist_ok=True)
        stored_name = assoc_name
        target = assoc_dir / stored_name
        counter = 2
        while target.exists():
            stored_name = f"{Path(assoc_name).stem}_{counter}{Path(assoc_name).suffix}"
            target = assoc_dir / stored_name
            counter += 1
        target.write_bytes(raw)
        associated_files.append({"name": assoc_name, "stored_name": stored_name})
        if text:
            associated_text_parts.append(f"Associated file: {assoc_name}\n{text}")

    associated_context = "\n\n".join(associated_text_parts)

    with _jobs_lock:
        _jobs[job_id]["audio_name"] = audio_name
        _jobs[job_id]["associated_files"] = associated_files
        _jobs[job_id]["audio_duration_sec"] = _probe_audio_duration_seconds(dest)
        _jobs[job_id]["progress"] = _default_progress("queued")
    _persist_job(job_id)

    with _jobs_lock:
        _jobs[job_id]["_pending_args"] = {
            "args": (dest, org_id, date_str, names, backend, server_id),
            "kwargs": {"associated_context": associated_context},
        }
    _run_queue.put(job_id)

    return jsonify({"ok": True, "job_id": job_id})


# ── Service health checks ─────────────────────────────────────────────────────

@app.route("/api/servers/<server_id>/vram")
def api_server_vram(server_id: str):
    """Proxy VRAM info from Ollama /api/ps."""
    server = load_servers().get(server_id)
    if not server:
        return jsonify({"error": "Not found"}), 404
    url = (server.get("ollama_url") or "").rstrip("/")
    if not url:
        return jsonify({"ok": False, "error": "No Ollama URL configured"})
    safe, reason = _is_safe_service_url(url)
    if not safe:
        return jsonify({"ok": False, "error": f"invalid url: {reason}"}), 400
    try:
        r = requests.get(f"{url}/api/ps", timeout=5)
        data = r.json()
        return jsonify({"ok": True, "models": data.get("models", [])})
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/health/whisper")
def api_health_whisper():
    """Proxy health-check for a Whisper API server."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "no url"}), 400
    safe, reason = _is_safe_service_url(url)
    if not safe:
        return jsonify({"ok": False, "error": f"invalid url: {reason}"}), 400
    try:
        r = requests.get(f"{url}/v1/models", timeout=5)
        if r.status_code < 500:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": f"HTTP {r.status_code}"}), 200
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 200


@app.route("/api/health/ollama")
def api_health_ollama():
    """Proxy health-check for an Ollama server."""
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "no url"}), 400
    safe, reason = _is_safe_service_url(url)
    if not safe:
        return jsonify({"ok": False, "error": f"invalid url: {reason}"}), 400
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code < 500:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": f"HTTP {r.status_code}"}), 200
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 200


@app.route("/api/jobs", methods=["GET"])
def api_jobs_list():
    """Return a summary list of all known jobs, newest first."""
    with _jobs_lock:
        snapshot = [
            {
                "job_id":       jid,
                "status":       j["status"],
                "meeting_name": j["meeting_name"],
                "created_at":   j["created_at"],
                "error":        j["error"],
                "associated_files": len(j.get("associated_files", [])),
                "progress":     j.get("progress") or _default_progress(j.get("status", "queued")),
                "audio_duration_sec": j.get("audio_duration_sec"),
            }
            for jid, j in _jobs.items()
        ]
    snapshot.sort(key=lambda x: x["created_at"], reverse=True)
    # Add queue positions for waiting jobs
    queued = sorted([j for j in snapshot if j["status"] == "queued"],
                    key=lambda j: j["created_at"])
    pos_map = {j["job_id"]: i + 1 for i, j in enumerate(queued)}
    total_q = len(queued)
    for j in snapshot:
        j["queue_pos"]   = pos_map.get(j["job_id"])
        j["queue_total"] = total_q if j["job_id"] in pos_map else None
    return jsonify(snapshot)


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id: str):
    """Cancel a queued or running job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    if job["status"] in ("done", "error", "cancelled"):
        return jsonify({"error": "Job already finished"}), 400
    job["cancel_event"].set()
    with _jobs_lock:
        job["status"]  = "cancelled"
        job["error"]   = "Cancelled by user"
        job["progress"] = _default_progress("cancelled")
    _persist_job(job_id)
    _emit(job_id, "progress", **_default_progress("cancelled"))
    _emit(job_id, "status", status="cancelled")
    _emit(job_id, "eof")
    return jsonify({"ok": True})


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def api_delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    if job["status"] in ("queued", "transcribing", "generating"):
        return jsonify({"error": "Cannot delete an active job"}), 400
    _delete_job(job_id)
    return jsonify({"ok": True})


@app.route("/api/jobs/<job_id>/stream")
def api_stream(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    sub_q = _queue.Queue(maxsize=500)
    with _jobs_lock:
        replay_logs  = list(job["logs"])
        curr_status  = job["status"]
        curr_progress = dict(job.get("progress") or _default_progress(curr_status))
        result_md    = job["result_md"]
        transcript   = job["transcript"]
        meeting_name = job["meeting_name"]
        error        = job["error"]
        already_done = curr_status in ("done", "error")
        if not already_done:
            job["subscribers"].append(sub_q)

    def generate():
        try:
            # Replay accumulated logs for late joiners
            for msg in replay_logs:
                ev = {"type": "log", "data": {"message": msg, "level": "info"}}
                yield f"data: {json.dumps(ev)}\n\n"
            # Replay current status
            if curr_status not in ("queued", ""):
                ev = {"type": "status", "data": {"status": curr_status}}
                yield f"data: {json.dumps(ev)}\n\n"
            ev = {"type": "progress", "data": curr_progress}
            yield f"data: {json.dumps(ev)}\n\n"
            # If job already finished, send terminal events and close
            if already_done:
                if result_md is not None:
                    ev = {"type": "result", "data": {
                        "minutes": result_md, "transcript": transcript or "",
                        "meeting_name": meeting_name,
                    }}
                    yield f"data: {json.dumps(ev)}\n\n"
                if error:
                    ev = {"type": "error", "data": {"message": error}}
                    yield f"data: {json.dumps(ev)}\n\n"
                yield f"data: {json.dumps({'type': 'eof', 'data': {}})}\n\n"
                return
            # Stream live events from this subscriber's queue
            while True:
                try:
                    event = sub_q.get(timeout=30)
                except _queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("eof", "error"):
                    break
        finally:
            with _jobs_lock:
                try:
                    job["subscribers"].remove(sub_q)
                except ValueError:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        queued_job_ids = [
            jid for jid, entry in _jobs.items()
            if entry["status"] == "queued"
        ]
    if not job:
        return jsonify({"error": "Not found"}), 404
    queued_job_ids.sort(key=lambda jid: _jobs[jid]["created_at"])
    queue_pos = None
    queue_total = None
    if job["status"] == "queued":
        queue_total = len(queued_job_ids)
        try:
            queue_pos = queued_job_ids.index(job_id) + 1
        except ValueError:
            queue_pos = None
    return jsonify({
        "status":       job["status"],
        "meeting_name": job["meeting_name"],
        "logs":         list(job["logs"]),
        "result_md":    job["result_md"] if job["result_md"] is not None else _read_job_artifact(job_id, job.get("minutes_file")),
        "transcript":   job["transcript"] if job["transcript"] is not None else _read_job_artifact(job_id, job.get("transcript_file")),
        "error":        job["error"],
        "queue_pos":    queue_pos,
        "queue_total":  queue_total,
        "associated_files": job.get("associated_files", []),
        "progress":     job.get("progress") or _default_progress(job.get("status", "queued")),
        "audio_duration_sec": job.get("audio_duration_sec"),
    })


@app.route("/api/jobs/<job_id>/download")
def api_download(job_id: str):
    filetype = request.args.get("type", "minutes")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    stem = re.sub(r"[^\w\s-]", "", job.get("meeting_name", "meeting")).strip()
    stem = stem.replace(" ", "_") or "meeting"

    if filetype == "log":
        logs = list(job.get("logs", []))
        # Also try reading from disk in case logs were trimmed from memory
        log_path = _job_log_path(job_id)
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8")
        else:
            content = "\n".join(logs)
        if not content:
            return jsonify({"error": "No log available"}), 404
        fname = f"{stem}_job.log"
        response = Response(content.encode("utf-8"), mimetype="text/plain")
        response.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
        return response

    content = job["result_md"] if filetype == "minutes" else job["transcript"]
    if content is None:
        artifact_name = job.get("minutes_file") if filetype == "minutes" else job.get("transcript_file")
        content = _read_job_artifact(job_id, artifact_name)
    if not content:
        return jsonify({"error": "Not available yet"}), 404

    ext  = "md" if filetype == "minutes" else "txt"
    fname = f"{stem}_{'minutes' if filetype == 'minutes' else 'transcript'}.{ext}"

    mime = "text/markdown" if filetype == "minutes" else "text/plain"
    response = Response(content.encode("utf-8"), mimetype=mime)
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{fname}"'
    )
    return response


# ── Stale job cleanup ─────────────────────────────────────────────────────────

def _cleanup_loop():
    while True:
        time.sleep(3600)
        cutoff = time.time() - (JOB_RETENTION_DAYS * 86400)
        with _jobs_lock:
            stale = [
                jid for jid, j in _jobs.items()
                if j["created_at"] < cutoff and j["status"] in ("done", "error", "cancelled")
            ]
        for jid in stale:
            _delete_job(jid)


_load_persisted_jobs()
threading.Thread(target=_cleanup_loop, daemon=True).start()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8082):
    print(f"\n{'=' * 50}")
    print(f"  Meeting Summarizer  \u2014  Web UI")
    print(f"  http://{host}:{port}")
    print(f"{'=' * 50}\n")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("WEB_PORT", 8082)))
    args = p.parse_args()
    run_server(host=args.host, port=args.port)
