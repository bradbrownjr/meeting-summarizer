"""Meeting Summarizer — Web UI.

Run:
    python web.py [--port 8082] [--host 0.0.0.0]
"""

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, send_file


# ── Data directory (orgs.json lives here) ────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ORGS_FILE = DATA_DIR / "orgs.json"

TEMPLATES_DIR = Path(os.environ.get(
    "TEMPLATES_DIR",
    Path.home() / "meeting-templates",
))

DEFAULT_BACKEND      = os.environ.get("DEFAULT_BACKEND",      "claude-api")
DEFAULT_OLLAMA_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "gemma4:e4b")

SERVERS_FILE = DATA_DIR / "servers.json"


def _default_servers() -> dict:
    """Seed one server from the container's env vars."""
    return {
        "default": {
            "label":         "Home Server",
            "whisper_url":   os.environ.get("WHISPER_URL",   "http://localhost:8000"),
            "whisper_model": os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3"),
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


# ── Flask app ─────────────────────────────────────────────────────────────────

_template_dir = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=_template_dir)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", tempfile.gettempdir())) / "meeting_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Job state ─────────────────────────────────────────────────────────────────

import queue as _queue

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "logs": [],
            "result_md": None,
            "transcript": None,
            "meeting_name": "",
            "error": None,
            "subscribers": [],  # one Queue per active SSE connection
            "created_at": time.time(),
        }
    return job_id


def _push(job_id: str, event: dict):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return
    etype = event.get("type")
    if etype == "log":
        job["logs"].append(event.get("data", {}).get("message", ""))
    elif etype == "status":
        job["status"] = event.get("data", {}).get("status", job["status"])
    elif etype == "result":
        job["result_md"] = event["data"].get("minutes", "")
        job["transcript"] = event["data"].get("transcript", "")
        job["status"] = "done"
    elif etype == "error":
        job["error"] = event["data"].get("message", "Unknown error")
        job["status"] = "error"
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
             date_str: str, names: str, backend: str, ollama_model: str,
             server_id: str = "default"):
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

    cleanup: list = []

    # Resolve server config
    server = load_servers().get(server_id) or load_servers().get("default") or {}
    _whisper_url   = server.get("whisper_url")   or None
    _whisper_model = server.get("whisper_model") or None
    _ollama_url    = server.get("ollama_url")    or None

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

        context = org.get("context_template", "").format(date=formatted_date)
        meeting_name = org.get("name_template", "Meeting \u2013 {month_year}").format(
            month_year=month_year
        )
        with _jobs_lock:
            _jobs[job_id]["meeting_name"] = meeting_name

        # Merge org vocabulary with any names entered at submit time
        org_vocab = org.get("vocabulary", "")
        combined_vocab = ", ".join(filter(None, [org_vocab, names]))

        log(f"Starting: {meeting_name}")
        log(f"Audio: {audio_path.name}  ({audio_path.stat().st_size // 1024:,} KB)")
        log(f"Whisper model: {WHISPER_MODEL}")
        if combined_vocab:
            log(f"Vocabulary hints: {combined_vocab[:120]}{'...' if len(combined_vocab) > 120 else ''}")

        result = transcribe(audio_path, cleanup, hotwords=combined_vocab,
                            emit_callback=emit_cb,
                            whisper_url=_whisper_url,
                            whisper_model=_whisper_model)
        transcript = build_clean_transcript(audio_path, result, cleanup,
                                            hotwords=combined_vocab,
                                            emit_callback=emit_cb,
                                            whisper_url=_whisper_url,
                                            whisper_model=_whisper_model)

        word_count = len(transcript.split())
        log(f"Transcript complete \u2014 {word_count:,} words")

        if backend == "transcript_only":
            _emit(job_id, "result", minutes="", transcript=transcript, meeting_name=meeting_name)
            log("Done (transcript only.)")
            return

        _emit(job_id, "status", status="generating")
        log(f"Generating minutes (backend: {backend})\u2026")

        minutes = generate_minutes(
            transcript,
            names=names,
            context=context,
            backend=backend,
            ollama_model=ollama_model or OLLAMA_MODEL,
            emit_callback=emit_cb,
            ollama_url=_ollama_url,
        )
        _emit(job_id, "result", minutes=minutes, transcript=transcript, meeting_name=meeting_name)
        log("Done.")

    except Exception as exc:
        log(f"Error: {exc}", level="error")
        _emit(job_id, "error", message=str(exc))

    finally:
        try:
            unload_whisper_model(WHISPER_MODEL, whisper_url=_whisper_url)
        except Exception:
            pass
        if backend == "ollama":
            try:
                unload_ollama_model(ollama_model or OLLAMA_MODEL, ollama_url=_ollama_url)
            except Exception:
                pass
        for p in cleanup:
            if p.exists():
                p.unlink()
        # Signal SSE stream to close
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job:
            try:
                job["q"].put_nowait({"type": "eof", "data": {}})
            except _queue.Full:
                pass


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
    servers = load_servers()
    server_id = data.get("id") or _make_org_id(data["label"], servers)
    servers[server_id] = {
        "label":         data["label"].strip(),
        "whisper_url":   data.get("whisper_url", "").strip(),
        "whisper_model": data.get("whisper_model", "").strip(),
        "ollama_url":    data.get("ollama_url", "").strip(),
    }
    save_servers(servers)
    return jsonify({"ok": True, "id": server_id, "servers": servers})


@app.route("/api/servers/<server_id>", methods=["PUT"])
def api_servers_update(server_id: str):
    data = request.get_json(force=True)
    if not data or not data.get("label", "").strip():
        return jsonify({"error": "label is required"}), 400
    servers = load_servers()
    if server_id not in servers:
        return jsonify({"error": "Not found"}), 404
    servers[server_id] = {
        "label":         data["label"].strip(),
        "whisper_url":   data.get("whisper_url", "").strip(),
        "whisper_model": data.get("whisper_model", "").strip(),
        "ollama_url":    data.get("ollama_url", "").strip(),
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
        }
        imported.append(label)
    return imported, errors


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

    org_id       = request.form.get("org", "")
    date_str     = request.form.get("date", datetime.today().strftime("%Y-%m-%d"))
    names        = request.form.get("names", "")
    backend      = request.form.get("backend", "claude-api")
    ollama_model = request.form.get("ollama_model", "")
    server_id    = request.form.get("server_id", "default")

    job_id = _new_job()
    dest = UPLOAD_DIR / job_id / Path(audio.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio.save(dest)

    threading.Thread(
        target=_run_job,
        args=(job_id, dest, org_id, date_str, names, backend, ollama_model, server_id),
        daemon=True,
    ).start()

    return jsonify({"ok": True, "job_id": job_id})


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
            }
            for jid, j in _jobs.items()
        ]
    snapshot.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(snapshot)


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
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status":       job["status"],
        "meeting_name": job["meeting_name"],
        "logs":         list(job["logs"]),
        "result_md":    job["result_md"],
        "transcript":   job["transcript"],
        "error":        job["error"],
    })


@app.route("/api/jobs/<job_id>/download")
def api_download(job_id: str):
    filetype = request.args.get("type", "minutes")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    content = job["result_md"] if filetype == "minutes" else job["transcript"]
    if not content:
        return jsonify({"error": "Not available yet"}), 404

    stem = re.sub(r"[^\w\s-]", "", job.get("meeting_name", "meeting")).strip()
    stem = stem.replace(" ", "_") or "meeting"
    ext  = "md" if filetype == "minutes" else "txt"
    fname = f"{stem}_{'minutes' if filetype == 'minutes' else 'transcript'}.{ext}"

    tmp = Path(tempfile.mktemp(suffix=f"_{fname}"))
    tmp.write_text(content, encoding="utf-8")
    return send_file(tmp, as_attachment=True, download_name=fname)


# ── Stale job cleanup ─────────────────────────────────────────────────────────

def _cleanup_loop():
    while True:
        time.sleep(3600)
        cutoff = time.time() - 86400
        with _jobs_lock:
            stale = [jid for jid, j in _jobs.items() if j["created_at"] < cutoff]
        for jid in stale:
            shutil.rmtree(UPLOAD_DIR / jid, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(jid, None)


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
