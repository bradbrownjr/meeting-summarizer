FROM python:3.12-alpine

# ffmpeg for audio processing
RUN apk add --no-cache ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY summarize_meeting.py .
COPY web.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Whisper model cache (avoids re-download on restart)
# /data holds orgs.json (committee templates, user-configured)
VOLUME ["/root/.cache/huggingface", "/data"]

ENV UPLOAD_DIR=/tmp/meeting_uploads
ENV DATA_DIR=/data
ENV TEMPLATES_DIR=/templates
ENV WEB_PORT=8082

EXPOSE 8082

# Single worker process (required — job state is in-memory and not shared
# across processes). Gevent provides greenlet-based concurrency so many
# SSE streams can be open simultaneously without blocking uploads.
CMD gunicorn \
    --worker-class gevent \
    --workers 1 \
    --worker-connections 100 \
    --bind 0.0.0.0:${WEB_PORT} \
    --timeout 3600 \
    --keep-alive 65 \
    web:app
