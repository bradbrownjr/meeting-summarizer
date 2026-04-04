FROM python:3.12-alpine

# ffmpeg for audio processing
RUN apk add --no-cache ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY summarize_meeting.py .
COPY web.py .
COPY templates/ ./templates/

# Whisper model cache (avoids re-download on restart)
VOLUME ["/root/.cache/huggingface"]

ENV UPLOAD_DIR=/tmp/meeting_uploads
ENV WEB_PORT=8082

EXPOSE 8082

CMD ["python", "web.py"]
