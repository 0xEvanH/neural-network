FROM oven/bun:1 AS base

# System
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend
COPY backend/requirements.txt ./backend/requirements.txt

RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/

# Frontend
COPY frontend/package.json frontend/bun.lockb* ./frontend/
RUN cd frontend && bun install --frozen-lockfile

COPY frontend/ ./frontend/


RUN mkdir -p /var/log/supervisor

COPY <<'EOF' /etc/supervisor/conf.d/app.conf
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:api]
command=/app/.venv/bin/python app.py
directory=/app/backend
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/api.err.log
stdout_logfile=/var/log/supervisor/api.out.log
environment=PYTHONUNBUFFERED="1"

[program:ui]
command=bun run dev
directory=/app/frontend
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/ui.err.log
stdout_logfile=/var/log/supervisor/ui.out.log
EOF

# Backend
EXPOSE 5000
# Frontend
EXPOSE 5173

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]