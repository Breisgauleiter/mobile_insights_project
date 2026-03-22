#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Mobile Insights — Deployment Script
# Server: syntopia@91.99.20.173
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──
SERVER="syntopia@91.99.20.173"
REMOTE_BASE="/home/syntopia"
REMOTE_APP="$REMOTE_BASE/mobile-insights"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()   { echo -e "${RED}[ERR]${RESET}   $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  deploy          Deploy / update Mobile Insights
  status          Show container status
  logs [service]  Tail logs (default: all)
  shell           SSH into the server
  restart         Restart containers
  stop            Stop containers
  cleanup         Remove old Docker images to free disk space

EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Deploy
# ─────────────────────────────────────────────────────────────────────────────

cmd_deploy() {
    info "Deploying Mobile Insights..."

    # Create remote dir
    ssh "$SERVER" "mkdir -p $REMOTE_APP"

    # Sync project files (only what's needed for Docker build)
    rsync -avz --delete \
        --exclude 'node_modules' \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude '.git' \
        --exclude 'server/uploads' \
        --exclude 'server/__tests__' \
        --exclude 'ml/tests' \
        --exclude '.env' \
        --exclude 'docker-compose.local.yml' \
        --exclude 'docker-compose.yml' \
        "$LOCAL_DIR/" \
        "$SERVER:$REMOTE_APP/"

    # Copy prod compose as the main compose
    ssh "$SERVER" "cd $REMOTE_APP && [ -f .env ] || cp .env.example .env"

    # Ensure traefik-public network exists
    ssh "$SERVER" "docker network create traefik-public 2>/dev/null || true"

    # Build and deploy
    info "Building Docker image on server (this may take a while on first run)..."
    ssh "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml build app"

    info "Starting containers..."
    ssh "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml up -d"

    # Wait for health
    info "Waiting for health check..."
    local retries=20
    while [ $retries -gt 0 ]; do
        if ssh "$SERVER" "docker inspect mobile-insights-app --format '{{.State.Health.Status}}' 2>/dev/null" | grep -q healthy; then
            break
        fi
        retries=$((retries - 1))
        sleep 5
    done

    if [ $retries -eq 0 ]; then
        warn "Health check timed out — check logs with: $0 logs"
    else
        ok "Mobile Insights deployed and healthy!"
    fi

    cmd_status
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

cmd_status() {
    ssh "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml ps"
}

cmd_logs() {
    local service="${1:-}"
    ssh -t "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml logs -f --tail=100 $service"
}

cmd_shell() {
    ssh -t "$SERVER"
}

cmd_restart() {
    info "Restarting Mobile Insights..."
    ssh "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml restart"
    ok "Restarted"
}

cmd_stop() {
    info "Stopping Mobile Insights..."
    ssh "$SERVER" "cd $REMOTE_APP && docker compose -f docker-compose.prod.yml down"
    ok "Stopped"
}

cmd_cleanup() {
    info "Cleaning up old Docker images..."
    ssh "$SERVER" "docker image prune -f && docker builder prune -f"
    ok "Cleanup done"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    deploy)         cmd_deploy ;;
    status)         cmd_status ;;
    logs)           cmd_logs "${2:-}" ;;
    shell)          cmd_shell ;;
    restart)        cmd_restart ;;
    stop)           cmd_stop ;;
    cleanup)        cmd_cleanup ;;
    help|--help|-h) usage ;;
    *) err "Unknown command: $1"; usage; exit 1 ;;
esac
