#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="kalshi-bot"

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs}"
}

cmd="${1:-}"
case "$cmd" in
  start)
    sudo systemctl start "$SERVICE_NAME"
    ;;
  stop)
    sudo systemctl stop "$SERVICE_NAME"
    ;;
  restart)
    sudo systemctl restart "$SERVICE_NAME"
    ;;
  status)
    systemctl status "$SERVICE_NAME" --no-pager
    ;;
  logs)
    journalctl -u "$SERVICE_NAME" -f -n 200
    ;;
  *)
    usage
    exit 1
    ;;
esac
