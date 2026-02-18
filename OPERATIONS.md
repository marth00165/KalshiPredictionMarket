# Operations Guide - Kalshi Bot

This document describes how to manage the Kalshi Bot in production, including locking, backups, and restoration.

## Overlap Prevention (Locking)

The bot uses a lock file to ensure that only one instance is running at a time. This prevents database contention and duplicate API calls.

- **Lock File Locations**:
    - Production: `/var/run/kalshi-bot.lock`
    - Development/Local: `/tmp/kalshi-bot.lock`
- **Mechanism**: The lock file contains the Process ID (PID) of the running instance. If the bot starts and finds a lock file, it checks if the PID is still active. If it is, the new instance exits immediately. If not, it assumes the lock is stale and overwrites it.
- **Manual Release**: If you are certain no bot is running but the lock persists, you can manually delete the lock file.

## Database Backups

It is recommended to back up the database regularly.

### Manual Backup
You can trigger a backup manually using the CLI:

```bash
python -m app --backup
```

This will create a timestamped copy of the database in the `backups/` directory (e.g., `backups/kalshi_20260216_210322.sqlite`).

### Scheduled Backup
To schedule a daily backup via cron:

```cron
0 2 * * * cd /path/to/bot && /path/to/venv/bin/python -m app --backup >> /path/to/bot/logs/backup.log 2>&1
```

## Database Restoration

If you need to restore the database from a backup, use the provided `restore_db.sh` script.

### Prerequisites
1. Stop any running instances of the bot.
2. Ensure you have the backup file path.

### Restoration Steps

```bash
./restore_db.sh <path_to_backup_file> [target_db_path]
```

**Example:**
```bash
./restore_db.sh backups/kalshi_20260216_210322.sqlite
```

The script will:
1. Verify the backup file exists.
2. Check for active lock files and warn you if the bot might be running.
3. Overwrite the current database with the backup.

## Troubleshooting

### Lock Conflict
If the bot fails with "Failed to acquire lock", check if another instance is running:
```bash
ps aux | grep "python -m app"
```

### Permission Denied
If the bot cannot create the lock file in `/var/run/`, ensure the user running the bot has write permissions to that directory, or use a custom lock file path:
```bash
python -m app --lock-file ./custom.lock
```

## Autonomous Live Checklist

Before enabling real money unattended runs:

1. Confirm `advanced_config.json` has:
- `trading.dry_run=false`
- `trading.autonomous_mode=true`
- `trading.non_interactive=true`
- explicit scope (`series_tickers` or allowed market/event IDs)
2. Validate config:
```bash
python -m app --verify-config --mode trade
```
3. Run a one-cycle smoke test:
```bash
python -m app --mode trade --once --skip-setup-wizard --non-interactive
```
4. Confirm heartbeat updates at `reports/heartbeat.json`.
5. Test kill switch:
```bash
export BOT_DISABLE_TRADING=1
python -m app --mode trade --once --skip-setup-wizard --non-interactive
unset BOT_DISABLE_TRADING
```

## systemd Service (Recommended on VPS)

Service template: `deploy/systemd/kalshi-bot.service`.

Install/update service:
```bash
sudo cp deploy/systemd/kalshi-bot.service /etc/systemd/system/kalshi-bot.service
sudo systemctl daemon-reload
sudo systemctl enable kalshi-bot
sudo systemctl restart kalshi-bot
```

One-command operations:
```bash
./deploy/systemd/manage.sh start
./deploy/systemd/manage.sh stop
./deploy/systemd/manage.sh status
./deploy/systemd/manage.sh logs
```

## Log Retention

Use journald retention and/or logrotate:

1. journald (example):
```ini
# /etc/systemd/journald.conf
SystemMaxUse=500M
MaxRetentionSec=14day
```
2. Reload journald:
```bash
sudo systemctl restart systemd-journald
```

## Health Watchdog

The bot writes `reports/heartbeat.json` each cycle. A simple watchdog should alert when:

- heartbeat file is stale (no update within expected cycle interval + buffer),
- status is not `active`,
- bankroll drops below your minimum threshold.
