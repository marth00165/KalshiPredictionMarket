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
