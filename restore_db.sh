#!/bin/bash

# restore_db.sh - Restore Kalshi Bot database from a backup
# Usage: ./restore_db.sh <backup_file> [target_db_path]

set -e

BACKUP_FILE=$1
DB_PATH=${2:-"kalshi.sqlite"}

# Possible lock file locations
LOCK_FILES=("/var/run/kalshi-bot.lock" "/tmp/kalshi-bot.lock")

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file> [target_db_path]"
    echo "Example: $0 backups/kalshi_20260216_210322.sqlite"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file '$BACKUP_FILE' not found."
    exit 1
fi

# Check if any lock file exists
for LOCK in "${LOCK_FILES[@]}"; do
    if [ -f "$LOCK" ]; then
        echo "⚠️  Warning: Lock file $LOCK exists. The bot might be running."
        echo "It is strongly recommended to stop the bot before restoring the database to prevent corruption."
        read -p "Do you want to proceed with the restoration anyway? (y/N) " confirm
        if [[ $confirm != [yY] ]]; then
            echo "Restoration aborted."
            exit 1
        fi
        break
    fi
done

# Perform restoration
echo "🔄 Restoring $BACKUP_FILE to $DB_PATH..."

# Create a temporary copy to ensure we don't corrupt the backup during copy
cp "$BACKUP_FILE" "$DB_PATH"

echo "✅ Database restored successfully at $DB_PATH"
