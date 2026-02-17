import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class LockManager:
    """
    Manages a lock file to prevent multiple instances of the bot from running simultaneously.
    Uses PID-based detection to handle stale locks.
    """

    def __init__(self, lock_file_path: Optional[str] = None):
        """
        Initialize the LockManager.

        Args:
            lock_file_path: Optional custom path for the lock file.
                           If not provided, it tries /var/run/kalshi-bot.lock,
                           falling back to /tmp/kalshi-bot.lock.
        """
        if lock_file_path:
            self.lock_path = Path(lock_file_path)
        else:
            # Default logic: /var/run for prod, /tmp for dev
            prod_path = Path("/var/run/kalshi-bot.lock")
            dev_path = Path("/tmp/kalshi-bot.lock")

            # Use /var/run if writable, otherwise /tmp
            if os.access("/var/run", os.W_OK):
                self.lock_path = prod_path
            else:
                self.lock_path = dev_path

        self.acquired = False

    def is_pid_running(self, pid: int) -> bool:
        """
        Check if a process with the given PID is still running.

        Args:
            pid: Process ID to check

        Returns:
            True if the process is running, False otherwise.
        """
        if pid <= 0:
            return False
        try:
            # Signal 0 does not kill the process but checks if it exists
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def acquire(self) -> bool:
        """
        Acquire the lock.

        Returns:
            True if the lock was successfully acquired, False otherwise.
        """
        if self.lock_path.exists():
            try:
                with open(self.lock_path, 'r') as f:
                    data = json.load(f)
                    old_pid = data.get('pid')
                    if old_pid and self.is_pid_running(old_pid):
                        logger.error(f"❌ Lock conflict: Process {old_pid} is already running.")
                        return False
                    else:
                        logger.warning(f"⚠️ Stale lock found (PID {old_pid} not running). Overwriting...")
            except (json.JSONDecodeError, KeyError, PermissionError) as e:
                logger.warning(f"⚠️ Could not read existing lock file ({e}). Overwriting...")

        # Prepare lock data
        lock_data = {
            "pid": os.getpid(),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        try:
            # Ensure parent directory exists
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.lock_path, 'w') as f:
                json.dump(lock_data, f, indent=2)

            self.acquired = True
            logger.info(f"✅ Lock acquired at {self.lock_path} (PID: {os.getpid()})")
            return True
        except PermissionError:
            logger.error(f"❌ Permission denied: Cannot write lock file to {self.lock_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create lock file: {e}")
            return False

    def release(self) -> None:
        """Release the lock by deleting the lock file."""
        if not self.acquired:
            return

        try:
            if self.lock_path.exists():
                # Verify it's still our lock before deleting
                with open(self.lock_path, 'r') as f:
                    data = json.load(f)
                    if data.get('pid') == os.getpid():
                        self.lock_path.unlink()
                        logger.info(f"🔓 Lock released: {self.lock_path}")
                    else:
                        logger.warning("⚠️ Not releasing lock: PID in file does not match current PID.")
            self.acquired = False
        except Exception as e:
            logger.error(f"❌ Error releasing lock: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit ensures lock release."""
        self.release()
