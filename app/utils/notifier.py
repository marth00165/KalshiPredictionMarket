import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

class Notifier:
    """Basic webhook notifier for critical bot events."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    async def send_notification(self, message: str, level: str = "INFO"):
        """Send a message to the configured webhook."""
        if not self.webhook_url:
            logger.debug(f"Notifier: No webhook configured. Message: {message}")
            return

        payload = {
            "text": f"[{level}] {message}",
            "level": level
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as response:
                    if response.status >= 400:
                        logger.warning(f"Failed to send notification: {response.status}")
        except Exception as e:
            logger.warning(f"Error sending notification: {e}")
