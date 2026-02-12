from __future__ import annotations

import httpx


class Notifier:
    def __init__(
        self,
        *,
        telegram_bot_token: str | None = None,
        telegram_chat_id: str | None = None,
        discord_webhook_url: str | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook_url = discord_webhook_url
        self.timeout_sec = timeout_sec

    def send(self, message: str) -> None:
        if self.telegram_bot_token and self.telegram_chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            with httpx.Client(timeout=self.timeout_sec) as client:
                client.post(url, json={"chat_id": self.telegram_chat_id, "text": message})
        if self.discord_webhook_url:
            with httpx.Client(timeout=self.timeout_sec) as client:
                client.post(self.discord_webhook_url, json={"content": message})
