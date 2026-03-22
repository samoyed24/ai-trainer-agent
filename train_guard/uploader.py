"""指标上报器。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests


logger = logging.getLogger(__name__)


class MetricsUploader:
    """负责向 dashboard 后端上报指标。"""

    def __init__(
        self,
        server_config: dict[str, Any],
        agent_config: dict[str, Any],
        auth_headers: dict[str, str] | None = None,
    ):
        self.server_url = server_config["url"]
        self.timeout = server_config.get("timeout", 10)
        self.retry_count = server_config.get("retry_count", 3)
        self.enable_async = agent_config.get("enable_async", True)
        self.auth_headers = dict(auth_headers or {})
        self.session = requests.Session()

    def upload(self, payload: dict[str, Any]) -> bool:
        if self.enable_async:
            thread = threading.Thread(target=self._upload_with_retry, args=(payload,), daemon=True)
            thread.start()
            return True
        return self._upload_with_retry(payload)

    def _upload_with_retry(self, payload: dict[str, Any]) -> bool:
        for attempt in range(self.retry_count):
            try:
                response = self.session.post(
                    self.server_url,
                    json=payload,
                    headers=self.auth_headers or None,
                    timeout=self.timeout,
                )
                if response.status_code in (200, 201, 202):
                    logger.info(
                        "指标上传成功 (train_id=%s, epoch=%s, step=%s, status=%s)",
                        payload.get("train_id"),
                        payload.get("epoch"),
                        payload.get("step"),
                        response.status_code,
                    )
                    return True

                logger.warning("上传失败 (status=%s): %s", response.status_code, response.text[:120])
            except requests.Timeout:
                logger.warning("上传超时 (尝试 %s/%s)", attempt + 1, self.retry_count)
            except requests.ConnectionError as exc:
                logger.warning("连接失败 (尝试 %s/%s): %s", attempt + 1, self.retry_count, str(exc)[:80])
            except Exception as exc:
                logger.error("上传出错: %s", str(exc)[:200])
                return False

            if attempt < self.retry_count - 1:
                time.sleep(2 ** attempt)

        logger.error("上传失败（已重试 %s 次）", self.retry_count)
        return False

    def close(self) -> None:
        self.session.close()
