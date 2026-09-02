import sys
import logging
from pathlib import Path
from typing import Optional, List
from collections import deque

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "app.log"

# In-memory buffer storing recent log entries for live UI display
LOG_BUFFER = deque(maxlen=200)


class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            self.handleError(record)


_memory_handler = MemoryLogHandler()
_memory_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_memory_handler.setFormatter(_memory_formatter)


def get_recent_logs() -> List[str]:
    """Retrieve recent log lines from memory buffer."""
    return list(LOG_BUFFER)


def get_logger(name: str = "ai_backend", level: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a configured logger instance with stdout, file, and in-memory UI streaming.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 1. Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (app.log)
        try:
            file_handler = logging.FileHandler(str(DEFAULT_LOG_FILE), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

        # 3. In-memory buffer handler for Streamlit UI
        logger.addHandler(_memory_handler)

        logger.setLevel(level or "INFO")
        logger.propagate = False

    if level:
        logger.setLevel(level)

    return logger
