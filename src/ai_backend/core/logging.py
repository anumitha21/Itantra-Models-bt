"""
Centralized Logging Utility for AI Backend Foundation.
"""

import sys
import logging
from typing import Optional


def get_logger(name: str = "ai_backend", level: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a configured logger instance for the AI Backend.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level or "INFO")
        logger.propagate = False
    
    if level:
        logger.setLevel(level)

    return logger
