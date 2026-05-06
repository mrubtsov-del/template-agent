from __future__ import annotations

import logging.config
import sys

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(
                msg
            )  # Use tqdm's write method to ensure output doesn't interfere with progress bars
        except Exception:
            self.handleError(record)


def get_python_logger(
    log_level,
    log_format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    log_date_format="%Y-%m-%d %H:%M:%S",
):
    log_level = log_level.upper()
    logger_config = {
        "version": 1,
        "formatters": {
            "shadowbot_agent_api_logger": {"format": log_format, "datefmt": log_date_format}
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "shadowbot_agent_api_logger",
                "stream": sys.stdout,
            },
            "tqdm_console": {
                "level": log_level,
                "()": TqdmLoggingHandler,  # Specify the custom handler here
                "formatter": "shadowbot_agent_api_logger",
            },
        },
        "loggers": {
            "shadowbot_agent_api_logger": {
                "level": log_level,
                "handlers": ["tqdm_console"],  # Use the custom tqdm handler
            }
        },
        "disable_existing_loggers": False,
    }

    logging.config.dictConfig(logger_config)
    return logging.getLogger("shadowbot_agent_api_logger")
