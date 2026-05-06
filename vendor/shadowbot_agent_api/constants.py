from __future__ import annotations

import os

from dotenv import find_dotenv
from dotenv import load_dotenv

load_dotenv(find_dotenv())


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Constants(metaclass=Singleton):
    # General Configuration
    PYTHON_LOG_LEVEL = os.getenv("PYTHON_LOG_LEVEL", "INFO")
    
    # Auth Configuration
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    AUTH_ISSUER = os.getenv("AUTH_ISSUER", "")
    AUTH_AUDIENCE = os.getenv("AUTH_AUDIENCE", "")
    AUTH_JWKS_URL = os.getenv("AUTH_JWKS_URL", "")
    AUTH_ALGORITHMS = os.getenv("AUTH_ALGORITHMS", "RS256").split(",")
    AUTH_VERIFY_EXP = os.getenv("AUTH_VERIFY_EXP", "true").lower() == "true"
    AUTH_VERIFY_AUD = os.getenv("AUTH_VERIFY_AUD", "true").lower() == "true"
