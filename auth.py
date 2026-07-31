"""JWT auth utilities and FastAPI dependencies.

Adapted from the Marketing Tool project's auth.py — same JWT/bearer-token
pattern, wired to FileUploadManager's SQLite-backed users instead of a
JSON-file store. init_auth() is called once from agent.py after
upload_manager exists, to avoid a circular import.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production-use-a-long-random-string")
ALGORITHM = "HS256"
TOKEN_DAYS = 30

bearer = HTTPBearer()

_user_lookup = None  # set via init_auth(); takes a username, returns a user dict or None


def init_auth(user_lookup_fn) -> None:
    global _user_lookup
    _user_lookup = user_lookup_fn


def create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=TOKEN_DAYS)
    return jwt.encode(
        {"sub": username, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def _decode(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        payload = _decode(creds.credentials)
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = _user_lookup(username) if _user_lookup else None
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
