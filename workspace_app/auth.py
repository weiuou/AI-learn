from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import Settings


PBKDF2_ITERATIONS = 390_000


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
    )


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _decode(salt), int(iterations))
        return hmac.compare_digest(actual, _decode(expected))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class InviteUser:
    id: str
    username: str
    password_hash: str


class AuthService:
    cookie_name = "workspace_session"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.serializer = URLSafeTimedSerializer(settings.session_secret, salt="workspace-session-v1")
        self.users = self._load_users()

    def _load_users(self) -> dict[str, InviteUser]:
        if self.settings.invite_users_json:
            raw: list[dict[str, Any]] = json.loads(self.settings.invite_users_json)
            users = [
                InviteUser(
                    id=str(item["id"]),
                    username=str(item["username"]),
                    password_hash=str(item["passwordHash"]),
                )
                for item in raw
            ]
        else:
            users = [InviteUser(id="dev-user", username="demo", password_hash=hash_password("demo", salt=b"workspace-demo!!"))]
        return {user.username.casefold(): user for user in users}

    def authenticate(self, username: str, password: str) -> InviteUser | None:
        user = self.users.get(username.strip().casefold())
        if user and verify_password(password, user.password_hash):
            return user
        return None

    def sign(self, user: InviteUser) -> str:
        return self.serializer.dumps({"sub": user.id, "username": user.username})

    def unsign(self, token: str | None) -> dict[str, str] | None:
        if not token:
            return None
        try:
            payload = self.serializer.loads(token, max_age=self.settings.session_max_age)
        except (BadSignature, SignatureExpired):
            return None
        user_id = str(payload.get("sub", ""))
        username = str(payload.get("username", ""))
        if not user_id or not username:
            return None
        if not any(user.id == user_id and user.username == username for user in self.users.values()):
            return None
        return {"id": user_id, "username": username}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 -m workspace_app.auth <password>")
        return 2
    print(hash_password(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
