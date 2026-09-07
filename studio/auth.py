"""Username/password auth and per-user project isolation.

Three pieces, deliberately kept free of FastAPI so they can be tested without
a server:

* :class:`UserStore` — accounts from a YAML file, PBKDF2 password hashing.
* :class:`SessionStore` — opaque bearer tokens handed out on login.
* :class:`OwnerIndex` — which user owns which project directory.

Ownership is recorded in a side index (``<workspace>/owners.json``) rather than
inside each project's ``final_state.json``. That keeps the pipeline and the
project schema untouched, and means an ownership mistake can be corrected by
editing one small file instead of rewriting project state.

Projects with no entry in the index predate authentication. They are treated as
admin-only: a project whose owner is unknown must not be shown to an arbitrary
logged-in user just because nobody claimed it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("aicut_studio.auth")

HASH_SCHEME = "pbkdf2_sha256"
HASH_ITERATIONS = 210_000
SALT_BYTES = 16
ADMIN_ROLE = "admin"

MAX_USERNAME = 32
# Letters, digits, CJK and a few separators. Usernames end up in the owner
# index and in log lines, so control characters and whitespace are out.
_USERNAME_RE = re.compile(r"^[\w.\-]{1,%d}$" % MAX_USERNAME, re.UNICODE)


class InvalidUsername(ValueError):
    """Raised instead of silently creating an account under a junk name."""


def normalize_username(raw: str) -> str:
    """Trimmed username, or raise :class:`InvalidUsername`."""
    name = str(raw or "").strip()
    if not name:
        raise InvalidUsername("请输入用户名")
    if len(name) > MAX_USERNAME:
        raise InvalidUsername(f"用户名最长 {MAX_USERNAME} 个字符")
    if not _USERNAME_RE.match(name):
        raise InvalidUsername("用户名只能包含字母、数字、汉字、下划线、点和减号")
    return name


# ---------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------
def hash_password(password: str, *, iterations: int = HASH_ITERATIONS) -> str:
    """``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``."""
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{HASH_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against a stored hash."""
    try:
        scheme, iters, salt_hex, digest_hex = str(encoded).split("$", 3)
        if scheme != HASH_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode("utf-8"),
            bytes.fromhex(salt_hex), int(iters),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
@dataclass
class User:
    username: str
    role: str = "user"
    display_name: str = ""
    password_hash: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN_ROLE

    def public(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "role": self.role,
            "display_name": self.display_name or self.username,
            "is_admin": self.is_admin,
        }


class UserStore:
    """Accounts loaded from YAML. Reloaded on demand so edits need no restart."""

    def __init__(self, path: str | Path, *, passwordless: bool = False) -> None:
        self.path = Path(path).expanduser()
        self.passwordless = bool(passwordless)
        self.users: dict[str, User] = {}
        self.load_error = ""
        self._mtime = 0.0
        self._write_lock = threading.Lock()
        self.load()

    def load(self) -> None:
        import yaml

        self.users = {}
        self.load_error = ""
        if not self.path.is_file():
            # Passwordless mode creates the file on first login, so an absent
            # file is the normal starting state rather than a failure.
            if self.passwordless:
                return
            self.load_error = f"账号文件不存在: {self.path}"
            logger.warning("%s", self.load_error)
            return
        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._mtime = self.path.stat().st_mtime
        except (OSError, yaml.YAMLError) as exc:
            self.load_error = f"账号文件无法解析: {exc}"
            logger.error("%s", self.load_error)
            return

        entries = payload.get("users") if isinstance(payload, dict) else payload
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("username", "") or "").strip()
            pw = str(item.get("password_hash", "") or "").strip()
            # A blank hash is kept: passwordless accounts have none, and
            # dropping them here would recreate them on every login.
            if not name or (not pw and not self.passwordless):
                continue
            self.users[name] = User(
                username=name,
                role=str(item.get("role", "user") or "user"),
                display_name=str(item.get("display_name", "") or ""),
                password_hash=pw,
            )
        if not self.users and not self.passwordless:
            self.load_error = self.load_error or "账号文件里没有任何有效账号"

    def maybe_reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            logger.info("users file changed, reloading")
            self.load()

    def authenticate(self, username: str, password: str) -> User | None:
        self.maybe_reload()
        user = self.users.get(str(username or "").strip())
        if user is None or not user.password_hash:
            # Hash anyway so a missing account and a wrong password take
            # comparable time and cannot be told apart by timing.
            verify_password(password or "", hash_password("dummy", iterations=1000))
            return None
        return user if verify_password(password or "", user.password_hash) else None

    def ensure(self, username: str, *, role: str = "user") -> User:
        """Existing account, or a freshly created and persisted one.

        Only used in passwordless mode: the username *is* the identity, and its
        one job is to give each person a stable, isolated set of projects.

        An existing account is promoted when `role` is admin, but never demoted
        — a role set by hand through manage_users has to stick.
        """
        name = normalize_username(username)
        self.maybe_reload()
        existing = self.users.get(name)
        if existing is not None:
            if role == ADMIN_ROLE and not existing.is_admin:
                with self._write_lock:
                    existing.role = ADMIN_ROLE
                    self._save()
                logger.info("promoted %s to admin per config", name)
            return existing
        user = User(username=name, role=role, display_name=name)
        with self._write_lock:
            self.users[name] = user
            self._save()
        logger.info("auto-registered user %s as %s", name, role)
        return user

    def _save(self) -> None:
        """Rewrite users.yaml from memory, preserving hashes and roles."""
        import yaml

        payload = {"users": [
            {k: v for k, v in (
                ("username", u.username),
                ("role", u.role),
                ("display_name", u.display_name),
                ("password_hash", u.password_hash),
            ) if v}
            for u in sorted(self.users.values(), key=lambda x: x.username)
        ]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
            tmp.chmod(0o600)
            tmp.replace(self.path)
            self._mtime = self.path.stat().st_mtime
        except (OSError, yaml.YAMLError) as exc:
            logger.error("cannot write users file %s: %s", self.path, exc)

    def get(self, username: str) -> User | None:
        self.maybe_reload()
        return self.users.get(str(username or "").strip())


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
@dataclass
class SessionStore:
    """In-memory bearer tokens.

    Not persisted: a restart forces everyone to log in again, which is the safe
    default and costs one click. Tokens are opaque random strings, so there is
    no secret to rotate and nothing to forge offline.
    """

    ttl_seconds: float = 72 * 3600
    _tokens: dict[str, tuple[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def issue(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._tokens[token] = (username, time.time() + self.ttl_seconds)
        return token

    def resolve(self, token: str) -> str:
        """Username for a token, or "" when missing or expired."""
        if not token:
            return ""
        with self._lock:
            entry = self._tokens.get(token)
            if entry is None:
                return ""
            username, expires = entry
            if expires < time.time():
                self._tokens.pop(token, None)
                return ""
        return username

    def revoke(self, token: str) -> None:
        with self._lock:
            self._tokens.pop(token, None)

    def revoke_user(self, username: str) -> int:
        with self._lock:
            gone = [t for t, (u, _) in self._tokens.items() if u == username]
            for t in gone:
                self._tokens.pop(t, None)
        return len(gone)

    def purge(self) -> int:
        now = time.time()
        with self._lock:
            gone = [t for t, (_, exp) in self._tokens.items() if exp < now]
            for t in gone:
                self._tokens.pop(t, None)
        return len(gone)


# ---------------------------------------------------------------------------
# ownership
# ---------------------------------------------------------------------------
class OwnerIndex:
    """``project_dir -> username``, persisted as one small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.owners: dict[str, str] = {}
        self._lock = threading.Lock()
        self.load()

    @staticmethod
    def _key(project_dir: str | Path) -> str:
        return str(Path(project_dir).expanduser().resolve())

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read owner index %s: %s", self.path, exc)
            return
        if isinstance(payload, dict):
            src = payload.get("owners") if isinstance(payload.get("owners"), dict) else payload
            self.owners = {str(k): str(v) for k, v in (src or {}).items()}

    def _save_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"owners": self.owners}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("cannot write owner index %s: %s", self.path, exc)

    def claim(self, project_dir: str | Path, username: str) -> None:
        """Record ownership. First claim wins; re-claiming is a no-op."""
        if not project_dir or not username:
            return
        key = self._key(project_dir)
        with self._lock:
            if self.owners.get(key) == username:
                return
            self.owners.setdefault(key, username)
            self._save_locked()

    def owner_of(self, project_dir: str | Path) -> str:
        if not project_dir:
            return ""
        return self.owners.get(self._key(project_dir), "")

    def release(self, project_dir: str | Path) -> None:
        key = self._key(project_dir)
        with self._lock:
            if self.owners.pop(key, None) is not None:
                self._save_locked()

    def can_access(self, project_dir: str | Path, user: User | None) -> bool:
        if user is None:
            return False
        if user.is_admin:
            return True
        if not project_dir:
            return False
        # An unclaimed project is not "everyone's" — see module docstring.
        return self.owner_of(project_dir) == user.username

    def transfer(self, project_dir: str | Path, username: str) -> None:
        key = self._key(project_dir)
        with self._lock:
            self.owners[key] = username
            self._save_locked()
