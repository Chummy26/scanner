import json
import secrets
import sqlite3
import time
import hashlib
import re
from typing import Optional, Dict, Any, List

SESSION_TTL_SEC = 12 * 3600
RESET_TOKEN_TTL_SEC = 2 * 3600
PASSWORD_MIN_LEN = 8
USERNAME_MIN_LEN = 3
USERNAME_MAX_LEN = 32
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]*$")
PBKDF2_ITER = 120000


class AuthResult:
    def __init__(self, ok: bool, user: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.ok = ok
        self.user = user
        self.error = error


class AuthManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    csrf_token TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    last_seen INTEGER NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reset_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    used_at INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    user_id INTEGER PRIMARY KEY,
                    settings_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )

    def _now(self) -> int:
        return int(time.time())

    def _normalize_username(self, username: str) -> str:
        return str(username or "").strip().lower()

    def _hash_password(self, password: str, salt: str) -> str:
        salt_bytes = bytes.fromhex(salt)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITER)
        return digest.hex()

    def _validate_password(self, password: str) -> Optional[str]:
        if not password or len(password) < PASSWORD_MIN_LEN:
            return "password_too_short"
        return None

    def _validate_username(self, username: str) -> Optional[str]:
        if not username:
            return "invalid_username"
        if len(username) < USERNAME_MIN_LEN or len(username) > USERNAME_MAX_LEN:
            return "invalid_username"
        if not USERNAME_RE.match(username):
            return "invalid_username"
        return None

    def _user_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "username": row["username"],
            "is_admin": bool(row["is_admin"]),
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def _get_user_row(self, conn: sqlite3.Connection, username: str) -> Optional[sqlite3.Row]:
        username = self._normalize_username(username)
        if not username:
            return None
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def ensure_admin_from_env(self, username: str, password: str) -> bool:
        username = self._normalize_username(username)
        if self._validate_username(username):
            return False
        if self._validate_password(password):
            return False
        now = self._now()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            salt = secrets.token_hex(16)
            
            # If user exists, check if we actually need to update anything
            if row:
                current_salt = row["password_salt"]
                current_hash = row["password_hash"]
                new_hash_check = self._hash_password(password, current_salt)
                
                # If password matches AND is already admin AND is active, do nothing
                if (new_hash_check == current_hash and 
                    row["is_admin"] == 1 and 
                    row["is_active"] == 1):
                    return True
                    
                # Otherwise, update password/admin status and invalidate sessions
                pwd_hash = self._hash_password(password, salt)
                conn.execute(
                    """
                    UPDATE users
                    SET password_hash = ?, password_salt = ?, is_admin = 1, is_active = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (pwd_hash, salt, now, row["id"])
                )
                conn.execute("UPDATE sessions SET is_active = 0 WHERE user_id = ?", (row["id"],))
            else:
                pwd_hash = self._hash_password(password, salt)
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, password_salt, is_admin, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, 1, 1, ?, ?)
                    """,
                    (username, pwd_hash, salt, now, now)
                )
        return True

    def create_user(self, username: str, password: str, is_admin: bool = False) -> AuthResult:
        username = self._normalize_username(username)
        err = self._validate_username(username)
        if err:
            return AuthResult(False, error=err)
        pwd_err = self._validate_password(password)
        if pwd_err:
            return AuthResult(False, error=pwd_err)
        now = self._now()
        salt = secrets.token_hex(16)
        pwd_hash = self._hash_password(password, salt)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, password_salt, is_admin, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    """,
                    (username, pwd_hash, salt, 1 if is_admin else 0, now, now)
                )
                row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        except sqlite3.IntegrityError:
            return AuthResult(False, error="user_exists")
        return AuthResult(True, user=self._user_row_to_dict(row))

    def authenticate(self, username: str, password: str) -> AuthResult:
        username = self._normalize_username(username)
        if not username:
            return AuthResult(False, error="invalid_username")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not row or not row["is_active"]:
                return AuthResult(False, error="invalid_credentials")
            pwd_hash = self._hash_password(password, row["password_salt"])
            if pwd_hash != row["password_hash"]:
                return AuthResult(False, error="invalid_credentials")
            return AuthResult(True, user=self._user_row_to_dict(row))

    def create_session(self, user_id: int) -> Dict[str, Any]:
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now = self._now()
        expires_at = now + SESSION_TTL_SEC
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (token, user_id, csrf_token, created_at, expires_at, last_seen, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (token, user_id, csrf_token, now, expires_at, now)
            )
        return {
            "token": token,
            "csrf_token": csrf_token,
            "expires_at": expires_at
        }

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.token, s.user_id, s.csrf_token, s.expires_at, s.is_active,
                       u.username, u.is_admin, u.is_active as user_active
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,)
            ).fetchone()
            if not row:
                return None
            if not row["is_active"] or not row["user_active"] or row["expires_at"] <= now:
                conn.execute("UPDATE sessions SET is_active = 0 WHERE token = ?", (token,))
                return None
            conn.execute("UPDATE sessions SET last_seen = ? WHERE token = ?", (now, token))
            return {
                "token": row["token"],
                "user_id": row["user_id"],
                "username": row["username"],
                "is_admin": bool(row["is_admin"]),
                "csrf_token": row["csrf_token"],
                "expires_at": row["expires_at"],
                "is_active": True
            }

    def revoke_session(self, token: str):
        if not token:
            return
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET is_active = 0 WHERE token = ?", (token,))

    def _revoke_user_sessions(self, user_id: int):
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET is_active = 0 WHERE user_id = ?", (user_id,))

    def set_password(self, user_id: int, new_password: str) -> AuthResult:
        pwd_err = self._validate_password(new_password)
        if pwd_err:
            return AuthResult(False, error=pwd_err)
        now = self._now()
        salt = secrets.token_hex(16)
        pwd_hash = self._hash_password(new_password, salt)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, updated_at = ?
                WHERE id = ?
                """,
                (pwd_hash, salt, now, user_id)
            )
        self._revoke_user_sessions(user_id)
        return AuthResult(True)

    def set_user_password(self, username: str, new_password: str) -> AuthResult:
        username = self._normalize_username(username)
        if not username:
            return AuthResult(False, error="invalid_username")
        with self._connect() as conn:
            row = self._get_user_row(conn, username)
            if not row:
                return AuthResult(False, error="not_found")
            user_id = row["id"]
        return self.set_password(user_id, new_password)

    def set_user_admin(self, username: str, is_admin: bool) -> AuthResult:
        username = self._normalize_username(username)
        if not username:
            return AuthResult(False, error="invalid_username")
        now = self._now()
        with self._connect() as conn:
            row = self._get_user_row(conn, username)
            if not row:
                return AuthResult(False, error="not_found")
            if row["is_admin"] and not is_admin:
                admin_count = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE is_admin = 1 AND is_active = 1"
                ).fetchone()[0]
                if admin_count <= 1:
                    return AuthResult(False, error="last_admin")
            conn.execute(
                "UPDATE users SET is_admin = ?, updated_at = ? WHERE username = ?",
                (1 if is_admin else 0, now, username)
            )
        return AuthResult(True)

    def revoke_user_sessions(self, username: str) -> bool:
        username = self._normalize_username(username)
        if not username:
            return False
        with self._connect() as conn:
            row = self._get_user_row(conn, username)
            if not row:
                return False
            conn.execute("UPDATE sessions SET is_active = 0 WHERE user_id = ?", (row["id"],))
        return True

    def reset_settings_for_user(self, username: str) -> bool:
        username = self._normalize_username(username)
        if not username:
            return False
        with self._connect() as conn:
            row = self._get_user_row(conn, username)
            if not row:
                return False
            conn.execute("DELETE FROM settings WHERE user_id = ?", (row["id"],))
        return True

    def delete_user(self, username: str) -> AuthResult:
        username = self._normalize_username(username)
        if not username:
            return AuthResult(False, error="invalid_username")
        with self._connect() as conn:
            row = self._get_user_row(conn, username)
            if not row:
                return AuthResult(False, error="not_found")
            if row["is_admin"]:
                admin_count = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE is_admin = 1 AND is_active = 1"
                ).fetchone()[0]
                if admin_count <= 1:
                    return AuthResult(False, error="last_admin")
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            conn.execute("DELETE FROM reset_tokens WHERE user_id = ?", (row["id"],))
            conn.execute("DELETE FROM settings WHERE user_id = ?", (row["id"],))
            conn.execute("DELETE FROM users WHERE id = ?", (row["id"],))
        return AuthResult(True)

    def create_reset_token(self, username: str) -> AuthResult:
        username = self._normalize_username(username)
        if not username:
            return AuthResult(False, error="invalid_username")
        now = self._now()
        expires_at = now + RESET_TOKEN_TTL_SEC
        token = secrets.token_urlsafe(24)
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                return AuthResult(False, error="not_found")
            conn.execute(
                """
                INSERT INTO reset_tokens (token, user_id, created_at, expires_at, used_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (token, row["id"], now, expires_at)
            )
        return AuthResult(True, user={"token": token, "expires_at": expires_at})

    def consume_reset_token(self, token: str, new_password: str) -> AuthResult:
        if not token:
            return AuthResult(False, error="invalid_token")
        pwd_err = self._validate_password(new_password)
        if pwd_err:
            return AuthResult(False, error=pwd_err)
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM reset_tokens
                WHERE token = ?
                """,
                (token,)
            ).fetchone()
            if not row or row["used_at"] or row["expires_at"] <= now:
                return AuthResult(False, error="invalid_token")
            user_id = row["user_id"]
        res = self.set_password(user_id, new_password)
        if not res.ok:
            return res
        with self._connect() as conn:
            conn.execute(
                "UPDATE reset_tokens SET used_at = ? WHERE token = ?",
                (now, token)
            )
        return AuthResult(True)

    def list_users(self) -> List[Dict[str, Any]]:
        now = self._now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.*,
                       (SELECT MAX(s.last_seen) FROM sessions s WHERE s.user_id = u.id) AS last_seen,
                       (SELECT MAX(s.created_at) FROM sessions s WHERE s.user_id = u.id) AS last_login,
                       (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id) AS sessions_total,
                       (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND s.is_active = 1 AND s.expires_at > ?) AS sessions_active
                FROM users u
                ORDER BY u.username
                """,
                (now,)
            ).fetchall()
            payload = []
            for row in rows:
                user = self._user_row_to_dict(row)
                user["last_seen"] = row["last_seen"]
                user["last_login"] = row["last_login"]
                user["sessions_total"] = row["sessions_total"]
                user["sessions_active"] = row["sessions_active"]
                payload.append(user)
            return payload

    def set_user_active(self, username: str, active: bool) -> bool:
        username = self._normalize_username(username)
        if not username:
            return False
        now = self._now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET is_active = ?, updated_at = ? WHERE username = ?",
                (1 if active else 0, now, username)
            )
        if cur.rowcount:
            if not active:
                with self._connect() as conn:
                    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
                    if row:
                        conn.execute("UPDATE sessions SET is_active = 0 WHERE user_id = ?", (row["id"],))
            return True
        return False

    def get_settings(self, user_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT settings_json FROM settings WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            if not row:
                return {}
            try:
                payload = json.loads(row["settings_json"])
            except Exception:
                payload = {}
            return payload if isinstance(payload, dict) else {}

    def set_settings(self, user_id: int, settings: Dict[str, Any]):
        payload = settings if isinstance(settings, dict) else {}
        now = self._now()
        with self._connect() as conn:
            existing = conn.execute("SELECT user_id FROM settings WHERE user_id = ?", (user_id,)).fetchone()
            if existing:
                conn.execute(
                    "UPDATE settings SET settings_json = ?, updated_at = ? WHERE user_id = ?",
                    (json.dumps(payload), now, user_id)
                )
            else:
                conn.execute(
                    "INSERT INTO settings (user_id, settings_json, updated_at) VALUES (?, ?, ?)",
                    (user_id, json.dumps(payload), now)
                )
