"""Password hashing with Argon2id (via argon2-cffi). Plain passwords are never stored."""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()  # library defaults: Argon2id with a random salt per hash

# Verified against when a username does not exist, so "unknown user" and
# "wrong password" take about the same time (limits username probing).
_DUMMY_HASH = _hasher.hash("edge-lab-timing-dummy")


def hash_password(password: str) -> str:
    """Return a salted Argon2id hash string."""
    return _hasher.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """True only if `password` matches `stored_hash`. Never raises on bad input."""
    try:
        return _hasher.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def burn_time(password: str) -> None:
    """Spend the same effort as a real check (used for unknown usernames)."""
    verify_password(_DUMMY_HASH, password)
