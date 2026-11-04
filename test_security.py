"""Password hashing."""

from app.security import hash_password, verify_password


def test_hash_is_not_the_password_and_is_argon2():
    hashed = hash_password("correct-horse-9")
    assert hashed != "correct-horse-9"
    assert hashed.startswith("$argon2id$")


def test_same_password_gives_different_hashes_because_of_salt():
    assert hash_password("correct-horse-9") != hash_password("correct-horse-9")


def test_verify_accepts_right_and_rejects_wrong_password():
    hashed = hash_password("correct-horse-9")
    assert verify_password(hashed, "correct-horse-9") is True
    assert verify_password(hashed, "correct-horse-8") is False
    assert verify_password(hashed, "") is False


def test_verify_never_raises_on_garbage_hashes():
    assert verify_password("!", "anything") is False
    assert verify_password("", "anything") is False
    assert verify_password("not-a-real-hash", "anything") is False
