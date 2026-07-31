"""Password hashing utilities — no app dependencies to avoid circular imports."""
from passlib.context import CryptContext

# pbkdf2_sha256 is the active scheme; bcrypt<4.0 is kept for legacy hash verification
_ctx = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated=["bcrypt"])


def hash_password(password: str) -> str:
    return _ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _ctx.verify(plain, hashed)
