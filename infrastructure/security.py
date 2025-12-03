"""Security utilities."""

import hashlib
import os
from typing import Optional


def hash_password(password: str) -> str:
    """Hash password securely."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + key.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify password against stored hash."""
    salt = bytes.fromhex(stored_password[:64])
    stored_key = stored_password[64:]
    key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    return key.hex() == stored_key


def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data (placeholder)."""
    # In production, use proper encryption library (e.g., cryptography)
    return data


def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt data (placeholder)."""
    # In production, use proper encryption library
    return encrypted_data
