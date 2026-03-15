"""Transparent AES-256-GCM encryption for GitDB object storage.

When enabled, every blob written to .gitdb/ is encrypted before writing
and decrypted on read. Thread-safe — each encrypt call generates its own
random nonce.

Wire format:
    [10B magic][12B nonce][ciphertext + 16B GCM tag]

Magic header: b"GITDB_ENC\\x01"
"""

import os
import hashlib
import threading
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

MAGIC_HEADER = b"GITDB_ENC\x01"  # 10 bytes
NONCE_SIZE = 12
KEY_SIZE = 32
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT_SIZE = 16


class EncryptionError(Exception):
    """Raised on encryption/decryption failures."""


class EncryptionManager:
    """Transparent encryption for GitDB object storage.

    Key sources (checked in order):
    1. GITDB_KEY environment variable (hex-encoded 32-byte key)
    2. GITDB_KEY_FILE environment variable (path to file containing key)
    3. .gitdb/keyfile (local keyfile, NOT committed)
    4. None = no encryption

    Thread-safe: each encrypt() call uses a fresh random nonce.
    """

    def __init__(self, key: Optional[bytes] = None):
        if key is not None:
            if len(key) != KEY_SIZE:
                raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")
        self._key = key
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """True if a key is loaded."""
        return self._key is not None

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a random 256-bit key."""
        return os.urandom(KEY_SIZE)

    @classmethod
    def from_env(cls) -> Optional["EncryptionManager"]:
        """Try to load key from environment, then .gitdb/keyfile.

        Returns None if no key source is found.
        """
        # 1. GITDB_KEY env var (hex)
        hex_key = os.environ.get("GITDB_KEY")
        if hex_key:
            try:
                key = bytes.fromhex(hex_key)
            except ValueError as e:
                raise EncryptionError(f"GITDB_KEY is not valid hex: {e}")
            return cls(key)

        # 2. GITDB_KEY_FILE env var
        key_file = os.environ.get("GITDB_KEY_FILE")
        if key_file:
            path = Path(key_file)
            if not path.exists():
                raise EncryptionError(f"GITDB_KEY_FILE does not exist: {key_file}")
            key = path.read_bytes().strip()
            if len(key) == KEY_SIZE * 2:
                # Looks hex-encoded
                try:
                    key = bytes.fromhex(key.decode("ascii"))
                except (ValueError, UnicodeDecodeError):
                    pass
            if len(key) != KEY_SIZE:
                raise EncryptionError(
                    f"Key file must contain {KEY_SIZE} bytes, got {len(key)}"
                )
            return cls(key)

        # 3. .gitdb/keyfile in cwd
        local_keyfile = Path.cwd() / ".gitdb" / "keyfile"
        if local_keyfile.exists():
            raw = local_keyfile.read_bytes()
            if len(raw) == KEY_SIZE:
                return cls(raw)
            # Try hex
            try:
                key = bytes.fromhex(raw.strip().decode("ascii"))
                if len(key) == KEY_SIZE:
                    return cls(key)
            except (ValueError, UnicodeDecodeError):
                pass
            raise EncryptionError(
                f"Local keyfile has invalid key length: {len(raw)} bytes"
            )

        return None

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> "EncryptionManager":
        """Derive a 256-bit key from a password using PBKDF2-HMAC-SHA256.

        If salt is None, generates a random salt and stores it as an attribute
        on the returned instance (instance.salt).
        """
        if salt is None:
            salt = os.urandom(PBKDF2_SALT_SIZE)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        key = kdf.derive(password.encode("utf-8"))
        mgr = cls(key)
        mgr.salt = salt
        return mgr

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt with AES-256-GCM.

        Returns: magic_header + nonce + ciphertext_with_tag
        """
        if not self.enabled:
            raise EncryptionError("No encryption key loaded")

        nonce = os.urandom(NONCE_SIZE)
        aesgcm = AESGCM(self._key)
        ct = aesgcm.encrypt(nonce, plaintext, None)
        return MAGIC_HEADER + nonce + ct

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt AES-256-GCM data.

        Input must start with magic header, followed by nonce + ciphertext + tag.
        """
        if not self.enabled:
            raise EncryptionError("No encryption key loaded")

        if not data.startswith(MAGIC_HEADER):
            raise EncryptionError("Data does not have GITDB_ENC header — not encrypted or corrupt")

        payload = data[len(MAGIC_HEADER):]
        if len(payload) < NONCE_SIZE + 16:
            raise EncryptionError("Encrypted data too short")

        nonce = payload[:NONCE_SIZE]
        ct = payload[NONCE_SIZE:]

        aesgcm = AESGCM(self._key)
        try:
            return aesgcm.decrypt(nonce, ct, None)
        except Exception as e:
            raise EncryptionError(f"Decryption failed (wrong key or corrupt data): {e}")

    def encrypt_file(self, path: Path) -> None:
        """Encrypt a file in place."""
        path = Path(path)
        plaintext = path.read_bytes()
        encrypted = self.encrypt(plaintext)
        path.write_bytes(encrypted)

    def decrypt_file(self, path: Path) -> bytes:
        """Read and decrypt a file, returning the plaintext.

        Does NOT modify the file on disk.
        """
        path = Path(path)
        data = path.read_bytes()
        return self.decrypt(data)

    @staticmethod
    def is_encrypted(data: bytes) -> bool:
        """Check if data starts with the GITDB_ENC magic header."""
        return data[:len(MAGIC_HEADER)] == MAGIC_HEADER

    @staticmethod
    def is_encrypted_file(path: Path) -> bool:
        """Check if a file starts with the GITDB_ENC magic header."""
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            header = f.read(len(MAGIC_HEADER))
        return header == MAGIC_HEADER


def init_keyfile(directory: Path) -> bytes:
    """Generate a new keyfile at directory/.gitdb/keyfile.

    Returns the generated key. Creates .gitdb/ if needed.
    Raises if keyfile already exists.
    """
    directory = Path(directory)
    gitdb_dir = directory / ".gitdb"
    gitdb_dir.mkdir(parents=True, exist_ok=True)

    keyfile = gitdb_dir / "keyfile"
    if keyfile.exists():
        raise EncryptionError("Keyfile already exists — refusing to overwrite")

    key = EncryptionManager.generate_key()
    keyfile.write_bytes(key)
    keyfile.chmod(0o600)

    # Ensure .gitignore includes keyfile
    gitignore = directory / ".gitignore"
    patterns = set()
    if gitignore.exists():
        patterns = set(gitignore.read_text().splitlines())
    if ".gitdb/keyfile" not in patterns:
        with open(gitignore, "a") as f:
            f.write("\n.gitdb/keyfile\n")

    return key


def encryption_status(directory: Path) -> dict:
    """Return a dict describing the encryption state for a GitDB repo."""
    directory = Path(directory)
    keyfile = directory / ".gitdb" / "keyfile"

    info = {
        "enabled": False,
        "source": None,
        "keyfile_exists": keyfile.exists(),
    }

    if os.environ.get("GITDB_KEY"):
        info["enabled"] = True
        info["source"] = "GITDB_KEY env var"
    elif os.environ.get("GITDB_KEY_FILE"):
        info["enabled"] = True
        info["source"] = f"GITDB_KEY_FILE ({os.environ['GITDB_KEY_FILE']})"
    elif keyfile.exists():
        info["enabled"] = True
        info["source"] = str(keyfile)

    return info
