"""Tests for GitDB encryption-at-rest module."""

import os
import threading
from pathlib import Path

import pytest

from gitdb.encryption import (
    MAGIC_HEADER,
    KEY_SIZE,
    NONCE_SIZE,
    EncryptionError,
    EncryptionManager,
    init_keyfile,
    encryption_status,
)


@pytest.fixture
def key():
    return EncryptionManager.generate_key()


@pytest.fixture
def mgr(key):
    return EncryptionManager(key)


# --- Key generation ---

class TestKeyGeneration:
    def test_generate_key_length(self):
        key = EncryptionManager.generate_key()
        assert len(key) == KEY_SIZE

    def test_generate_key_randomness(self):
        k1 = EncryptionManager.generate_key()
        k2 = EncryptionManager.generate_key()
        assert k1 != k2

    def test_reject_wrong_key_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptionManager(b"short")


# --- Encrypt / decrypt round-trip ---

class TestRoundTrip:
    def test_basic_round_trip(self, mgr):
        plaintext = b"hello gitdb"
        ct = mgr.encrypt(plaintext)
        assert mgr.decrypt(ct) == plaintext

    def test_round_trip_large(self, mgr):
        plaintext = os.urandom(1_000_000)
        ct = mgr.encrypt(plaintext)
        assert mgr.decrypt(ct) == plaintext

    def test_empty_data(self, mgr):
        ct = mgr.encrypt(b"")
        assert mgr.decrypt(ct) == b""

    def test_unique_ciphertexts(self, mgr):
        """Same plaintext should produce different ciphertexts (random nonce)."""
        pt = b"same data"
        c1 = mgr.encrypt(pt)
        c2 = mgr.encrypt(pt)
        assert c1 != c2
        assert mgr.decrypt(c1) == mgr.decrypt(c2) == pt


# --- Wrong key ---

class TestWrongKey:
    def test_wrong_key_fails(self, mgr):
        ct = mgr.encrypt(b"secret")
        other = EncryptionManager(EncryptionManager.generate_key())
        with pytest.raises(EncryptionError, match="Decryption failed"):
            other.decrypt(ct)

    def test_tampered_ciphertext_fails(self, mgr):
        ct = bytearray(mgr.encrypt(b"secret"))
        ct[-1] ^= 0xFF  # flip last byte (tag)
        with pytest.raises(EncryptionError):
            mgr.decrypt(bytes(ct))


# --- Magic header ---

class TestMagicHeader:
    def test_header_present(self, mgr):
        ct = mgr.encrypt(b"data")
        assert ct[:len(MAGIC_HEADER)] == MAGIC_HEADER

    def test_is_encrypted(self, mgr):
        ct = mgr.encrypt(b"data")
        assert EncryptionManager.is_encrypted(ct)
        assert not EncryptionManager.is_encrypted(b"plain data")

    def test_no_header_raises(self, mgr):
        with pytest.raises(EncryptionError, match="header"):
            mgr.decrypt(b"this is not encrypted")

    def test_truncated_data_raises(self, mgr):
        # Header only, no nonce or ciphertext
        with pytest.raises(EncryptionError, match="too short"):
            mgr.decrypt(MAGIC_HEADER + b"\x00" * 10)


# --- Password-based key derivation ---

class TestPasswordDerivation:
    def test_from_password_round_trip(self):
        mgr = EncryptionManager.from_password("hunter2")
        ct = mgr.encrypt(b"secret payload")
        assert mgr.decrypt(ct) == b"secret payload"

    def test_same_password_same_salt_same_key(self):
        m1 = EncryptionManager.from_password("pass", salt=b"0" * 16)
        m2 = EncryptionManager.from_password("pass", salt=b"0" * 16)
        assert m1._key == m2._key

    def test_different_salt_different_key(self):
        m1 = EncryptionManager.from_password("pass", salt=b"A" * 16)
        m2 = EncryptionManager.from_password("pass", salt=b"B" * 16)
        assert m1._key != m2._key


# --- File encryption ---

class TestFileEncryption:
    def test_encrypt_decrypt_file(self, mgr, tmp_path):
        f = tmp_path / "data.bin"
        original = b"file contents here"
        f.write_bytes(original)

        mgr.encrypt_file(f)
        assert f.read_bytes() != original
        assert EncryptionManager.is_encrypted_file(f)

        recovered = mgr.decrypt_file(f)
        assert recovered == original

    def test_is_encrypted_file_false(self, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_bytes(b"just text")
        assert not EncryptionManager.is_encrypted_file(f)

    def test_is_encrypted_file_missing(self, tmp_path):
        assert not EncryptionManager.is_encrypted_file(tmp_path / "nope")


# --- Keyfile init ---

class TestKeyfileInit:
    def test_init_creates_keyfile(self, tmp_path):
        key = init_keyfile(tmp_path)
        assert len(key) == KEY_SIZE
        kf = tmp_path / ".gitdb" / "keyfile"
        assert kf.exists()
        assert kf.read_bytes() == key
        assert oct(kf.stat().st_mode)[-3:] == "600"

    def test_init_refuses_overwrite(self, tmp_path):
        init_keyfile(tmp_path)
        with pytest.raises(EncryptionError, match="already exists"):
            init_keyfile(tmp_path)

    def test_gitignore_updated(self, tmp_path):
        init_keyfile(tmp_path)
        gi = (tmp_path / ".gitignore").read_text()
        assert ".gitdb/keyfile" in gi


# --- Env loading ---

class TestFromEnv:
    def test_from_env_hex_key(self, monkeypatch, key):
        monkeypatch.setenv("GITDB_KEY", key.hex())
        mgr = EncryptionManager.from_env()
        assert mgr is not None
        assert mgr._key == key

    def test_from_env_key_file(self, monkeypatch, key, tmp_path):
        kf = tmp_path / "mykey"
        kf.write_bytes(key)
        monkeypatch.setenv("GITDB_KEY_FILE", str(kf))
        monkeypatch.delenv("GITDB_KEY", raising=False)
        mgr = EncryptionManager.from_env()
        assert mgr is not None
        assert mgr._key == key

    def test_from_env_none(self, monkeypatch):
        monkeypatch.delenv("GITDB_KEY", raising=False)
        monkeypatch.delenv("GITDB_KEY_FILE", raising=False)
        # Won't find .gitdb/keyfile in tmp either
        assert EncryptionManager.from_env() is None


# --- Status ---

class TestEncryptionStatus:
    def test_status_no_encryption(self, tmp_path):
        s = encryption_status(tmp_path)
        assert not s["enabled"]

    def test_status_with_keyfile(self, tmp_path):
        init_keyfile(tmp_path)
        s = encryption_status(tmp_path)
        assert s["enabled"]
        assert s["keyfile_exists"]


# --- Thread safety ---

class TestThreadSafety:
    def test_concurrent_encrypt_decrypt(self, mgr):
        plaintext = b"thread-safe data"
        errors = []

        def worker():
            try:
                for _ in range(50):
                    ct = mgr.encrypt(plaintext)
                    assert mgr.decrypt(ct) == plaintext
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# --- No key loaded ---

class TestNoKey:
    def test_encrypt_without_key_raises(self):
        mgr = EncryptionManager(key=None)
        assert not mgr.enabled
        with pytest.raises(EncryptionError, match="No encryption key"):
            mgr.encrypt(b"data")

    def test_decrypt_without_key_raises(self):
        mgr = EncryptionManager(key=None)
        with pytest.raises(EncryptionError, match="No encryption key"):
            mgr.decrypt(MAGIC_HEADER + b"\x00" * 30)
