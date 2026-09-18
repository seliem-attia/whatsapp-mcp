"""Tests for local voice-note transcription.

Two failure modes here are worth guarding permanently. Hand-quoted SQL broke on
apostrophes and skipped exactly the rows whose transcript contained one, and a
database write that failed silently left the text on disk but invisible to
clients — which is indistinguishable from "this note was never transcribed".
"""

import sqlite3
import subprocess

import pytest

import transcription


def _make_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE messages (
            id TEXT,
            chat_jid TEXT,
            content TEXT,
            media_type TEXT,
            PRIMARY KEY (id, chat_jid)
        );
        """
    )
    connection.executemany(
        "INSERT INTO messages (id, chat_jid, content, media_type) VALUES (?, ?, ?, ?)",
        [
            ("voice1", "chat@g.us", "", "audio"),
            ("voice2", "chat@g.us", None, "audio"),
            ("text1", "chat@g.us", "a message someone typed", ""),
            ("voice3", "chat@g.us", "a caption a human wrote", "audio"),
        ],
    )
    connection.commit()
    connection.close()
    return str(path)


@pytest.fixture
def db(tmp_path):
    return _make_db(tmp_path / "messages.db")


def _content(db_path, message_id):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute("SELECT content FROM messages WHERE id = ?", (message_id,)).fetchone()[0]
    finally:
        connection.close()


def test_empty_content_is_filled(db):
    assert transcription.store_transcript(db, "voice1", "hello there", "ggml-small.bin")
    assert _content(db, "voice1").endswith("hello there")
    assert _content(db, "voice1").startswith(transcription.TRANSCRIPT_MARKER)


def test_null_content_is_filled(db):
    assert transcription.store_transcript(db, "voice2", "hello", "ggml-small.bin")
    assert _content(db, "voice2").endswith("hello")


def test_apostrophes_survive(db):
    """The regression that motivated parameter binding."""
    text = "Hey, ich hoffe, dir geht's gut — we can't skip this one"
    assert transcription.store_transcript(db, "voice1", text, "ggml-small.bin")
    assert _content(db, "voice1").endswith(text)


def test_backslashes_and_quotes_survive(db):
    text = "a \\ backslash, a \"quote\" and a 'single' one"
    assert transcription.store_transcript(db, "voice1", text, "ggml-small.bin")
    assert _content(db, "voice1").endswith(text)


def test_an_earlier_transcript_is_refreshed(db):
    transcription.store_transcript(db, "voice1", "first pass", "ggml-small.bin")
    transcription.store_transcript(db, "voice1", "better pass", "ggml-large-v3-turbo.bin")
    stored = _content(db, "voice1")
    assert stored.endswith("better pass")
    assert "large-v3-turbo" in stored


def test_a_human_written_message_is_never_overwritten(db):
    assert not transcription.store_transcript(db, "voice3", "machine text", "ggml-small.bin")
    assert _content(db, "voice3") == "a caption a human wrote"


def test_non_audio_rows_are_left_alone(db):
    assert not transcription.store_transcript(db, "text1", "machine text", "ggml-small.bin")
    assert _content(db, "text1") == "a message someone typed"


def test_a_failed_write_is_reported_rather_than_assumed(db, monkeypatch):
    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(transcription.sqlite3, "connect", fail)
    monkeypatch.setattr(transcription.time, "sleep", lambda _s: None)
    assert not transcription.store_transcript(db, "voice1", "text", "m.bin", attempts=2)


def test_stored_transcript_only_reports_its_own_output(db):
    assert transcription.stored_transcript(db, "voice1") is None
    transcription.store_transcript(db, "voice1", "words", "m.bin")
    assert transcription.stored_transcript(db, "voice1").endswith("words")
    assert transcription.stored_transcript(db, "voice3") is None


def test_missing_model_configuration_says_what_to_set(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    with pytest.raises(transcription.TranscriptionError) as excinfo:
        transcription.model_path()
    assert "WHISPER_MODEL" in str(excinfo.value)


def test_model_pointing_nowhere_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_MODEL", str(tmp_path / "absent.bin"))
    with pytest.raises(transcription.TranscriptionError):
        transcription.model_path()


def test_output_is_collapsed_to_one_line():
    assert transcription.normalise(" a\n b \n\n c \n") == "a b c"


def test_decode_targets_mono_16k_pcm(tmp_path):
    command = transcription.decode_command(tmp_path / "a.ogg", tmp_path / "a.wav")
    assert command[command.index("-ar") + 1] == transcription.TARGET_SAMPLE_RATE
    assert command[command.index("-ac") + 1] == "1"


def test_whisper_runs_without_timestamps(tmp_path):
    command = transcription.whisper_command("m.bin", tmp_path / "a.wav", "de")
    assert "-nt" in command
    assert command[command.index("-l") + 1] == "de"


def test_transcribe_file_returns_recognised_text(tmp_path, monkeypatch):
    audio = tmp_path / "audio_1.ogg"
    audio.write_bytes(b"opus")
    monkeypatch.setattr(transcription.shutil, "which", lambda _tool: "/usr/bin/tool")

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            transcription.Path(command[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, b"  recognised\n  words \n", b"")

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    assert transcription.transcribe_file(audio, tmp_path, model="m.bin") == "recognised words"


def test_transcribe_file_surfaces_a_decode_failure(tmp_path, monkeypatch):
    audio = tmp_path / "audio_1.ogg"
    audio.write_bytes(b"broken")
    monkeypatch.setattr(transcription.shutil, "which", lambda _tool: "/usr/bin/tool")
    monkeypatch.setattr(
        transcription.subprocess,
        "run",
        lambda command, **_k: subprocess.CompletedProcess(command, 1, b"", b"Invalid data"),
    )
    with pytest.raises(transcription.TranscriptionError) as excinfo:
        transcription.transcribe_file(audio, tmp_path, model="m.bin")
    assert "Invalid data" in str(excinfo.value)


def test_missing_whisper_binary_says_what_to_install(tmp_path, monkeypatch):
    audio = tmp_path / "audio_1.ogg"
    audio.write_bytes(b"opus")
    monkeypatch.setattr(
        transcription.shutil,
        "which",
        lambda tool: "/usr/bin/ffmpeg" if tool == "ffmpeg" else None,
    )
    with pytest.raises(transcription.TranscriptionError) as excinfo:
        transcription.transcribe_file(audio, tmp_path, model="m.bin")
    assert "whisper" in str(excinfo.value).lower()
