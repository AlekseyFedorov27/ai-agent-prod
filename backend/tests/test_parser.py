import pytest

from app.documents.parser import extract_text, sanitize_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("notes.txt", "notes.txt"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\windows\\evil.md", "evil.md"),
        ("/abs/path/report.PDF", "report.PDF"),
        ("мой документ.docx", "мой документ.docx"),
        ("we<ird>|name?.txt", "we_ird__name_.txt"),
    ],
)
def test_sanitize_filename_keeps_only_safe_name(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize(
    "raw", [None, "", "   ", "..", "script.exe", "noextension", ".txt", "___.txt"]
)
def test_sanitize_filename_rejects_bad_names(raw):
    with pytest.raises(ValueError):
        sanitize_filename(raw)


def test_sanitize_filename_limits_length_and_keeps_extension():
    result = sanitize_filename("a" * 300 + ".txt")
    assert len(result) <= 100
    assert result.endswith(".txt")


def test_extract_text_reads_plain_text(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("Секретный проект «Омега»", encoding="utf-8")
    assert "Омега" in extract_text(path, "a.txt")


def test_extract_text_rejects_unknown_format(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_text(path, "a.bin")
