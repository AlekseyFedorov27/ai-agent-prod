import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
_MAX_NAME_LENGTH = 100


def sanitize_filename(raw: str | None) -> str:
    """Приводит имя файла от клиента к безопасному виду.

    Убирает любые каталоги (в том числе с обратными слэшами Windows и
    последовательности `..`), заменяет странные символы, проверяет расширение.
    Бросает ValueError с понятным сообщением, если имя непригодно.
    """
    if not raw:
        raise ValueError("Не указано имя файла")

    # Берём только последний компонент пути, учитывая и / и \
    name = Path(raw.replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]", "_", name).strip(" .")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Неподдерживаемый формат файла. Допустимо: {allowed}")

    stem = Path(name).stem
    if not stem.strip("_ "):
        raise ValueError("Недопустимое имя файла")

    if len(name) > _MAX_NAME_LENGTH:
        name = stem[: _MAX_NAME_LENGTH - len(suffix)] + suffix
    return name


def extract_text(file_path: Path, filename: str) -> str:
    """Извлекает текст из поддерживаемых форматов (синхронно, вызывать в потоке)."""
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        import docx

        document = docx.Document(str(file_path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise ValueError("Неподдерживаемый формат файла")
