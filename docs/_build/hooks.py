"""Keep repository Markdown links usable in the generated documentation site."""

import re
from pathlib import Path
from urllib.parse import quote

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/SII-LiuLab/manimux"
REVISION = "main"


def on_page_markdown(markdown, *, page, config, files):
    """Route source-code links outside docs to the matching repository branch."""
    source = Path(page.file.abs_src_path)
    docs = Path(config["docs_dir"]).resolve()

    def replace(match):
        target = match.group(2)
        if target.startswith(("http:", "https:", "mailto:", "#", "data:")):
            return match.group(0)
        path, separator, anchor = target.partition("#")
        resolved = (source.parent / path).resolve()
        if resolved.is_relative_to(docs) or not resolved.is_relative_to(ROOT):
            return match.group(0)
        relative = resolved.relative_to(ROOT).as_posix()
        kind = "tree" if resolved.is_dir() else "blob"
        url = f"{REPOSITORY}/{kind}/{REVISION}/{quote(relative)}"
        return match.group(1) + url + (separator + anchor if separator else "") + match.group(3)

    markdown = re.sub(r"(\]\()([^\s)]+)(\))", replace, markdown)

    def html_link(match):
        target = (source.parent / match.group(1)).resolve().relative_to(docs)
        document = files.get_file_from_path(target.as_posix())
        if document is None:
            raise ValueError(f"Missing documentation card target: {target}")
        return f'href="{document.url_relative_to(page.file)}"'

    # Card links remain normal Markdown-file paths when viewed on GitHub.
    return re.sub(r'href="([^":#]+\.md)"', html_link, markdown)


def on_files(files, *, config):
    """Publish existing demo media without duplicating it in the source tree."""
    media = {
        "assets/media/robogui.webp": ROOT / "assets/manimux-robogui-demo.webp",
        "assets/media/demo.mp4": ROOT / "assets" / (
            "manimux_2026-09-05_23-17-35-00.00.03.144-00.00.34.914-"
            "seg1-00.00.02.596-00.00.34.966.mp4"
        ),
    }
    for target, source in media.items():
        files.append(File.generated(config, target, abs_src_path=source))
    return files
