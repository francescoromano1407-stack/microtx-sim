from __future__ import annotations

from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocumentationTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self) -> None:
        documents = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
        missing: list[str] = []

        for document in documents:
            text = document.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK.findall(text):
                target = raw_target.strip().split(maxsplit=1)[0]
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                resolved = (document.parent / unquote(parsed.path)).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    missing.append(
                        f"{document.relative_to(ROOT)} -> {target}"
                    )

        self.assertEqual(missing, [], "broken local documentation links")


if __name__ == "__main__":
    unittest.main()
