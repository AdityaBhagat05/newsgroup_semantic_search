
import tarfile
import re
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NewsDocument:
    doc_id: str
    newsgroup: str
    subject: str
    body: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.body.split())

    @property
    def full_text(self) -> str:
        """Combine subject + body for embedding — subject is high-signal."""
        return f"{self.subject}. {self.body}" if self.subject else self.body


_JUNK_HEADERS = re.compile(
    r'^(From|Reply-To|NNTP-Posting-Host|Organization|Lines|Message-ID|'
    r'References|Distribution|X-Newsreader|Newsgroups|Path|Date|'
    r'Xref|Article-I\.D\.|Sender|Approved|Expires|Followup-To):.+$',
    re.MULTILINE | re.IGNORECASE,
)
_QUOTED_LINES = re.compile(r'^[ \t]*[>|].*$', re.MULTILINE)
_EMAIL = re.compile(r'\S+@\S+\.\S+')
_URL = re.compile(r'https?://\S+|www\.\S+')
_WHITESPACE = re.compile(r'\n{3,}')
_SHORT_LINES = re.compile(r'^.{0,2}$', re.MULTILINE) 


def _clean_body(raw: str) -> tuple[str, str]:
    """
    Returns (subject, cleaned_body).
    Splits on the blank line separating headers from body.
    """
    parts = raw.split('\n\n', 1)
    header_block = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    subject_match = re.search(r'^Subject:\s*(.+)$', header_block, re.MULTILINE | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ""
    subject = re.sub(r'^(Re|Fwd|AW|WG):\s*', '', subject, flags=re.IGNORECASE).strip()

    sig_split = re.split(r'\n--\s*\n', body, maxsplit=1)
    body = sig_split[0]

    body = _QUOTED_LINES.sub('', body)

    body = _EMAIL.sub('', body)
    body = _URL.sub('', body)

    body = _WHITESPACE.sub('\n\n', body)
    body = _SHORT_LINES.sub('', body)
    body = body.strip()

    return subject, body


def load_corpus(archive_path: str, min_words: int = 50) -> list[NewsDocument]:
    docs = []
    skipped = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        print(f"Found {len(members)} raw files in archive.")

        for member in members:
            f = tar.extractfile(member)
            if f is None:
                continue

            raw = f.read().decode('utf-8', errors='ignore')
            subject, body = _clean_body(raw)

            parts = member.name.split('/')
            newsgroup = parts[1] if len(parts) >= 3 else "unknown"
            doc_id = member.name.replace('/', '_')

            doc = NewsDocument(
                doc_id=doc_id,
                newsgroup=newsgroup,
                subject=subject,
                body=body,
            )

            if doc.word_count < min_words:
                skipped += 1
                continue

            docs.append(doc)

    print(f"Loaded {len(docs)} documents ({skipped} dropped, < {min_words} words after cleaning).")
    return docs


def get_newsgroup_stats(docs: list[NewsDocument]) -> dict:
    """Return per-newsgroup document counts for sanity checking."""
    from collections import Counter
    counts = Counter(d.newsgroup for d in docs)
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    archive = r"../data/original/20_newsgroups.tar.gz"
    docs = load_corpus(archive)
    stats = get_newsgroup_stats(docs)
    print("\nPer-newsgroup counts:")
    for ng, count in stats.items():
        print(f"  {ng:<40} {count}")