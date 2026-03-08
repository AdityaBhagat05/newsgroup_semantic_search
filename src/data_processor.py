

"""
data_processor.py
-----------------
Loads and cleans the 20 Newsgroups corpus from the raw .tar.gz archive.

Design decisions (preprocessing):
  - Strip full email headers: Things like NNTP-Posting-Host or Organization 
    add noise without semantic content. I kept only the Subject line because 
    it is a highly condensed topical signal.
  - Remove quoted reply text: Lines starting with > or | duplicate content 
    from other documents and heavily pollute the embeddings. I want to cluster 
    based on novel content, not reply chains.
  - Remove signatures: Everything after the standard signature delimiter 
    is boilerplate, not topical content.
  - Remove URLs and email addresses: I remove these so the model does not 
    cluster documents simply because they share a domain name or contact info.
  - Drop short documents: Documents with fewer than 50 words after cleaning 
    are discarded. They carry too little semantic content to cluster or 
    retrieve meaningfully in a vector space.
"""

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
        """Combine subject and body for embedding. The subject is a high-signal indicator."""
        return f"{self.subject}. {self.body}" if self.subject else self.body


# Headers that carry zero topical signal. I strip them entirely to prevent noise.
_JUNK_HEADERS = re.compile(
    r'^(From|Reply-To|NNTP-Posting-Host|Organization|Lines|Message-ID|'
    r'References|Distribution|X-Newsreader|Newsgroups|Path|Date|'
    r'Xref|Article-I\.D\.|Sender|Approved|Expires|Followup-To):.+$',
    re.MULTILINE | re.IGNORECASE,
)
# Quoted replies duplicate other documents' content.
_QUOTED_LINES = re.compile(r'^[ \t]*[>|].*$', re.MULTILINE)
# Strip emails and URLs as they are identifiers, not semantics.
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
    
    # Extract Subject because it is semantically valuable
    subject_match = re.search(r'^Subject:\s*(.+)$', header_block, re.MULTILINE | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ""
    
    # Clean subject of Re:, Fwd:, etc. so it only contains the core topic
    subject = re.sub(r'^(Re|Fwd|AW|WG):\s*', '', subject, flags=re.IGNORECASE).strip()

    # Drop signature blocks (standard email/usenet sig delimiter)
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

            # Filter out near-empty documents. Empirically, shorter documents 
            # after cleaning are too sparse to produce meaningful embeddings.
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