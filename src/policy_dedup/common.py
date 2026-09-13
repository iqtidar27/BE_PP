#!/usr/bin/env python3
"""Shared extraction, normalization, MinHash/LSH, and I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

POLICY_EXTS = {".md", ".txt", ".html", ".htm", ".pdf"}
CHROME_ID_RE = re.compile(r"([a-p]{32})")
DATE_LINE_RE = re.compile(
    r"^\s*(last\s*updated|effective\s*date|updated\s*on|date\s*of\s*last\s*revision)"
    r"\s*[:\-]?\s*.{0,80}$",
    re.I,
)
URL_LINE_RE = re.compile(
    r"^\s*(source|url|fetched from|downloaded from)\s*[:\-]\s*\S+\s*$", re.I
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.DOTALL
)
LEGAL_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|llc|ltd|l\.l\.c|l\.t\.d|gmbh|srl|s\.r\.l|"
    r"co|corp|corporation|company|limited|team|studio|labs|lab|"
    r"dev|developer)\b",
    re.I,
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
WORD_RE = re.compile(r"[a-z0-9]+")
CONSUMER_EMAIL_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "aol.com",
}


def setup_logging(out_dir: Path, name: str = "dedup") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = logging.FileHandler(out_dir / "dedup.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_extensions(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace").lstrip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = []
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return {r["extension_id"].strip(): r for r in rows if r.get("extension_id")}


def extract_extension_id(filename: str) -> str:
    m = CHROME_ID_RE.search(Path(filename).name.lower())
    return m.group(1) if m else ""


def _host(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url if "://" in url else "https://" + url)
        host = (p.netloc or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def normalize_dev_key(meta: dict | None, extension_id: str) -> str:
    meta = meta or {}
    name = (meta.get("developer_name") or "").strip()
    email = (meta.get("developer_email") or "").strip().lower()
    website = (meta.get("developer_website") or meta.get("developer_url") or "").strip()
    host = _host(website)
    name_n = NON_ALNUM_RE.sub("", LEGAL_SUFFIX_RE.sub(" ", name.lower())).strip()
    domain = email.split("@")[1] if "@" in email else ""
    if name_n and len(name_n) >= 3:
        return f"name:{name_n}"
    if host:
        return f"host:{host}"
    if domain and domain not in CONSUMER_EMAIL_DOMAINS:
        return f"emaildom:{domain}"
    if email:
        return f"email:{email}"
    return f"unknown:{extension_id or 'nofile'}"


def extract_raw_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        try:
            from pypdf import PdfReader

            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
            if text.strip():
                return text
        except Exception:
            pass
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    raw = path.read_bytes().decode("utf-8", errors="replace")
    if suf in {".html", ".htm"}:
        raw = SCRIPT_STYLE_RE.sub(" ", raw)
        try:
            from bs4 import BeautifulSoup

            return BeautifulSoup(raw, "lxml").get_text("\n")
        except Exception:
            return HTML_TAG_RE.sub(" ", raw)
    return raw


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if DATE_LINE_RE.match(s) or URL_LINE_RE.match(s):
            continue
        lines.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    text = re.sub(r"[ \t]+", " ", text)
    return text.lower().strip()


def words_of(norm: str) -> list[str]:
    return WORD_RE.findall(norm)


def shingles(word_list: list[str], k: int = 5) -> set[int]:
    if len(word_list) < k:
        blob = " ".join(word_list)
        return {int(hashlib.sha1(blob.encode()).hexdigest()[:16], 16)} if blob else set()
    out: set[int] = set()
    for i in range(len(word_list) - k + 1):
        gram = " ".join(word_list[i : i + k])
        out.add(int(hashlib.sha1(gram.encode("utf-8")).hexdigest()[:16], 16))
    return out


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42):
        rng = random.Random(seed)
        self.P = (1 << 61) - 1
        self.n = num_perm
        self.a = [rng.randrange(1, self.P) for _ in range(num_perm)]
        self.b = [rng.randrange(0, self.P) for _ in range(num_perm)]

    def signature(self, shingle_ids: set[int]) -> list[int]:
        if not shingle_ids:
            return [self.P] * self.n
        sig = [self.P] * self.n
        for s in shingle_ids:
            for i in range(self.n):
                hv = (self.a[i] * (s % self.P) + self.b[i]) % self.P
                if hv < sig[i]:
                    sig[i] = hv
        return sig


def estimated_jaccard(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 0.0


def lsh_candidates(
    indexed: list[tuple[int, list[int]]], bands: int, rows: int, max_bucket: int = 400
) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for loc, sig in indexed:
        for b in range(bands):
            sl = tuple(sig[b * rows : (b + 1) * rows])
            key = hashlib.md5(repr((b, sl)).encode()).hexdigest()
            buckets[(b, key)].append(loc)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        members = sorted(set(members))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}
        self.r: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.p:
            self.p[x] = x
            self.r[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def jaccard_bin(j: float) -> str:
    if j >= 0.99:
        return "0.99-1.00"
    if j >= 0.98:
        return "0.98-0.99"
    if j >= 0.96:
        return "0.96-0.98"
    if j >= 0.94:
        return "0.94-0.96"
    if j >= 0.92:
        return "0.92-0.94"
    if j >= 0.90:
        return "0.90-0.92"
    return "below-0.90"
