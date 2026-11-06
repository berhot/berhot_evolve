"""Open-web text archive and corpus cleaning using Python's STANDARD LIBRARY ONLY.

Collection is independent of neural-training admission: any publicly accessible
HTML page the existing safe fetcher can retrieve may be archived without a
CC0/PD declaration. We retain URL, decoded HTML snapshot digest, cleaned-text digest,
title, declared rights or UNKNOWN, and no-AI signal as observable metadata.
The archive is NOT a training-data approval, fact verifier, or trained model.

No bypassing robots.txt, authentication, paywalls, rate limits, or access blocks.
Exact normalized-text and decoded-HTML-snapshot hashing; NOT semantic deduplication.
"""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import gzip
import hashlib
import re
import unicodedata


def normalize_web_text(text):
    """Conservative NFC, invisible/control removal, whitespace cleanup."""
    if not isinstance(text, str):
        raise TypeError("Expected extracted Unicode text")
    text = unicodedata.normalize("NFC", text)
    cleaned = []
    for char in text:
        cat = unicodedata.category(char)
        if char.isspace():
            cleaned.append(" ")
        elif cat in ("Cf", "Cc", "Cs"):
            continue
        else:
            cleaned.append(char)
    text = re.sub(r"\s+", " ", "".join(cleaned)).strip()
    return text


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _quality(text):
    words = re.findall(r"\b[^\W_]+\b", text, re.UNICODE)
    # A documented heuristic diagnostic, not a proof of factual quality.
    if not words:
        return {"heuristic_score": 0, "word_count": 0}
    diversity = len({w.casefold() for w in words[:2000]}) / min(len(words), 2000)
    length_component = min(len(words) / 150, 1.0)
    return {
        "heuristic_score": round(100 * (.55 * length_component + .45 * diversity), 2),
        "word_count": len(words),
    }


def _immutable_file(base, digest, content, suffix):
    """Content-addressed blobs and atomic writes; same hash is stored once."""
    filename = base / digest[:2] / (digest + suffix)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not filename.exists():
        pending = filename.with_name(filename.name + ".tmp")
        pending.write_bytes(content)
        pending.replace(filename)
    return filename


def archive_page(store, url, html, extracted_text, title="", noai=False,
                 declared_license=None, rights_url=None):
    """Keep text from public HTML without requiring a reusable declaration.

    The caller MUST use fetch_page() or a trusted in-memory fixture. This
    function does no networking. It never inserts into neural training docs.
    """
    if not isinstance(html, str):
        raise TypeError("HTML must be Unicode")
    text = normalize_web_text(extracted_text)
    if not text:
        return {"status": "skipped", "reason": "Empty extracted text"}
    # fetch_page returns decoded HTML Unicode; this preserves that exact
    # decoded snapshot, NOT the original HTTP response byte sequence.
    original_bytes = html.encode("utf-8")
    clean_bytes = text.encode("utf-8")
    original_sha = _sha(original_bytes)
    clean_sha = _sha(clean_bytes)
    quality = _quality(text)
    base = store.root / "web_corpus"
    raw_path = _immutable_file(
        base / "raw", original_sha, gzip.compress(original_bytes, mtime=0), ".html.gz")
    clean_path = _immutable_file(
        base / "clean", clean_sha, clean_bytes, ".txt")

    # Distinct URLs can refer to identical cleaned text. Preserve both source
    # records but store only ONE cleaned content blob / unique corpus record.
    existed = bool(store.db.execute(
        "SELECT 1 FROM web_corpus_texts WHERE sha256=?", (clean_sha,)
    ).fetchone())
    url_seen = bool(store.db.execute(
        "SELECT 1 FROM web_corpus_sources WHERE url=?", (url,)
    ).fetchone())
    split_bucket = int(clean_sha[:8], 16) % 100
    split = "train" if split_bucket < 80 else "val" if split_bucket < 90 else "test"
    when = datetime.now(timezone.utc).isoformat()
    store.db.execute(
        "INSERT OR IGNORE INTO web_corpus_texts "
        "(sha256,clean_path,characters,words,quality_score,split,first_seen_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (clean_sha, str(clean_path), len(text), quality["word_count"],
         quality["heuristic_score"], split, when),
    )
    rights_status = declared_license if declared_license else "UNKNOWN"
    store.db.execute(
        "INSERT INTO web_corpus_sources "
        "(url,sha256,raw_html_sha256,raw_path,title,host,rights_status,"
        "rights_evidence_url,noai,observed_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(url) DO UPDATE SET "
        "sha256=excluded.sha256,raw_html_sha256=excluded.raw_html_sha256,"
        "raw_path=excluded.raw_path,title=excluded.title,host=excluded.host,"
        "rights_status=excluded.rights_status,"
        "rights_evidence_url=excluded.rights_evidence_url,noai=excluded.noai,"
        "observed_at=excluded.observed_at",
        (url, clean_sha, original_sha, str(raw_path), str(title)[:300],
         urlsplit(url).hostname or "", rights_status, rights_url,
         int(bool(noai)), when),
    )
    from .training_approval import approve_future_snapshot
    operator_approved = approve_future_snapshot(store, url, clean_sha, noai, when)
    store.db.commit()
    return {
        "status": "archived",
        "new_unique_text": not existed,
        "revisited_url": url_seen,
        "clean_sha256": clean_sha, "raw_sha256": original_sha,
        "characters": len(text), "words": quality["word_count"],
        "quality_heuristic": quality["heuristic_score"],
        "split": split, "rights_status": rights_status,
        "noai": bool(noai), "neural_training_admitted": False,
        "operator_training_approved": operator_approved,
    }


def corpus_status(store):
    """Read-only status: corpus files DO NOT imply trained neural knowledge."""
    texts = store.db.execute(
        "SELECT COUNT(*),COALESCE(SUM(characters),0),"
        "COALESCE(SUM(words),0) FROM web_corpus_texts"
    ).fetchone()
    sources = int(store.db.execute(
        "SELECT COUNT(*) FROM web_corpus_sources").fetchone()[0])
    rights = {str(row[0]): int(row[1]) for row in store.db.execute(
        "SELECT rights_status, COUNT(*) FROM web_corpus_sources "
        "GROUP BY rights_status ORDER BY rights_status")}
    splits = {str(row[0]): int(row[1]) for row in store.db.execute(
        "SELECT split,COUNT(*) FROM web_corpus_texts GROUP BY split")}
    noai = int(store.db.execute(
        "SELECT COUNT(*) FROM web_corpus_sources WHERE noai=1").fetchone()[0])
    from .training_approval import training_condition
    approved = store.db.execute(
        "SELECT COUNT(*) FROM web_corpus_sources WHERE " + training_condition()).fetchone()[0]
    return {
        "status": "audited",
        "urls_collected": sources,
        "unique_cleaned_texts": int(texts[0]),
        "duplicate_url_content_references": max(0, sources - int(texts[0])),
        "unique_cleaned_characters": int(texts[1]),
        "unique_cleaned_words": int(texts[2]),
        "corpus_splits": splits,
        "rights_metadata": rights,
        "noai_source_count": noai,
        "training_approved_from_corpus": approved,
        "neural_training_from_corpus": "approved_focus_plain_shards",
        "plain_focus_dataset": store.get("focus_plain_dataset_latest_saudi-history-ar"),
        "latest_plain_export": store.get("focus_plain_export_latest_saudi-history-ar"),
        "extraction_batches_directory": str(store.root / "corpus_runs"),
        "collected_text_folder": str(store.root / "corpus_runs" / "COLLECTED_TEXT")
            if (store.root / "corpus_runs" / "COLLECTED_TEXT").is_dir() else None,
        "latest_populated_extraction": store.get("plain_extraction_latest_populated_saudi-history-ar"),
        "no_extraction_audit_directory": str(store.root / "collection_audit"),
        "dataset_catalog_root": str(store.root / "datasets" / "focus_plain"),
        "trained_datasets_directory": str(store.root / "datasets" / "focus_plain" / "trained"),
        "active_model_dataset_folder": str(store.root / "datasets" / "focus_plain" / "active" / "saudi-history-ar"),
        "collection_implementation": "BERHOT-owned Python module using stdlib only",
        "deduplication": "SHA256 exact post-normalization plus decoded HTML snapshot digest; no semantic dedup",
        "warning": (
            "Collected text is NOT proof of legal permission to train, factual "
            "correctness, or neural learning. Unknown rights and noai are kept "
            "as metadata. Use/redistribution remain subject to applicable rules."
        ),
    }


def corpus_seed(store, urls, cross_host=True):
    """Queue multiple public HTTPS seed URLs, without requiring CC0/PD license.

    This explicitly enters research-only exploration mode, which ensures
    no web text silently enters existing neural training documents.
    """
    from .web_discovery import canonical_url
    if not 1 <= len(urls) <= 20:
        raise ValueError("Provide 1..20 URLs")
    canonical = [canonical_url(url) for url in urls]
    profile = store.get("web_source_profile", {}) or {}
    profile.update({"mode": "research_only", "active": True,
                    "cross_host": bool(cross_host),
                    "seed_url": canonical[0],
                    "host": urlsplit(canonical[0]).hostname})
    # Prevent stale site-level rights assertions from being interpreted as
    # rights for newly seeded, unrelated sites.
    for key in ("license", "rights_evidence_url", "rights_mode"):
        profile.pop(key, None)
    store.put("web_source_profile", profile)
    added = sum(int(store.add_web_url(url, depth=0)) for url in canonical)
    # RFC 2606 example domains are legitimate URL syntax but deliberately
    # illustrative: warn rather than incorrectly treating them as research
    # content. Do not silently rewrite/remove the user's queue.
    placeholders = [
        url for url in canonical
        if (urlsplit(url).hostname or "").casefold()
        in {"example.org", "example.net", "example.com"}
    ]
    return {
        "status": "queued", "submitted": len(canonical),
        "new_queue_urls": added, "cross_host": bool(cross_host),
        "mode": "research_only",
        "illustrative_domains_queued": placeholders,
        "warning": (
            "Example domains are reserved for documentation and provide no "
            "meaningful training corpus; seed actual content pages instead."
            if placeholders else None),
        "note": "No network requests; queued URLs may be skipped by robots/security rules.",
    }
