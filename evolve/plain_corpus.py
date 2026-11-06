"""Auditable raw-text batches and immutable plain UTF-8 training datasets."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile

from .open_corpus import normalize_web_text
from .training_approval import training_condition

RUN_FORMAT = "berhot_plain_extraction_run_v1"
DATASET_FORMAT = "berhot_plain_focus_dataset_v1"
SPLIT_FILES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Corpus path escapes its bundle")
    return target


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_directory(store, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,180}", run_id):
        raise ValueError("Invalid extraction run ID")
    return _inside(store.root / "corpus_runs", run_id)


def begin_extraction_run(store, run_id, profile, source_mode="live_collection"):
    directory = _run_directory(store, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "sources.json"
    if not index.exists():
        _atomic_write(index, _json({"format": RUN_FORMAT, "run_id": run_id,
            "profile": profile, "source_mode": source_mode, "started_at_utc": _now(), "pages": []}))
        for name in ("raw.txt", "clean.txt", "corpus.txt"):
            _atomic_write(directory / name, b"")
    return directory


def capture_extraction(store, run_id, sequence, url, title, html, raw_text,
                       profile_match=True, noai=False, clean_text=None, approval_parent_sha256=None):
    """Persist each successful extraction before topic/archive admission."""
    directory = _run_directory(store, run_id)
    index_path = directory / "sources.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (directory / "manifest.json").exists():
        raise ValueError("Cannot append to a finalized extraction run")
    raw = raw_text.encode("utf-8")
    html_bytes = html.encode("utf-8")
    compressed_html = gzip.compress(html_bytes, mtime=0)
    cleaned = (normalize_web_text(raw_text) if clean_text is None else clean_text).encode("utf-8")
    name = f"{int(sequence):06d}-" + _sha(url.encode("utf-8"))[:12]
    raw_file, html_file = "raw_pages/" + name + ".txt", "raw_html/" + name + ".html.gz"
    clean_file = "clean_pages/" + _sha(cleaned) + ".txt"
    for relative, content in ((raw_file, raw), (html_file, compressed_html),
                              (clean_file, cleaned)):
        target = _inside(directory, relative)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Extraction page identity already exists with different bytes")
        if not target.exists():
            _atomic_write(target, content)
    record = {"sequence": int(sequence), "url": url, "title": title,
        "raw_text_file": raw_file, "raw_html_file": html_file, "clean_text_file": clean_file,
        "raw_text_sha256": _sha(raw), "decoded_html_sha256": _sha(html_bytes),
        "raw_html_file_sha256": _sha(compressed_html),
        "clean_sha256": _sha(cleaned), "raw_utf8_bytes": len(raw), "clean_utf8_bytes": len(cleaned),
        "profile_match": bool(profile_match), "publisher_noai": bool(noai), "extracted_at_utc": _now()}
    if approval_parent_sha256 is not None:
        record['approval_parent_sha256']=approval_parent_sha256
        record['cleaning']='article_body_derived_from_approved_parent'
    if not any(p["sequence"] == record["sequence"] for p in index["pages"]):
        first_clean = bool(cleaned) and not any(p["clean_sha256"] == record["clean_sha256"] for p in index["pages"])
        index["pages"].append(record)
        _atomic_write(index_path, _json(index))
        # Individual files and the index are durable first. Finalization can
        # rebuild aggregates after interruption; raw.txt is useful while live.
        for filename, payload in (("raw.txt", raw + b"\n\n"),
                                  ("clean.txt", cleaned + b"\n\n" if first_clean else b"")):
            with (directory / filename).open("ab") as stream:
                stream.write(payload)
    return record


def _verified_bytes(directory, relative, digest):
    data = _inside(directory, relative).read_bytes()
    if _sha(data) != digest:
        raise ValueError("Corpus file SHA-256 mismatch: " + relative)
    return data


def finalize_extraction_run(store, run_id, dataset=None):
    directory = _run_directory(store, run_id)
    existing = directory / "manifest.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    index = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    eligible = {(r["url"], r["sha256"]) for r in store.db.execute(
        "SELECT url,sha256 FROM web_corpus_sources WHERE " + training_condition())}
    raw_parts, clean_parts, corpus_parts = [], [], []
    seen_clean, seen_training = set(), set()
    records = []
    raw_offset, clean_offset, corpus_offset = 0, 0, 0
    for page in index["pages"]:
        raw = _verified_bytes(directory, page["raw_text_file"], page["raw_text_sha256"])
        _verified_bytes(directory, page["raw_html_file"], page["raw_html_file_sha256"])
        clean = _verified_bytes(directory, page["clean_text_file"], page["clean_sha256"])
        record = dict(page, raw_offset=raw_offset, raw_length=len(raw))
        raw_parts.append(raw + b"\n\n")
        raw_offset += len(raw) + 2
        if clean and page["clean_sha256"] not in seen_clean:
            record.update(clean_offset=clean_offset, clean_length=len(clean))
            clean_parts.append(clean + b"\n\n")
            clean_offset += len(clean) + 2
            seen_clean.add(page["clean_sha256"])
        admitted = (page["profile_match"] and not page["publisher_noai"] and
                    (page["url"], page.get('approval_parent_sha256',page["clean_sha256"])) in eligible and bool(clean))
        record["eligible_for_training"] = bool(admitted)
        if admitted and page["clean_sha256"] not in seen_training:
            record.update(corpus_offset=corpus_offset, corpus_length=len(clean))
            corpus_parts.append(clean + b"\n\n")
            corpus_offset += len(clean) + 2
            seen_training.add(page["clean_sha256"])
        records.append(record)
    files = {}
    for name, parts in (("raw.txt", raw_parts), ("clean.txt", clean_parts), ("corpus.txt", corpus_parts)):
        data = b"".join(parts)
        _atomic_write(directory / name, data)
        files[name] = {"sha256": _sha(data), "bytes": len(data)}
    _atomic_write(directory / "documents.jsonl", b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in records))
    result = {k: v for k, v in index.items() if k != "pages"}
    result.update(status="prepared" if records else "no_extraction", finished_at_utc=_now(), directory=str(directory),
        extracted_pages=len(records), unique_clean_texts=len(seen_clean),
        duplicate_clean_texts=sum(bool(r["clean_utf8_bytes"]) for r in records)-len(seen_clean),
        training_eligible_unique_texts=len(seen_training), files=files,
        training_dataset=dataset, document_separator="two UTF-8 newline bytes",
        raw_definition="Decoded HTML snapshot and extracted text before corpus normalization; not original HTTP wire bytes")
    _atomic_write(existing, _json(result))
    if records and files["raw.txt"]["bytes"]:
        from .dataset_catalog import _link
        _link(directory, store.root / "corpus_runs" / "LATEST_WITH_TEXT")
        store.put("plain_extraction_latest_populated_" + index["profile"], result)
    return result


def finish_collection_cycle(store, run_id, profile, dataset=None, *, reason=None, attempted=0):
    """No successful read means an audit event, not an empty text dataset."""
    directory = _run_directory(store, run_id)
    if (directory / "sources.json").is_file():
        return finalize_extraction_run(store, run_id, dataset)
    result = {"status": "no_extraction", "run_id": run_id, "profile": profile,
              "directory": None, "extracted_pages": 0, "unique_clean_texts": 0,
              "duplicate_clean_texts": 0, "training_eligible_unique_texts": 0,
              "training_dataset": dataset, "attempted_page_requests": int(attempted),
              "reason": reason or "no_successful_page_reads", "finished_at_utc": _now()}
    audit = store.root / "collection_audit" / "no_extraction_cycles.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("ab") as stream:
        stream.write((json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8"))
    result["audit_log"] = str(audit)
    return result


def organize_extraction_runs(store):
    """Archive only proven empty finalized bundles; retain all audit evidence."""
    root = store.root / "corpus_runs"
    archive = store.root / "collection_audit" / "empty_batches"
    moved, populated = [], []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if directory.is_symlink() or not directory.is_dir():
            continue
        manifest_path, index_path = directory / "manifest.json", directory / "sources.json"
        if not manifest_path.is_file() or not index_path.is_file():
            continue  # An in-progress or unrecognized bundle is never moved.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if manifest.get("format") != RUN_FORMAT:
            continue
        if manifest.get("extracted_pages", 0) > 0:
            if (directory / "raw.txt").is_file() and (directory / "raw.txt").stat().st_size:
                populated.append((manifest.get("finished_at_utc", ""), directory, manifest))
            continue
        if manifest.get("extracted_pages") != 0 or index.get("pages") != []:
            continue
        metadata = {"manifest.json", "sources.json", ".DS_Store"}
        if any(p.is_file() and p.name not in metadata and p.stat().st_size
               for p in directory.rglob("*")):
            continue  # Never archive a bundle that contains any extracted data.
        destination = archive / directory.name
        if destination.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        mapping = {"old_directory": str(directory), "audit_directory": str(destination),
                   "manifest_sha256": _sha(manifest_path.read_bytes()),
                   "reason": "finalized_batch_extracted_zero_pages", "archived_at_utc": _now()}
        # Write the relocation evidence before moving; original manifest stays
        # byte-for-byte intact even though its historical directory is moved.
        with (archive.parent / "empty_batch_relocations.jsonl").open("ab") as stream:
            stream.write((json.dumps(mapping, ensure_ascii=False) + "\n").encode("utf-8"))
        directory.rename(destination)
        moved.append(mapping)
    if populated:
        from .dataset_catalog import _link
        _, directory, manifest = max(populated, key=lambda entry: entry[0])
        _link(directory, root / "LATEST_WITH_TEXT")
        store.put("plain_extraction_latest_populated_" + manifest["profile"], manifest)
    _atomic_write(root / "README.txt", (
        "COLLECTED WEBSITE TEXT\n\n"
        "Open LATEST_WITH_TEXT for the latest completed nonempty bundle.\n"
        "Open COLLECTED_TEXT for the latest combined saved-corpus export, if present.\n"
        "raw.txt = extracted text; clean.txt = cleaned/deduplicated text.\n"
        "corpus.txt = approved relevant text; it can be empty when no text qualifies.\n"
        "A live bundle may be unfinished. Failed/idle cycles contain no article text.\n"
        "Older zero-page bundles are kept in storage/collection_audit/empty_batches/.\n"
        "Future no-extraction cycles are recorded in collection_audit/no_extraction_cycles.jsonl.\n"
    ).encode("utf-8"))
    return {"empty_batches_archived": len(moved), "populated_batches": len(populated),
            "text_root": str(root), "empty_batch_archive": str(archive),
            "latest_with_text": str(root / "LATEST_WITH_TEXT") if populated else None}


def _source_spec(check):
    return sorted((p["url"], p["sha256"], p["rights"], check["_partitions"][p["sha256"]])
                  for p in check["_pages"])


def prepare_focus_dataset(store, profile, check):
    """Freeze all currently approved unique article texts into plain shards."""
    revision = check["dataset_revision_sha256"]
    if check["status"] != "ready" or not re.fullmatch(r"[a-f0-9]{64}", revision):
        raise ValueError("A ready, revision-identified corpus is required")
    base = store.root / "datasets" / "focus_plain"
    base.mkdir(parents=True, exist_ok=True)
    directory = _inside(base, revision)
    if not directory.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=base))
        try:
            parts = {k: bytearray() for k in SPLIT_FILES}
            corpus = bytearray()
            documents = []
            for page in check["_pages"]:
                raw = page["data"]
                if _sha(raw) != page["sha256"]:
                    raise ValueError("Source bytes changed before dataset preparation")
                split = check["_partitions"][page["sha256"]]
                documents.append({"url": page["url"], "host": page["host"],
                    "clean_sha256": page["sha256"], "rights_status": page["rights"],
                    "operator_training_approved": page["operator_training_approved"],
                    "split": split, "offset": len(parts[split]), "length": len(raw),
                    "corpus_offset": len(corpus)})
                parts[split].extend(raw + b"\n\n")
                corpus.extend(raw + b"\n\n")
            payloads = {SPLIT_FILES[k]: bytes(data) for k, data in parts.items()}
            payloads["corpus.txt"] = bytes(corpus)
            payloads["documents.jsonl"] = b"".join((json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8") for d in documents)
            manifest = {"format": DATASET_FORMAT, "profile": profile,
                "dataset_revision_sha256": revision, "created_at_utc": _now(),
                "split_method": check["split_method"], "model_config": check["model_config"],
                "independent_publisher_evaluation": check["independent_publisher_evaluation"],
                "source_spec_sha256": _sha(_json(_source_spec(check))),
                "documents_by_split": check["documents_by_split"],
                "files": {name: {"sha256": _sha(data), "bytes": len(data)} for name, data in payloads.items()},
                "document_separator": "two UTF-8 newline bytes",
                "corpus_txt_contains": "all approved unique documents; optimizer reads train.txt only"}
            for name, data in payloads.items():
                (temporary / name).write_bytes(data)
            (temporary / "manifest.json").write_bytes(_json(manifest))
            try:
                temporary.rename(directory)
            except OSError:
                if not directory.exists():
                    raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    descriptor = {"directory": str(directory), "manifest": str(directory / "manifest.json"),
                  "dataset_revision_sha256": revision}
    # Validate existing bundles too; never silently regenerate tampered data.
    read_focus_dataset(descriptor, check)
    store.put("focus_plain_dataset_latest_" + profile, descriptor)
    from .dataset_catalog import register_dataset
    register_dataset(store, descriptor)
    return descriptor


def read_focus_dataset(descriptor, check):
    directory = Path(descriptor["directory"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != DATASET_FORMAT or
            manifest.get("dataset_revision_sha256") != check["dataset_revision_sha256"] or
            manifest.get("source_spec_sha256") != _sha(_json(_source_spec(check))) or
            manifest.get("model_config") != check["model_config"] or
            manifest.get("split_method") != check["split_method"]):
        raise ValueError("Plain dataset manifest does not match the approved corpus")
    if set(manifest["files"]) != {*SPLIT_FILES.values(), "corpus.txt", "documents.jsonl"}:
        raise ValueError("Plain dataset file set is incomplete")
    payloads = {name: _verified_bytes(directory, name, meta["sha256"])
                for name, meta in manifest["files"].items()}
    if any(len(payloads[name]) != meta["bytes"] for name, meta in manifest["files"].items()):
        raise ValueError("Plain dataset file size mismatch")
    records = [json.loads(line) for line in payloads["documents.jsonl"].decode("utf-8").splitlines()]
    splits = {k: [] for k in SPLIT_FILES}
    seen = set()
    actual_spec = []
    for record in records:
        digest, split = record["clean_sha256"], record["split"]
        if digest in seen or split not in splits:
            raise ValueError("Duplicate article or invalid dataset partition")
        data = payloads[SPLIT_FILES[split]][record["offset"]:record["offset"] + record["length"]]
        corpus_data = payloads["corpus.txt"][record["corpus_offset"]:record["corpus_offset"] + record["length"]]
        if _sha(data) != digest or _sha(corpus_data) != digest:
            raise ValueError("Plain dataset document SHA-256 mismatch")
        seen.add(digest)
        actual_spec.append((record["url"], digest, record["rights_status"], split))
        splits[split].append(data)
    if sorted(actual_spec) != _source_spec(check):
        raise ValueError("Plain dataset document list differs from approved source list")
    return splits


def export_saved_focus_corpus(store, profile="saudi-history-ar"):
    """Re-extract saved HTML locally; explicitly label historical reconstruction."""
    from .focus_training import readiness
    from .web_discovery import extract
    from .focus_data import get_profile, topic_relevance
    config = get_profile(profile)
    run_id = "snapshot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    begin_extraction_run(store, run_id, profile, "reconstructed_from_archived_html")
    rows = store.db.execute(
        "SELECT s.url,s.raw_path,s.raw_html_sha256 FROM web_corpus_sources s "
        "JOIN focus_archived_sources f ON f.url=s.url WHERE f.profile=? ORDER BY s.url", (profile,)).fetchall()
    for sequence, row in enumerate(rows, 1):
        path = Path(row["raw_path"]).resolve()
        if not path.is_relative_to((store.root / "web_corpus" / "raw").resolve()):
            raise ValueError("Archived HTML path escapes corpus")
        with gzip.open(path, "rb") as stream:
            html_bytes = stream.read(2_000_001)
        if len(html_bytes) > 2_000_000 or _sha(html_bytes) != row["raw_html_sha256"]:
            raise ValueError("Archived HTML snapshot failed verification")
        html = html_bytes.decode("utf-8")
        text, title, _, noai, raw_text = extract(html, include_raw=True)
        quality = topic_relevance(text, title, config)
        capture_extraction(store, run_id, sequence, row["url"], title, html, raw_text,
            profile_match=quality["language_ok"] and quality["topical"] and len(text) >= 400, noai=noai)
    check = readiness(store, profile)
    dataset = prepare_focus_dataset(store, profile, check) if check["status"] == "ready" else None
    result = finalize_extraction_run(store, run_id, dataset)
    if result["extracted_pages"] and result["files"]["raw.txt"]["bytes"]:
        from .dataset_catalog import _link
        _link(Path(result["directory"]), store.root / "corpus_runs" / "COLLECTED_TEXT")
    store.put("focus_plain_export_latest_" + profile, result)
    return result
