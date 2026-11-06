"""Real raw/clean text files, document dedup and verified training shards."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evolve.store import Store
from evolve.open_corpus import archive_page, normalize_web_text
from evolve.web_discovery import extract
from evolve.training_approval import approve_current_corpus
from evolve.focus_training import readiness, set_article_holdout_policy, train_if_eligible
from evolve.focus_overnight import init_state, cycle
from evolve.settings import load_config
from evolve.plain_corpus import (
    begin_extraction_run, capture_extraction, finalize_extraction_run,
    prepare_focus_dataset, read_focus_dataset, export_saved_focus_corpus,
)
from test_v127_focus_overnight import arabic_text, fake_page

try:
    import torch
except ImportError:
    torch = None


class PlainCorpusTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(self.folder.name)
        self.addCleanup(self.store.close)
        approve_current_corpus(self.store, "Approve collected snapshots", include_future=True)
        set_article_holdout_policy(self.store, "saudi-history-ar", True)

    def source(self, n, noai=False, text=None):
        url = f"https://saudipedia.com/history-{n}"
        html = fake_page(text or (arabic_text(size=65) + " سجل " + str(n)))
        content, title, _, _, raw = extract(html, include_raw=True)
        result = archive_page(self.store, url, html, content, title=title, noai=noai)
        self.store.record_focus_archive("saudi-history-ar", url, result["clean_sha256"])
        return url, html, raw, result

    def test_raw_clean_dedup_and_noai_exclusion_with_utf8_offsets(self):
        directory = begin_extraction_run(self.store, "test-batch", "saudi-history-ar")
        base = "Cafe\u0301 \u200b نص عربي \n\t " + arabic_text(size=65)
        for n, (text, noai) in enumerate(((base, False), (normalize_web_text(base), False), (base + " مختلف ", True)), 1):
            url, html, raw, _ = self.source(n, noai=noai, text=text)
            capture_extraction(self.store, "test-batch", n, url, "title", html, raw, noai=noai)
        self.assertTrue((directory / "raw.txt").stat().st_size > 0)
        result = finalize_extraction_run(self.store, "test-batch")
        self.assertEqual(result["extracted_pages"], 3)
        self.assertEqual(result["unique_clean_texts"], 2)
        self.assertEqual(result["duplicate_clean_texts"], 1)
        self.assertEqual(result["training_eligible_unique_texts"], 1)
        self.assertIn("\u200b", (directory / "raw.txt").read_text())
        self.assertNotIn("\u200b", (directory / "clean.txt").read_text())
        records = [json.loads(line) for line in (directory / "documents.jsonl").read_text().splitlines()]
        combined = (directory / "raw.txt").read_bytes()
        for r in records:
            data = combined[r["raw_offset"]:r["raw_offset"] + r["raw_length"]]
            self.assertEqual(hashlib.sha256(data).hexdigest(), r["raw_text_sha256"])
            self.assertTrue(gzip.decompress((directory / r["raw_html_file"]).read_bytes()).startswith(b"<html>"))
        self.assertEqual(finalize_extraction_run(self.store, "test-batch"), result)
        with self.assertRaises(ValueError):
            capture_extraction(self.store, "test-batch", 4, "https://example.org/a", "", "", "x")
        with self.assertRaises(ValueError):
            begin_extraction_run(self.store, "../../outside", "saudi-history-ar")

    def test_empty_run_has_raw_file_and_manifest(self):
        directory = begin_extraction_run(self.store, "empty-batch", "saudi-history-ar")
        result = finalize_extraction_run(self.store, "empty-batch")
        self.assertEqual(result["extracted_pages"], 0)
        self.assertEqual((directory / "raw.txt").read_bytes(), b"")
        self.assertTrue((directory / "manifest.json").is_file())

    def test_snapshot_and_shards_preserve_whole_documents_and_detect_tampering(self):
        for n in range(5):
            self.source(n)
        duplicate = arabic_text(size=65) + " سجل 0"
        self.source(6, text=duplicate)
        self.source(7, noai=True)
        result = export_saved_focus_corpus(self.store)
        self.assertEqual(result["source_mode"], "reconstructed_from_archived_html")
        self.assertEqual(result["extracted_pages"], 7)
        self.assertEqual(result["duplicate_clean_texts"], 1)
        check = readiness(self.store, "saudi-history-ar")
        self.assertEqual(check["rights_eligible_arabic_pages"], 5)
        descriptor = result["training_dataset"]
        splits = read_focus_dataset(descriptor, check)
        groups = [{hashlib.sha256(raw).hexdigest() for raw in values} for values in splits.values()]
        self.assertEqual(sum(len(g) for g in groups), 5)
        self.assertFalse(any(groups[i] & groups[j] for i in range(3) for j in range(i)))
        self.assertEqual(prepare_focus_dataset(self.store, "saudi-history-ar", check), descriptor)
        shard = Path(descriptor["directory"]) / "train.txt"
        original = shard.read_bytes()
        shard.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            prepare_focus_dataset(self.store, "saudi-history-ar", check)

    def test_focus_cycle_exports_raw_even_for_off_topic_reads(self):
        settings = load_config()
        settings.update(storage_root=self.folder.name, daily_request_budget=100, crawl_delay_seconds=0)
        state = init_state(self.store, "saudi-history-ar", max_pages=3)
        urls = [f"https://saudipedia.com/history-{n}" for n in range(3)]
        state["_queue"] = [{"url": url, "depth": 0} for url in urls]
        shared = fake_page(arabic_text(size=65))
        pages = {urls[0]: shared, urls[1]: shared,
                 urls[2]: "<html><title>English</title><p>" + "A distant science topic. " * 40 + "</p></html>"}
        result = cycle(self.store, settings, state, pages=3, train_steps=1,
                       fetcher=pages.__getitem__, sleep_fn=lambda _: None)
        manifest = json.loads((Path(result["extraction_bundle"]["directory"]) / "manifest.json").read_text())
        self.assertEqual(manifest["extracted_pages"], 3)
        self.assertEqual(manifest["unique_clean_texts"], 2)
        self.assertEqual(manifest["training_eligible_unique_texts"], 1)
        self.assertEqual(state["pages_archived"], 2)

    @unittest.skipUnless(torch is not None, "Optional PyTorch integration")
    def test_real_trainer_uses_prepared_plain_files_and_rejects_tampered_shard(self):
        for n in range(5):
            self.source(n)
        result = train_if_eligible(self.store, "saudi-history-ar", steps=2)
        self.assertEqual(result["status"], "trained")
        directory = Path(result["plain_dataset_directory"])
        self.assertTrue((directory / "train.txt").is_file())
        self.assertTrue(Path(result["plain_dataset_manifest"]).is_file())
        status = json.loads((directory / "training_status.json").read_text())
        receipt = json.loads(Path(status["last_completed_attempt"]).read_text())
        self.assertEqual(status["status"], "trained/accepted" if result["accepted"] else "trained/rejected")
        self.assertEqual(receipt["result"]["sha256"], result["sha256"])
        self.assertEqual(receipt["result"]["steps"], 2)
        (directory / "train.txt").write_text("corrupted", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            train_if_eligible(self.store, "saudi-history-ar", steps=2)


if __name__ == "__main__":
    unittest.main()
