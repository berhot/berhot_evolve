"""v1.12: extract many public sites, exact dedup and honest model audit.

Synthetic HTML only; never fetches live sites. No user-storage mutation.
"""
import gzip
import tempfile
import unittest
from pathlib import Path

from evolve.open_corpus import (
    normalize_web_text, archive_page, corpus_status, corpus_seed,
)
from evolve.model_audit import audit_model_and_ownership
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import discover_web


class StdlibPublicCorpusTests(unittest.TestCase):
    def test_normalization_is_unicode_safe_and_never_fabricates_content(self):
        source = "Cafe\u0301 \u200b \n\n is  open \t today.  "
        self.assertEqual(normalize_web_text(source), "Café is open today.")
        self.assertEqual(normalize_web_text("a\tb"), "a b")
        with self.assertRaises(TypeError):
            normalize_web_text(None)

    def test_unlicensed_html_can_be_archived_cleaned_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            text = ("Energy is the capacity to perform useful work across "
                    "physical systems and measurable processes. " * 3)
            html = f"<html><body><p>{text}</p></body></html>"
            first = archive_page(store, "https://a.example.org/energy", html,
                                 text, "Energy", noai=False)
            second = archive_page(store, "https://b.example.net/energy", html,
                                  text + " ", "Energy copy", noai=True)
            self.assertTrue(first["new_unique_text"])
            self.assertFalse(second["new_unique_text"])
            self.assertEqual(first["clean_sha256"], second["clean_sha256"])
            self.assertEqual(first["rights_status"], "UNKNOWN")
            self.assertEqual(second["rights_status"], "UNKNOWN")
            self.assertFalse(first["neural_training_admitted"])
            report = corpus_status(store)
            self.assertEqual(report["urls_collected"], 2)
            self.assertEqual(report["unique_cleaned_texts"], 1)
            self.assertEqual(report["duplicate_url_content_references"], 1)
            self.assertEqual(report["noai_source_count"], 1)
            self.assertEqual(report["rights_metadata"], {"UNKNOWN": 2})
            self.assertEqual(store.count("documents"), 0)
            row = store.db.execute("SELECT clean_path FROM web_corpus_texts").fetchone()
            self.assertEqual(Path(row["clean_path"]).read_text(), text.strip())
            rows = store.db.execute(
                "SELECT raw_path FROM web_corpus_sources").fetchall()
            self.assertTrue(all(
                gzip.decompress(Path(row["raw_path"]).read_bytes()).decode() == html
                for row in rows))
            store.close()

    def test_seed_multiple_https_sites_without_automatic_training(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            result = corpus_seed(store, [
                "https://a.example.org/topic", "https://b.example.net/subject"])
            self.assertEqual(result["status"], "queued")
            self.assertEqual(result["new_queue_urls"], 2)
            self.assertTrue(result["cross_host"])
            self.assertEqual(store.get("web_source_profile")["mode"], "research_only")
            self.assertEqual(store.db.execute(
                "SELECT COUNT(*) FROM web_frontier").fetchone()[0], 2)
            self.assertEqual(store.count("documents"), 0)
            self.assertEqual(corpus_seed(store, [
                "https://a.example.org/topic"])["new_queue_urls"], 0)
            with self.assertRaises(ValueError):
                corpus_seed(store, ["http://not-https.example.org/"])
            store.close()

    def test_discovery_keeps_public_html_even_without_cc0_never_promotes(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            config = load_config()
            config["storage_root"] = folder
            config["daily_request_budget"] = 30
            config["crawl_delay_seconds"] = 0
            config["web"]["max_depth"] = 0
            corpus_seed(store, [
                "https://a.example.org/energy",
                "https://b.example.net/energy"])
            html = """
                <html><head><title>Energy systems</title>
                  <meta name="robots" content="noai">
                </head><body>
                  <h1>Energy systems</h1>
                  <nav>Skip this menu</nav>
                  <p>Energy systems include a wide range of physical processes
                  such as electrical generation, heat transfer and motion.</p>
                  <p>These processes can be studied with careful measurements
                  and repeatable scientific experiments every day.</p>
                </body></html>
            """
            report = discover_web(
                store, config, limit=2, fetcher=lambda url: html,
                sleep_fn=lambda _: None)
            self.assertEqual(report["corpus_archived"], 2)
            self.assertEqual(report["corpus_new_unique_texts"], 1)
            self.assertEqual(report["corpus_duplicates"], 1)
            self.assertEqual(report["research_observed"], 2)
            self.assertEqual(report["saved"], 0)
            self.assertEqual(store.count("documents"), 0)
            self.assertEqual(store.db.execute(
                "SELECT COUNT(*) FROM web_corpus_sources WHERE noai=1").fetchone()[0], 2)
            self.assertEqual(corpus_status(store)["urls_collected"], 2)
            store.close()

    def test_model_audit_separates_data_model_training_and_external_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.save_doc("verified:arithmetic:sample", "Question: 2 + 2 = ?\nAnswer: 4\n",
                           "internal://sample", "SELF-GENERATED-VERIFIED",
                           topic="verified:arithmetic")
            result = audit_model_and_ownership(store)
            self.assertEqual(result["datasets"]["verified_arithmetic_examples"], 1)
            self.assertEqual(result["datasets"]["verified_text_facts_with_training_eligibility"], 0)
            self.assertEqual(result["saved_models"]["general_training_candidate_runs"], 0)
            self.assertEqual(result["saved_models"]["general_promoted_runs"], 0)
            self.assertFalse(result["implementation"]["is_zero_external_dependencies"])
            self.assertFalse(result["implementation"]["is_fully_original_neural_framework"])
            self.assertEqual(result["implementation"]["collector_external_python_packages_required"], [])
            self.assertEqual(result["autonomy"]["daemon_running_now"],
                             "unknown_from_database")
            self.assertFalse(result["autonomy"]["neural_weights_update_when_pages_are_saved"])
            self.assertIn(result["source_code"]["license_file"], ("MIT", "other_or_custom"))
            store.close()

    def test_corpus_separation_preserves_existing_documents_and_models(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.save_doc("legacy-math", "Question: 6 + 1 = ?\nAnswer: 7\n",
                           "internal://example", "SELF-GENERATED-VERIFIED",
                           topic="verified:arithmetic")
            before = [dict(x) for x in store.documents()]
            archive_page(store, "https://a.example.org/info",
                         "<p>Educational background content</p>",
                         "Educational background content", "Info")
            self.assertEqual([dict(x) for x in store.documents()], before)
            self.assertEqual(store.status()["collected_web_urls"], 1)
            self.assertEqual(store.status()["unique_cleaned_web_texts"], 1)
            self.assertIsNone(store.latest_promoted())
            store.close()


if __name__ == "__main__":
    unittest.main()
