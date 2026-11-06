"""Explicit operator approval is snapshot-bound and does not rewrite licenses."""
import tempfile
import unittest

from evolve.focus_training import readiness, train_if_eligible
from evolve.open_corpus import archive_page
from evolve.store import Store
from evolve.training_approval import approve_current_corpus, training_condition
from evolve.v118_core import _approved_text
from test_v127_focus_overnight import arabic_text, fake_page

try:
    import torch
except ImportError:
    torch = None


class TrainingApprovalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(self.folder.name)
        self.addCleanup(self.store.close)

    def archive(self, host="saudipedia.com", suffix="", noai=False):
        url = "https://" + host + "/history" + suffix
        text = arabic_text(size=80) + host + suffix
        archive_page(self.store, url, fake_page(text), text,
                     title="تاريخ المملكة العربية السعودية", noai=noai)
        return url

    def test_bulk_approval_is_explicit_idempotent_and_keeps_license(self):
        url = self.archive()
        with self.assertRaises(ValueError):
            _approved_text(self.store, url)
        result = approve_current_corpus(self.store, "Operator approved current corpus")
        self.assertEqual(result["new_approvals"], 1)
        self.assertEqual(_approved_text(self.store, url)[0]["rights_status"], "UNKNOWN")
        self.assertEqual(approve_current_corpus(self.store, "Same approval")["new_approvals"], 0)
        self.assertEqual(readiness(self.store, "saudi-history-ar")["rights_eligible_arabic_pages"], 1)
        self.assertNotEqual(readiness(self.store, "saudi-history-ar")["status"], "ready")
        with self.assertRaises(ValueError):
            approve_current_corpus(self.store, "")

    def test_new_sources_and_changed_snapshots_need_new_approval(self):
        url = self.archive()
        approve_current_corpus(self.store, "Current corpus only")
        new_url = self.archive(suffix="-new")
        with self.assertRaises(ValueError):
            _approved_text(self.store, new_url)
        text = arabic_text(size=90) + " محتوى جديد "
        archive_page(self.store, url, fake_page(text), text, title="تاريخ المملكة")
        with self.assertRaises(ValueError):
            _approved_text(self.store, url)
        self.assertEqual(self.store.db.execute(
            "SELECT count(*) FROM web_corpus_sources WHERE " + training_condition()
        ).fetchone()[0], 0)

    def test_noai_is_not_overridden_and_tampering_aborts_batch(self):
        url = self.archive(noai=True)
        self.assertEqual(approve_current_corpus(self.store, "Operator approval")["publisher_noai_skipped"], 1)
        with self.assertRaises(ValueError):
            _approved_text(self.store, url)
        self.archive(suffix="-good")
        row = self.store.db.execute("SELECT clean_path FROM web_corpus_texts ORDER BY rowid DESC LIMIT 1").fetchone()
        from pathlib import Path
        Path(row["clean_path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            approve_current_corpus(self.store, "Operator approval")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM corpus_training_approvals").fetchone()[0], 0)

    @unittest.skipUnless(torch is not None, "Optional PyTorch integration")
    def test_approved_independent_arabic_sources_reach_real_optimizer(self):
        for host in ("saudipedia.com", "www.spa.gov.sa", "www.mofa.gov.sa"):
            self.archive(host)
        approve_current_corpus(self.store, "Operator approval")
        self.assertEqual(readiness(self.store, "saudi-history-ar")["status"], "ready")
        steps = []
        result = train_if_eligible(self.store, "saudi-history-ar", steps=2,
                                  progress=lambda step, *_: steps.append(step))
        self.assertEqual(result["status"], "trained")
        self.assertEqual(steps, [1, 2])

    def test_general_language_corpus_honors_approval_without_license_rewrite(self):
        from test_v113_web_lab import make_store
        from evolve.v117 import prepare
        from evolve.web_lab import _read_frozen
        with tempfile.TemporaryDirectory() as root:
            store = make_store(root, hosts=("one.example.org", "two.example.org", "three.example.org"))
            try:
                approve_current_corpus(store, "Operator approval")
                result = prepare(store)
                self.assertEqual(result["status"], "prepared")
                manifest, _ = _read_frozen(store, result["dataset_sha256"])
                self.assertTrue(all(a["operator_training_approved"] for a in manifest["eligible_articles"]))
                self.assertTrue(all(a["rights"] == "UNKNOWN" for a in manifest["eligible_articles"]))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
