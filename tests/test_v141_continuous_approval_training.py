"""Standing operator approval and explicit single-publisher article training."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolve.store import Store
from evolve.open_corpus import archive_page
from evolve.training_approval import approve_current_corpus, future_approval_policy
from evolve.focus_training import readiness, train_if_eligible, set_article_holdout_policy, ARTICLE_SPLIT
from evolve.v118_core import _approved_text
from evolve.settings import load_config
from test_v127_focus_overnight import arabic_text, fake_page

try:
    import torch
except ImportError:
    torch = None


class ContinuousApprovalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(self.folder.name)
        self.addCleanup(self.store.close)

    def archive(self, number, noai=False, text=None):
        url = "https://saudipedia.com/history-" + str(number)
        text = text or (arabic_text(size=60) + " سجل " + str(number))
        result = archive_page(self.store, url, fake_page(text), text,
            title="تاريخ المملكة العربية السعودية", noai=noai)
        return url, result

    def test_standing_approval_survives_restart_updates_and_can_be_disabled(self):
        approve_current_corpus(self.store, "Approve future collections", include_future=True)
        other = Store(self.folder.name)
        try:
            self.assertTrue(future_approval_policy(other)["enabled"])
            url, result = self.archive(1)
            self.assertTrue(result["operator_training_approved"])
            self.assertEqual(_approved_text(other, url)[0]["rights_status"], "UNKNOWN")
            self.archive(1, text=arabic_text(size=70) + " محتوى جديد ")
            self.assertIn("محتوى جديد", _approved_text(other, url)[1])
            blocked, result = self.archive(2, noai=True)
            self.assertFalse(result["operator_training_approved"])
            with self.assertRaises(ValueError):
                _approved_text(other, blocked)
            approve_current_corpus(self.store, "Disable future approval", include_future=False)
            unapproved, _ = self.archive(3)
            with self.assertRaises(ValueError):
                _approved_text(other, unapproved)
            self.assertTrue(_approved_text(other, url)[1])
        finally:
            other.close()

    @unittest.skipUnless(torch is not None, "Optional PyTorch integration")
    def test_article_training_uses_disjoint_persistent_groups_and_real_weights(self):
        approve_current_corpus(self.store, "Approve future collections", include_future=True)
        for i in range(15):
            self.archive(i)
        self.assertNotEqual(readiness(self.store, "saudi-history-ar")["status"], "ready")
        set_article_holdout_policy(self.store, "saudi-history-ar", True)
        before = readiness(self.store, "saudi-history-ar")
        self.assertEqual(before["status"], "ready")
        self.assertEqual(before["split_method"], ARTICLE_SPLIT)
        self.assertFalse(before["independent_publisher_evaluation"])
        result = train_if_eligible(self.store, "saudi-history-ar", steps=3)
        self.assertEqual(result["status"], "trained")
        self.assertTrue(result["accepted"])
        self.assertEqual(hashlib.sha256(Path(result["checkpoint"]).read_bytes()).hexdigest(), result["sha256"])
        groups = [set(result["source_hashes_by_split"][k]) for k in ("train", "val", "test")]
        self.assertFalse(any(groups[i] & groups[j] for i in range(3) for j in range(i)))
        for i in range(15, 25):
            self.archive(i)
        after = readiness(self.store, "saudi-history-ar")
        self.assertTrue(after["new_training_data_since_last_attempt"])
        self.assertTrue(all(after["_partitions"][sha] == part for sha, part in before["_partitions"].items()))
        resumed = train_if_eligible(self.store, "saudi-history-ar", steps=2)
        self.assertTrue(resumed["resumed_accepted_weights"])
        self.assertEqual(resumed["steps_total"], 5)
        self.assertEqual(train_if_eligible(self.store, "saudi-history-ar", steps=2)["status"], "skipped_unchanged_approved_data")

    @unittest.skipUnless(torch is not None, "Optional PyTorch integration")
    def test_studio_saved_arabic_training_records_job_without_network(self):
        from evolve.v118_studio import LearningController
        approve_current_corpus(self.store, "Approve future collections", include_future=True)
        for i in range(5):
            self.archive(i)
        set_article_holdout_policy(self.store, "saudi-history-ar", True)
        settings = load_config()
        settings["storage_root"] = self.folder.name
        controller = LearningController(settings)
        try:
            with patch("evolve.v118_studio.discover_web") as network:
                result = controller.submit("focus-language", steps=2)
                self.assertEqual(result["status"], "started")
                controller.active.join(timeout=10)
                self.assertFalse(controller.active.is_alive())
                network.assert_not_called()
            job = self.store.get("v118_last_job")
            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(job["completed_steps"], 2)
            self.assertEqual(job["result_summary"]["split_method"], ARTICLE_SPLIT)
        finally:
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
