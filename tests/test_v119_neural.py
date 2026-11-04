"""Actual CPU gradient integration for automatically created reading examples."""
import copy
import tempfile
import unittest
from pathlib import Path

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_core import qa_readiness, train_qa, qa_neural_response
from evolve.v119_curiosity import sentence_lessons

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch required")
class AutonomousReadingNeuralTests(unittest.TestCase):
    def test_reviewed_source_sentences_train_separate_weights(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            cfg["train"]["max_seconds"] = 80
            try:
                for i, subject in enumerate(("mountain", "ocean", "planet")):
                    parts = []
                    for n in range(120):
                        parts.append(
                            f"Document {n} explains how {subject} formations "
                            f"were observed by careful researchers through "
                            f"repeated experiments with instruments and "
                            f"measurements from multiple independent field "
                            f"studies. Reliable observation {n} is needed "
                            f"to evaluate complex scientific processes.")
                    text = " ".join(parts)
                    url = f"https://synthetic-{i}.example.org/articles/{subject}"
                    archived = archive_page(
                        store, url,
                        "<html><title>Science</title><body>"+text+"</body></html>",
                        text, title=subject,
                        declared_license="CC0-1.0",
                        rights_url="fixture:self-authored")
                    self.assertEqual(archived["status"], "archived")
                exercises = sentence_lessons(store, per_host=12)
                self.assertGreaterEqual(exercises["total_source_derived_lessons"], 30)
                self.assertEqual(qa_readiness(store)["status"], "ready")
                before = store.get("v117_latest")
                report = train_qa(store, cfg, steps=4)
                self.assertEqual(report["status"], "trained", report)
                self.assertEqual(report["actual_training_steps"], 4)
                self.assertGreaterEqual(
                    report["auto_source_continuation_examples"], 12)
                self.assertTrue(Path(report["checkpoint"]).is_file())
                self.assertTrue(Path(report["qa_manifest_path"]).is_file())
                self.assertEqual(before, store.get("v117_latest"))
                self.assertIn("test_answer_loss_after", report)
                if report["accepted_by_validation"]:
                    self.assertEqual(
                        qa_neural_response(store, "Continue: Document 1")["mode"],
                        "qa_neural_raw")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
