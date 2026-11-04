import tempfile
import unittest
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class TrainingTests(unittest.TestCase):
    def test_random_weights_train_and_checkpoint(self):
        from evolve.training import train_candidate
        from evolve.brain import load_checkpoint, sample_text
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            store = Store(temp)
            config = load_config()
            config["storage_root"] = temp
            config["minimum_training_documents"] = 6
            config["minimum_new_documents_for_training"] = 6
            config["model"] = dict(block_size=16, embedding_dim=16, layers=1, heads=2, dropout=0.0)
            config["train"].update(steps=2, batch_size=2, eval_batches=1, device="cpu", max_seconds=60)
            for i in range(7):
                store.save_doc(f"Q{i}", f"English phrase number {i}. Repeated patterns teach byte prediction.",
                               f"https://www.wikidata.org/wiki/Q{i}", "CC0-1.0")
            result = train_candidate(store, config)
            self.assertEqual(result["status"], "trained")
            self.assertTrue(result["promoted"])
            model, meta = load_checkpoint(result["candidate_path"])
            generated = sample_text(model, "Hello", block_size=16, max_new_bytes=3)
            self.assertTrue(generated.startswith("Hello"))
            self.assertEqual(meta["format_version"], 1)
            self.assertEqual(train_candidate(store, config)["status"], "skipped")
            store.close()
