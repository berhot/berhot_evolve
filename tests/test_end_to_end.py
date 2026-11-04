import tempfile
import unittest
from evolve.settings import load_config
from evolve.runtime import once
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class EndToEndTests(unittest.TestCase):
    def test_offline_mocked_full_autonomous_cycle(self):
        torch.set_num_threads(2)
        def fetcher(qid, ua, cap):
            n = int(qid[1:])
            return {"entities":{qid:{
                "type":"item", "labels":{"en":{"value":f"Concept {n}"},"ar":{"value":f"مفهوم {n}"}},
                "descriptions":{"en":{"value":f"A concept about learning, knowledge, numbers and sciences, entry {n}."}},
                "claims":{"P279":[{"rank":"normal", "mainsnak":{"datavalue":{"value":{"entity-type":"item","id":f"Q{n+1}"}}}}]}
            }}}
        with tempfile.TemporaryDirectory() as temp:
            settings = load_config()
            settings.update(storage_root=temp, seed_entities=["Q50000"], discovery_per_cycle=12,
                            max_linked_entities_per_source=1, crawl_delay_seconds=0.0, daily_request_budget=50,
                            minimum_training_documents=6, minimum_new_documents_for_training=6,
                            max_experiments_per_cycle=2)
            settings["model"] = dict(block_size=16, embedding_dim=16, layers=1, heads=2, dropout=0.0)
            settings["train"].update(device="cpu", steps=2, batch_size=2, eval_batches=1, max_seconds=60)
            report = once(settings, fetcher=fetcher, sleep_fn=lambda s:None)
            self.assertEqual(report["discovery"]["saved"], 12)
            self.assertEqual(report["curriculum"]["status"], "generated")
            self.assertEqual(report["training"]["status"], "trained")
            self.assertTrue(report["training"]["promoted"])
            self.assertEqual(report["lab"]["status"], "completed")
            self.assertGreater(report["status"]["documents"], 12)
            store = Store(temp)
            self.assertGreater(len(store.relevant_memories("Concept 50001")), 0)
            self.assertEqual(store.count("experiences"), 2)
            store.close()
