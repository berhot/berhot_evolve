import tempfile
import unittest
from evolve.store import Store
from evolve.connectors import SourceCandidate, can_ingest
from evolve.curriculum import expand_math_curriculum
from evolve.settings import load_config


class StorageTests(unittest.TestCase):
    def test_dedup_provenance_splits_and_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(temp)
            for i in range(10):
                saved = store.save_doc(f"Q{i}", f"Object {i} and its long description", f"https://wikidata.org/wiki/Q{i}", "CC0-1.0")
                self.assertTrue(saved)
            self.assertFalse(store.save_doc("Q0", "Object 0 and its long description", "example", "CC0-1.0"))
            self.assertEqual(len(store.documents("val")), 2)
            self.assertEqual(len(store.documents("train")), 8)
            settings = load_config()
            status = expand_math_curriculum(store, settings)
            self.assertEqual(status["status"], "generated")
            self.assertGreater(status["new_math_lessons"], 0)
            self.assertTrue(all(d["license"] == "SELF-GENERATED" for d in store.documents() if d["source_id"].startswith("synth:math:")))
            store.experience("math", "1 + 2", "3", "3", True, "python")
            self.assertEqual(store.status()["verified_experiences"], 1)
            store.close()

    def test_social_candidate_requires_rights(self):
        candidate = SourceCandidate("https://social.example/post", "text", "UNKNOWN", "", False, contains_personal_data=True)
        allowed, reason = can_ingest(candidate)
        self.assertFalse(allowed)
        self.assertTrue(reason)
        safe = SourceCandidate("https://example.org/item", "3d", "CC0-1.0", "https://example.org/license", True)
        self.assertTrue(can_ingest(safe)[0])
