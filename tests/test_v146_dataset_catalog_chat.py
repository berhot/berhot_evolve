"""Dataset training evidence and actual accepted-checkpoint inference."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from evolve.dataset_catalog import (register_dataset, begin_dataset_training,
    finish_dataset_training, sync_saved_training_records, dataset_inventory)
from evolve.focus_inference import FocusGenerator, brain_status
from evolve.focus_training import FORMAT, _network
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_studio import LearningController, _handler, ThreadingHTTPServer

try:
    import torch
except ImportError:
    torch = None

PROFILE = "saudi-history-ar"


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.addCleanup(self.store.close)
        self.directory = self.store.root / "datasets" / "focus_plain" / ("a" * 64)
        self.directory.mkdir(parents=True)
        self.descriptor = {"directory": str(self.directory), "dataset_revision_sha256": "a" * 64}
        self.original = "نص عربي ثابت".encode()
        (self.directory / "train.txt").write_bytes(self.original)
        (self.directory / "manifest.json").write_text(json.dumps({
            "profile": PROFILE, "documents_by_split": {"train": 1, "val": 1, "test": 1},
            "files": {"train.txt": {"bytes": len(self.original)}}}))

    def test_states_create_links_preserve_bytes_and_separate_acceptance(self):
        register_dataset(self.store, self.descriptor)
        base = self.directory.parent
        self.assertEqual((base / "ready" / self.directory.name).resolve(), self.directory.resolve())
        receipt = begin_dataset_training(self.store, self.descriptor, PROFILE, 2)
        self.assertFalse((base / "ready" / self.directory.name).exists())
        self.assertTrue((base / "training" / self.directory.name).is_dir())
        finish_dataset_training(self.store, self.descriptor, receipt, record={"steps": 2, "accepted": False})
        inventory = dataset_inventory(self.store)
        self.assertEqual(inventory["counts"]["trained/rejected"], 1)
        self.assertEqual(inventory["datasets"][0]["training_attempts"][0]["steps"], 2)
        self.assertEqual((self.directory / "train.txt").read_bytes(), self.original)
        register_dataset(self.store, self.descriptor)
        self.assertFalse((base / "ready" / self.directory.name).exists())
        receipt2 = begin_dataset_training(self.store, self.descriptor, PROFILE, 2)
        finish_dataset_training(self.store, self.descriptor, receipt2, record={"steps": 2, "accepted": True})
        self.assertTrue((base / "trained" / "accepted" / self.directory.name).is_dir())
        (base / "trained" / "accepted" / self.directory.name).unlink()
        register_dataset(self.store, self.descriptor)
        self.assertTrue((base / "trained" / "accepted" / self.directory.name).is_dir())

    def test_failure_has_receipt_without_claiming_accepted_checkpoint(self):
        receipt = begin_dataset_training(self.store, self.descriptor, PROFILE, 2)
        finish_dataset_training(self.store, self.descriptor, receipt, error=RuntimeError("interrupted"))
        content = json.loads(Path(receipt).read_text())
        self.assertEqual(content["status"], "failed")
        self.assertFalse(content["checkpoint_accepted"])

    def test_backfill_uses_saved_results_once_and_links_active_dataset(self):
        record = {"dataset_revision_sha256": "a" * 64, "sha256": "b" * 64,
                  "accepted": True, "steps": 3, "created_at_utc": "2026-11-06T00:00:00+00:00"}
        self.store.put("focus_train_latest_" + PROFILE, record)
        first = sync_saved_training_records(self.store)
        self.assertEqual(first["receipts_imported"], 1)
        self.assertEqual(sync_saved_training_records(self.store)["receipts_imported"], 0)
        self.assertEqual((self.directory.parent / "active" / PROFILE).resolve(), self.directory.resolve())
        self.assertTrue(first["datasets"][0]["active_model_dataset"])

    def test_candidate_evidence_without_acceptance_is_explicitly_unclassified(self):
        models = self.store.root / "models" / "focus" / PROFILE
        models.mkdir(parents=True)
        (models / "manifest-old.json").write_text(json.dumps({"dataset_revision_sha256": "a" * 64,
                                                            "training_steps": 5}))
        (models / "candidate-old.pt").write_bytes(b"historical candidate existence fixture")
        result = sync_saved_training_records(self.store)
        self.assertEqual(result["counts"]["trained/unclassified"], 1)
        self.assertIsNone(result["datasets"][0]["training_attempts"][0]["accepted"])


@unittest.skipIf(torch is None, "CPU PyTorch required")
class FocusInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.addCleanup(self.store.close)
        self.config = {"embedding_dim": 16, "hidden_size": 16}

    def checkpoint(self, char="X", key="focus_train_latest_"):
        _, model = _network(self.config)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[ord(char)] = 100
        folder = self.store.root / "models" / "focus" / PROFILE
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (char + ".pt")
        torch.save({"format": FORMAT, "model_config": self.config, "weights": model.state_dict()}, path)
        record = {"format": FORMAT, "model_config": self.config, "accepted": key == "focus_train_latest_",
                  "checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "steps_total": 7}
        self.store.put(key + PROFILE, record)
        return record

    def test_active_checkpoint_generates_actual_weights_not_last_rejected(self):
        active = self.checkpoint("X")
        self.checkpoint("Y", "focus_train_attempted_")
        changes = self.store.db.total_changes
        result = FocusGenerator().generate(self.store, "تاريخ السعودية", max_new_bytes=16, temperature=0)
        self.assertEqual(result["answer"], "X" * 16)
        self.assertEqual(result["checkpoint_sha256"], active["sha256"])
        self.assertFalse(result["training_updated"])
        self.assertEqual(self.store.db.total_changes, changes)
        status = brain_status(self.store)
        self.assertEqual(status['model']['training_context_bytes'], 48)
        self.assertFalse(status['capabilities']['history_qa_evaluated'])
        self.assertFalse(status['capabilities']['historical_answers_verified'])

    def test_no_checkpoint_never_uses_random_or_rejected_weights(self):
        self.checkpoint("Y", "focus_train_attempted_")
        result = FocusGenerator().generate(self.store, "التاريخ")
        self.assertEqual(result["status"], "no_accepted_checkpoint")
        self.assertEqual(result["answer"], "")
        self.assertEqual(brain_status(self.store)["status"], "no_accepted_checkpoint")

    def test_corrupted_checkpoint_is_rejected_even_after_cache_load(self):
        record = self.checkpoint()
        generator = FocusGenerator()
        generator.generate(self.store, "نص", max_new_bytes=2)
        with Path(record["checkpoint"]).open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            generator.generate(self.store, "نص", max_new_bytes=2)

    def test_sampling_is_seeded_valid_utf8_and_input_is_bounded(self):
        self.checkpoint()
        generator = FocusGenerator()
        a = generator.generate(self.store, "الدرعية", max_new_bytes=31, seed=72)
        b = generator.generate(self.store, "الدرعية", max_new_bytes=31, seed=72)
        self.assertEqual(a["answer"], b["answer"])
        self.assertEqual(len(a["answer"].encode()), 31)
        self.assertNotIn("\ufffd", a["answer"])
        for kwargs in ({"temperature": float("nan")}, {"max_new_bytes": 1025},
                       {"seed": -1}, {"top_k": True}):
            with self.assertRaises(ValueError):
                generator.generate(self.store, "text", **kwargs)

    def test_chat_route_api_and_csrf_protection(self):
        self.checkpoint()
        cfg = copy.deepcopy(load_config())
        cfg["storage_root"] = self.tmp.name
        controller = LearningController(cfg, interval=3600)
        self.addCleanup(controller.shutdown)
        server = ThreadingHTTPServer(("127.0.0.1", 0), http.client.HTTPConnection)
        port = server.server_address[1]
        origin = "http://127.0.0.1:" + str(port)
        server.RequestHandlerClass = _handler(controller, "local-test-token", origin)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.addCleanup(client.close)
        client.request("GET", "/chat")
        response = client.getresponse()
        page = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("local-test-token", page)
        self.assertNotIn("__EVOLVE_TOKEN__", page)
        for endpoint in ("/api/brain", "/api/datasets"):
            client.request("GET", endpoint)
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIsInstance(json.loads(response.read()), dict)
        body = json.dumps({"mode": "focus", "question": "الدرعية", "max_new_bytes": 8, "temperature": 0})
        client.request("POST", "/api/chat", body, headers={"Content-Type": "application/json", "Origin": origin})
        response = client.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        client.request("POST", "/api/chat", body, headers={"Content-Type": "application/json", "Origin": origin,
                                                          "X-Evolve-Key": "local-test-token"})
        response = client.getresponse()
        result = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(result["answer"], "X" * 8)


if __name__ == "__main__":
    unittest.main()
