"""Capacity upgrades preserve old predictions and train the added parameters."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from evolve.focus_training import (
    _network, _expand_accepted_weights, LEGACY_MODEL_CONFIG, parameter_count,
    set_model_capacity, set_article_holdout_policy, readiness, train_if_eligible,
)
from evolve.focus_control import model_size_snapshot, acquire_lock, release_lock
from evolve.open_corpus import archive_page
from evolve.store import Store
from evolve.training_approval import approve_current_corpus
from test_v127_focus_overnight import arabic_text, fake_page

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Optional PyTorch integration")
class FocusModelCapacityTests(unittest.TestCase):
    def test_exact_counts_and_prediction_preserving_widening(self):
        torch.manual_seed(42)
        _, old = _network()
        config = {"embedding_dim": 64, "hidden_size": 128}
        _, larger = _network(config)
        _expand_accepted_weights(torch, larger, old.state_dict(), LEGACY_MODEL_CONFIG, config)
        x = torch.randint(0, 256, (2, 24))
        with torch.no_grad():
            self.assertTrue(torch.allclose(old(x), larger(x), atol=2e-6, rtol=1e-5))
        self.assertEqual(parameter_count(LEGACY_MODEL_CONFIG), 79168)
        self.assertEqual(parameter_count(config), sum(p.numel() for p in larger.parameters()))
        large_config = {"embedding_dim": 256, "hidden_size": 768}
        _, full = _network(large_config)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 2626304)
        self.assertTrue(all(p.requires_grad for p in full.parameters()))
        loss = torch.nn.functional.cross_entropy(larger(x).reshape(-1, 256), x.reshape(-1))
        loss.backward()
        self.assertGreater(float(larger.head.weight.grad[:, 96:].abs().sum()), 0)

    def test_upgrade_loads_legacy_checkpoint_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                approve_current_corpus(store, "Operator approval", include_future=True)
                set_article_holdout_policy(store, "saudi-history-ar", True)
                for i in range(5):
                    text = arabic_text(size=70) + " سجل " + str(i)
                    archive_page(store, f"https://saudipedia.com/history-{i}", fake_page(text), text,
                                 title="تاريخ المملكة العربية السعودية")
                first = train_if_eligible(store, "saudi-history-ar", steps=3)
                self.assertTrue(first["accepted"])
                # Represent the real pre-upgrade checkpoint schema, which did
                # not yet record architecture dimensions.
                state = torch.load(first["checkpoint"], map_location="cpu", weights_only=True)
                state.pop("model_config")
                legacy = Path(first["checkpoint"]).parent / "legacy.pt"
                torch.save(state, legacy)
                parent = dict(first, checkpoint=str(legacy), sha256=hashlib.sha256(legacy.read_bytes()).hexdigest())
                parent.pop("model_config")
                store.put("focus_train_latest_saudi-history-ar", parent)
                set_model_capacity(store, "saudi-history-ar", 64, 128)
                before = model_size_snapshot(store)["saudi_history_arabic"]
                self.assertFalse(before["accepted_checkpoint_matches_architecture"])
                self.assertEqual(before["accepted_checkpoint_parameters"], 79168)
                self.assertTrue(readiness(store, "saudi-history-ar")["new_training_data_since_last_attempt"])
                upgraded = train_if_eligible(store, "saudi-history-ar", steps=3)
                self.assertTrue(upgraded["accepted"])
                self.assertTrue(upgraded["expanded_accepted_weights"])
                self.assertTrue(upgraded["resumed_accepted_weights"])
                self.assertEqual(upgraded["steps_total"], 6)
                self.assertEqual(upgraded["parent_checkpoint_sha256"], parent["sha256"])
                self.assertEqual(upgraded["new_parameters_added"], parameter_count(upgraded["model_config"]) - 79168)
                self.assertEqual(hashlib.sha256(legacy.read_bytes()).hexdigest(), parent["sha256"])
                self.assertTrue(model_size_snapshot(store)["saudi_history_arabic"]["accepted_checkpoint_matches_architecture"])
                self.assertEqual(train_if_eligible(store, "saudi-history-ar", steps=3)["status"], "skipped_unchanged_approved_data")
                self.assertEqual(len(store.get("focus_train_accepted_history_saudi-history-ar")), 3)
            finally:
                store.close()

    def test_capacity_update_requires_stopped_worker_and_valid_increases(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                lock = acquire_lock(store)
                try:
                    with self.assertRaisesRegex(ValueError, "Stop the focus worker"):
                        set_model_capacity(store, "saudi-history-ar", 64, 128)
                finally:
                    release_lock(lock)
                set_model_capacity(store, "saudi-history-ar", 64, 128)
                for embed, hidden in ((48, 96), (True, 128), (64, 0), (64, 2048)):
                    with self.subTest(embed=embed, hidden=hidden), self.assertRaises(ValueError):
                        set_model_capacity(store, "saudi-history-ar", embed, hidden)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
