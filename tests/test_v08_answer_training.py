"""Regression tests for answer-focused learning and honest arithmetic assessment."""
import tempfile
import unittest
from pathlib import Path

from evolve.answer_training import _parse
from evolve.evaluation import parse_numeric_prefix, _select_questions
from evolve.practice import practice_arithmetic
from evolve.settings import load_config
from evolve.store import Store

try:
    import torch
except ImportError:
    torch = None


class FormatAndSplitTests(unittest.TestCase):
    def test_number_parser_rejects_non_numeric_garbage(self):
        self.assertIsNone(parse_numeric_prefix("Answer: [invalid]"))
        self.assertIsNone(parse_numeric_prefix("random 123"))
        self.assertEqual(parse_numeric_prefix("  -36\n"), -36)
        self.assertEqual(parse_numeric_prefix("123 \nrest"), 123)

    def test_answer_only_mask_never_trains_on_question_text(self):
        row = {"text": "Question: 8 + 2 = ?\nAnswer: 10\n"}
        x, y = _parse(row, 64)
        prompt = "Question: 8 + 2 = ?\nAnswer:"
        start = len(prompt) - 1
        self.assertEqual(len(x), 64)
        self.assertEqual(len(y), 64)
        self.assertTrue(all(v == -100 for v in y[:start]))
        self.assertEqual(y[start:start+4], [ord(" "), ord("1"), ord("0"), ord("\n")])
        self.assertTrue(all(v == -100 for v in y[start+4:]))
        self.assertIsNone(_parse({"text": "Question: 8 + 2 = ?\nAnswer: 100\n"}, 64))

    def test_unseen_questions_do_not_overlap_saved_train_or_heldout(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            result = practice_arithmetic(store, 32, skill="single-digit-add")
            self.assertEqual(result["new_examples"], 32)
            seen = {r["text"].split(" = ?")[0].replace("Question: ", "")
                    for r in store.documents()}
            questions = _select_questions(store, 8, 991, "single-digit-add", "in-range")
            self.assertEqual(len(questions), 8)
            self.assertTrue(all(f"{a} {op} {b}" not in seen for a, b, op, _ in questions))
            self.assertTrue(all(a < 10 and b < 10 for a, b, _, _ in questions))
            store.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class AnswererTests(unittest.TestCase):
    def test_separate_math_track_two_rounds_and_raw_accuracy(self):
        from evolve.answer_training import train_math_answerer
        from evolve.evaluation import evaluate_arithmetic
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            config = load_config()
            config.update(storage_root=root, minimum_new_documents_for_training=6)
            config["model"] = {
                "block_size": 64, "embedding_dim": 16, "heads": 2,
                "layers": 1, "dropout": 0.0,
            }
            config["train"].update(
                device="cpu", steps=8, batch_size=4,
                learning_rate=0.003, max_seconds=70, eval_batches=1,
            )
            result = practice_arithmetic(store, 32, skill="single-digit-add")
            self.assertEqual(result["new_examples"], 32)
            first = train_math_answerer(store, config, "single-digit-add", steps=8)
            self.assertEqual(first["status"], "trained")
            self.assertEqual(first["objective"], "arithmetic_answer_only_v1")
            self.assertTrue(first["accepted"])
            self.assertTrue(Path(first["candidate_path"]).exists())
            self.assertIsNone(store.latest_promoted())  # separate from general model
            self.assertTrue(store.get("math_answerer_checkpoint_single-digit-add"))
            ev = evaluate_arithmetic(
                store, config, questions=4, seed=403, skill="single-digit-add",
                distribution="both", model_source="math")
            self.assertEqual(ev["status"], "evaluated")
            self.assertEqual(ev["questions"], 8)
            self.assertIn("in-range", ev["distributions"])
            self.assertIn("extrapolation", ev["distributions"])
            self.assertEqual(len(ev["examples"]), 8)
            self.assertTrue(all("raw_completion" in item for item in ev["examples"]))
            self.assertGreaterEqual(ev["numeric_output_rate"], 0.0)
            self.assertLessEqual(ev["numeric_output_rate"], 1.0)
            self.assertTrue(0.0 <= ev["accuracy"] <= 1.0)
            self.assertEqual(store.status()["latest_promoted_model"], None)

            more = practice_arithmetic(store, 32, skill="single-digit-add")
            self.assertEqual(more["new_examples"], 32)
            second = train_math_answerer(store, config, "single-digit-add", steps=8)
            self.assertEqual(second["status"], "trained")
            self.assertIsNotNone(second["previous_checkpoint_val_loss"])
            self.assertGreater(second["train_examples"], first["train_examples"])
            skipped = train_math_answerer(store, config, "single-digit-add", steps=8)
            self.assertEqual(skipped["status"], "skipped")
            store.close()


if __name__ == "__main__":
    unittest.main()
