"""Unseen arithmetic probes: evaluate actual model answers, not only byte loss."""
import random
import re

from .brain import get_device, load_checkpoint, sample_text
from .practice import checked_answer


def evaluate_arithmetic(store, settings, questions=10, seed=90210):
    questions = int(questions)
    if not 1 <= questions <= 50:
        raise ValueError("Evaluation supports 1..50 challenges")
    active = store.latest_promoted()
    if active is None:
        return {"status": "skipped", "reason": "No promoted neural checkpoint"}
    model, meta = load_checkpoint(active["path"])
    if not meta.get("verified_only"):
        return {"status": "skipped", "reason": "Active checkpoint was not trained on verified-only examples"}
    rng = random.Random(int(seed))
    device = get_device(settings["train"].get("device", "auto"))
    block_size = int(meta["model_config"]["block_size"])
    successes = 0
    trials = []
    for i in range(questions):
        # Disjoint from the practice generator's operand range (1..79).
        a, b = rng.randrange(80, 100), rng.randrange(80, 100)
        op = rng.choice(["+", "-", "*"])
        correct = checked_answer(a, b, op)
        prompt = f"Question: {a} {op} {b} = ?\nAnswer:"
        result = sample_text(model, prompt, block_size, max_new_bytes=12,
                             temperature=0.2, seed=seed+i, device=device)
        generated = result[len(prompt):].strip()
        match = re.match(r"(-?\d+)", generated)
        guess = int(match.group(1)) if match else None
        passed = guess == correct
        successes += int(passed)
        if i < 10:
            trials.append({"question": f"{a} {op} {b}", "expected": correct,
                           "model_answer": guess, "correct": passed})
    report = {
        "status": "evaluated",
        "evaluation": "unseen_arithmetic_exact_match_v1",
        "checkpoint": active["sha256"],
        "questions": questions,
        "correct": successes,
        "accuracy": round(successes / questions, 4),
        "examples": trials,
        "note": "Unseen operand range; exact-match measures reasoning generalization, not just byte-level validation loss.",
    }
    store.put("last_arithmetic_evaluation", {k: v for k, v in report.items() if k != "examples"})
    store.log("arithmetic_model_evaluation", "completed", report)
    return report
