"""Unseen arithmetic evaluation with inspectable raw model completions.

An exact numeric answer is the skill metric; falling token loss alone is not.
The test excludes all training AND held-out problems from new probes.
"""
import random
import re

from .brain import get_device, load_checkpoint, sample_text
from .practice import checked_answer


def parse_numeric_prefix(completion):
    """Treat only a leading number as the answer; never search arbitrary text."""
    found = re.match(r"\s*(-?\d+)(?=$|\s|[,.;!?])", completion)
    return int(found.group(1)) if found else None


def _seen_math_questions(store):
    questions = set()
    rows = store.db.execute(
        "SELECT text FROM documents WHERE topic='verified:arithmetic'"
    ).fetchall()
    for row in rows:
        found = re.match(r"Question:\s*(\d{1,2}\s*[+*\-]\s*\d{1,2})\s*=\s*\?", row["text"])
        if found:
            questions.add(" ".join(found.group(1).split()))
    return questions


def _select_questions(store, n, seed, skill, distribution):
    rng = random.Random(int(seed))
    seen = _seen_math_questions(store)
    chosen = []
    chosen_strings = set()
    if skill not in ("mixed", "single-digit-add"):
        raise ValueError("Skill must be mixed or single-digit-add")
    if distribution not in ("in-range", "extrapolation"):
        raise ValueError("Distribution must be in-range or extrapolation")
    for _ in range(12000):
        if len(chosen) >= n:
            break
        if skill == "single-digit-add":
            lower, upper = (0, 10) if distribution == "in-range" else (10, 20)
            op = "+"
        else:
            lower, upper = (1, 80) if distribution == "in-range" else (80, 100)
            op = rng.choice(["+", "-", "*"])
        a, b = rng.randrange(lower, upper), rng.randrange(lower, upper)
        problem = f"{a} {op} {b}"
        if problem in seen or problem in chosen_strings:
            continue
        chosen_strings.add(problem)
        chosen.append((a, b, op, checked_answer(a, b, op)))
    return chosen


def evaluate_arithmetic(store, settings, questions=10, seed=90210,
                        skill="mixed", distribution="extrapolation", model_source="auto"):
    count = int(questions)
    if not 1 <= count <= 50:
        raise ValueError("Evaluation supports 1..50 questions per distribution")
    if distribution not in ("in-range", "extrapolation", "both"):
        raise ValueError("Distribution must be in-range, extrapolation, or both")
    if model_source not in ("auto", "general", "math"):
        raise ValueError("Model source must be auto, general, or math")
    math_ckpt = store.get("math_answerer_checkpoint_" + skill)
    general_ckpt = store.latest_promoted()
    if model_source == "math" or (model_source == "auto" and math_ckpt):
        active = math_ckpt
        track = "math_answerer"
    else:
        active = general_ckpt
        track = "general"
    if not active:
        return {"status": "skipped", "reason": f"No accepted {track} checkpoint"}
    model, meta = load_checkpoint(active["path"])
    if not meta.get("verified_only"):
        return {"status": "skipped", "reason": "Checkpoint was not trained on verified-only examples"}
    block = int(meta["model_config"]["block_size"])
    device = get_device(settings["train"].get("device", "auto"))
    distributions = ["in-range", "extrapolation"] if distribution == "both" else [distribution]
    trials = []
    summary = {}
    for index, group in enumerate(distributions):
        problems = _select_questions(store, count, seed + index * 19237, skill, group)
        if len(problems) < count:
            summary[group] = {"status": "insufficient_unseen_questions",
                              "available": len(problems), "requested": count}
            continue
        correct_count = 0
        numeric_count = 0
        examples = []
        for a, b, op, correct in problems:
            prompt = f"Question: {a} {op} {b} = ?\nAnswer:"
            generated = sample_text(
                model, prompt, block, max_new_bytes=12, device=device,
                seed=seed, temperature=0.2, greedy=True)
            suffix = generated[len(prompt):]
            predicted = parse_numeric_prefix(suffix)
            parsed = predicted is not None
            right = parsed and predicted == correct
            correct_count += int(right)
            numeric_count += int(parsed)
            item = {
                "question": f"{a} {op} {b}", "expected": correct,
                "model_answer": predicted, "numeric_output": parsed,
                "correct": right, "raw_completion": repr(suffix[:40]),
            }
            trials.append(item)
            if len(examples) < 10:
                examples.append(item)
        summary[group] = {
            "status": "evaluated", "questions": len(problems),
            "correct": correct_count, "numeric_outputs": numeric_count,
            "accuracy": round(correct_count / len(problems), 4),
            "numeric_output_rate": round(numeric_count / len(problems), 4),
            "examples": examples,
        }
    evaluated = [r for r in summary.values() if r.get("status") == "evaluated"]
    if not evaluated:
        return {"status": "skipped", "reason": "Not enough unseen problems",
                "distributions": summary}
    total = sum(row["questions"] for row in evaluated)
    correct = sum(row["correct"] for row in evaluated)
    numeric = sum(row["numeric_outputs"] for row in evaluated)
    report = {
        "status": "evaluated",
        "evaluation": "unseen_arithmetic_exact_match_v2",
        "checkpoint": active["sha256"], "model_track": track,
        "model_objective": meta.get("objective", "next_byte_prediction"),
        "training_skill": meta.get("skill", "mixed/general"),
        "tested_skill": skill, "tested_distribution": distribution,
        "questions": total, "correct": correct,
        "numeric_outputs": numeric,
        "numeric_output_rate": round(numeric / total, 4),
        "accuracy": round(correct / total, 4),
        "distributions": summary,
        "examples": trials[:10],
        "note": "Unseen exact-match numeric accuracy, not generic token-loss improvement. 'raw_completion' exposes formatting errors.",
    }
    store.put("last_arithmetic_evaluation", {
        key: value for key, value in report.items() if key not in ("examples", "distributions")
    })
    store.log("arithmetic_model_evaluation", "completed", report)
    return report
