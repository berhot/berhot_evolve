"""Seeded, mathematically verified experiments; never count a wrong guess as learning."""
import random
import re

from .brain import get_device, load_checkpoint, sample_text


def generate_math_problems(count=10, seed=42):
    rng = random.Random(seed)
    for _ in range(count):
        a, b = rng.randint(0, 15), rng.randint(0, 15)
        operator = rng.choice(["+", "-", "*"])
        expected = {"+": a + b, "-": a - b, "*": a * b}[operator]
        yield f"Question: {a} {operator} {b} = ", str(expected)


def run_math_experiments(store, settings, count=None):
    active = store.latest_promoted()
    if not active:
        return {"status":"skipped", "reason":"Train and promote a model before running self-evaluation"}
    model, meta = load_checkpoint(active["path"])
    device = get_device(settings["train"]["device"])
    model.to(device)
    count = count or int(settings["max_experiments_per_cycle"])
    correct = 0
    examples = []
    for prompt, answer in generate_math_problems(count, seed=store.count("experiences") + 42):
        output = sample_text(model, prompt, meta["model_config"]["block_size"],
                             max_new_bytes=12, device=device)
        generated = output[len(prompt):]
        match = re.match(r"\s*(-?\d+)", generated)
        prediction = match.group(1) if match else ""
        success = prediction == answer
        correct += int(success)
        store.experience("arithmetic", prompt, prediction, answer, success, "python_integer_arithmetic", active["sha256"])
        if len(examples) < 4:
            examples.append({"question":prompt.strip(), "predicted":prediction or "(no number)", "expected":answer, "correct":success})
    # A procedural skill is accepted only after multiple verified successes.
    # In this MVP, the stored rule is a human-readable hint; it is not a learned
    # neural algorithm and must not be treated as general mathematical mastery.
    for op, phrase in (("+", "addition"), ("-", "subtraction"), ("*", "multiplication")):
        n = store.verified_math_count(op)
        if n >= 3:
            store.save_skill(f"arithmetic:{op}",
                             f"Verified examples exist for integer {phrase} using operator {op}. "
                             f"Recheck every new result with the integer verifier.",
                             "python_integer_arithmetic", n)
    result = {"status":"completed", "total":count, "correct":correct, "accuracy":round(correct / count, 3), "examples":examples}
    store.log("math_experiments", "completed", result)
    return result
