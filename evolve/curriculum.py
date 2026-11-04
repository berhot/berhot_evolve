"""A first automatic curriculum: verified arithmetic examples, never scraped answers."""
import hashlib
import random


def expand_math_curriculum(store, settings):
    """Generate deterministic, computer-verified lessons once enough source text exists.

    This is engineered curriculum, not evidence the random model invented mathematics.
    As abilities emerge, replace policy with measured frontier learning progress.
    """
    source_docs = store.db.execute(
        "SELECT COUNT(*) FROM documents WHERE source_id NOT LIKE 'synth:math:%'"
    ).fetchone()[0]
    if source_docs < 8:
        return {"status":"not_ready", "source_documents":source_docs}
    max_total = 3000
    existing = store.db.execute(
        "SELECT COUNT(*) FROM documents WHERE source_id LIKE 'synth:math:%'"
    ).fetchone()[0]
    if existing >= max_total:
        return {"status":"complete", "existing":existing}
    round_id = store.get("math_curriculum_round", 0)
    rng = random.Random(1234 + round_id)
    # Adaptive difficulty remains deliberately primitive in MVP.
    difficulty = 1 if round_id < 5 else 2
    limit = min(30, max_total - existing)
    added = 0
    for _ in range(limit * 3):
        if added >= limit:
            break
        highest = 9 if difficulty == 1 else 30
        a, b = rng.randint(0, highest), rng.randint(0, highest)
        op = rng.choice(["+", "-"] if difficulty == 1 else ["+", "-", "*"])
        result = a + b if op == "+" else a - b if op == "-" else a * b
        # Answer calculated independently by Python's integer arithmetic.
        text = f"Question: {a} {op} {b} = {result}\n"
        digest = hashlib.sha256(text.encode()).hexdigest()[:24]
        if store.save_doc(f"synth:math:{digest}", text, "internal://integer-verifier",
                          "SELF-GENERATED", topic="arithmetic"):
            added += 1
    store.put("math_curriculum_round", round_id + 1)
    report = {"status":"generated", "round":round_id, "new_math_lessons":added, "difficulty":difficulty}
    store.log("curriculum", "completed", report)
    return report
