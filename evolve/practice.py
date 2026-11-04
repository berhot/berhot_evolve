"""Independently checkable arithmetic practice for continual-learning smoke tests.

This is engineered math practice, not facts learned from webpages. Examples are
generated and checked in Python without executing website-provided code.
"""
import hashlib
import random

METHOD = "python_integer_arithmetic_plus_inverse_check_v1"


def checked_answer(a, b, operator):
    """Verify integer arithmetic through an independent inverse relation."""
    if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a <= 99 and 0 <= b <= 99):
        raise ValueError("Small nonnegative integers only")
    if operator == "+":
        answer = a + b
        valid = answer - a == b and answer - b == a
    elif operator == "-":
        answer = a - b
        valid = answer + b == a and a - answer == b
    elif operator == "*":
        answer = a * b
        valid = (a == 0 or answer // a == b) and (b == 0 or answer // b == a)
        valid = valid and answer == sum([a] * b)
    else:
        raise ValueError("Only +, - and * are supported")
    if not valid:
        raise ArithmeticError("Independent inverse verification failed")
    return answer


def practice_arithmetic(store, limit=32, maximum=2000, skill="mixed"):
    """Generate bounded, unique, verified QA documents; never use web content."""
    if skill not in ("mixed", "single-digit-add"):
        raise ValueError("Skill must be mixed or single-digit-add")
    limit = int(limit)
    if not 1 <= limit <= 100:
        raise ValueError("Practice batch must contain 1..100 exercises")
    previous = store.db.execute(
        "SELECT COUNT(*) FROM documents WHERE topic='verified:arithmetic'"
    ).fetchone()[0]
    if previous >= maximum:
        return {"status": "complete", "new_examples": 0, "total_examples": previous}
    round_key = "verified_arithmetic_round" if skill == "mixed" else "verified_arithmetic_round_single_digit_add"
    round_id = int(store.get(round_key, 0))
    rng = random.Random(1031 + round_id)
    new = 0
    problems = []
    for _ in range(limit * 30):
        if new >= min(limit, maximum - previous):
            break
        if skill == "single-digit-add":
            a, b, op = rng.randrange(0, 10), rng.randrange(0, 10), "+"
        else:
            a, b = rng.randrange(1, 80), rng.randrange(1, 80)
            op = rng.choice(["+", "-", "*"])
        answer = checked_answer(a, b, op)
        question = f"{a} {op} {b}"
        lesson = f"Question: {question} = ?\nAnswer: {answer}\n"
        key_prefix = "verified:arithmetic:single-digit-add:" if skill == "single-digit-add" else "verified:arithmetic:"
        source_id = key_prefix + hashlib.sha256(question.encode()).hexdigest()
        if store.save_doc(
            source_id=source_id, text=lesson, url="internal://python-arithmetic-verifier",
            license_id="SELF-GENERATED-VERIFIED", topic="verified:arithmetic",
            modality="text",
        ):
            new += 1
            if len(problems) < 5:
                problems.append({"question": question, "answer": answer})
    if new:
        store.put(round_key, round_id + 1)
    report = {
        "status": "generated" if new else "complete",
        "method": METHOD,
        "skill": skill,
        "round": round_id,
        "new_examples": new,
        "total_examples": previous + new,
        "examples": problems,
        "note": "Independent Python arithmetic practice; separate from web discovery and not proof of neural comprehension.",
    }
    store.log("verified_arithmetic_practice", report["status"], report)
    return report
