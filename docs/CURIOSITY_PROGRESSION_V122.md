# BERHOT EVOLVE v1.22 — Curiosity progression, rights qualification, neural baseline

This upgrade uses the existing BERHOT EVOLVE Python repository, SQLite database
and PyTorch ranker. It does **not** replace or reset prior checkpoints.

## One sequence on the Mac

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'

# The newest unanswered topics (offline, without modifying network quota)
python3 -m evolve v121-plan --max-topics 8

# Archive eligibility and explicit test-host gaps (read-only)
python3 -m evolve v122-evidence

# Perform a bounded independent cycle; remove --network for offline
python3 -m evolve v121-cycle --network --max-topics 8 --max-requests 6 --steps 120

# Compact status (including holdout and lexical comparison when trained)
python3 -m evolve v121-report
```

## What changed

### 1. Curious about **new** observed words, not the same words forever

The planner checks previously researched/cached subjects in
`v119_finding:...`, same-UTC-day attempts from the v1.21 cycle, and unresolved
research retry cooldowns. It excludes subjects already answered by an
approved-source definition. If no previously unattempted topics remain, it
returns an explicit empty queue. It does not manufacture arbitrary new topics.

Research-only citations remain source memory. A successful source lookup does
**not** change the language model's neural weights, and is not retrained as
a model answer without appropriate rights and evaluation.

### 2. Rights/source evidence qualification

`python3 -m evolve v122-evidence` reads each archived, hash-verified HTML
extraction, without making network requests or modifying its rights status.
The report shows:

- Host counts, currently approved versus UNKNOWN/no-AI pages, and heuristic
  definition-topic candidates.
- Exact archived URLs needing an **individual rights review** before their
  candidate sentences could be admitted.
- Already-approved pages that contain no extractable definitions.
- The host partition and missing test examples **at the last cycle**, not
  a promise that the deterministic split will stay fixed when new hosts arrive.

No automatic conversion from public website to PUBLIC-DOMAIN occurs.
Publisher access restrictions, robots.txt, and 200/day shared network quotas
continue to apply. Third-party material cannot be declared reusable merely
because it is hosted on a government domain.

### 3. Real PyTorch versus lexical benchmark on the **same test pairs**

When enough distinct reviewed-source hosts exist, v1.20's ranker still
trains on training-only pairs, chooses whether to accept by held-out
validation, and scores the test set once. v1.22 additionally scores those
same frozen source-host-held-out pairs using the original lexical formula.

The model record's `benchmark` includes:

- `lexical_validation`, `lexical_test`
- `neural_validation_pair_accuracy`, `neural_test_pair_accuracy`
- `validation_accuracy_delta_vs_lexical`, `test_accuracy_delta_vs_lexical`
- `outperformed_lexical_on_unseen_test_hosts`
- explicit flags that **test is not used to promote a checkpoint**.

This is an honestly limited benchmark. The negative candidates come from
different topics, and the lexical features have automatically extracted
topic metadata. The task is easy and **does not establish new semantic
generalization or fluent understanding**. A 100% score is not proof of
scientific intelligence. Better neural learning benchmarks will require
more diverse sourced and independently reviewed QA data.

## Current real-world blocker

Your last run had 4 USGS training, 5 NASA validation, 0 NOAA testing
definitions. This is enough for source-quote exercises but not an independent
neural ranker evaluation. EVOLVE will report `needs_diverse_definitions`
instead of training on the validation publisher or inventing test data.
The evidence report helps identify archived candidate articles; additional
lawfully reusable material from an independent host is still needed.

No hallucinated answers, no auto-approved licenses, no held-out host leakage.
