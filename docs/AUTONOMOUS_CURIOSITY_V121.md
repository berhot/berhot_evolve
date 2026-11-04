# BERHOT EVOLVE v1.21 — Autonomous Curiosity Study Cycle

This is **one integrated, bounded learning experiment**. It runs on your Mac,
uses the existing BERHOT repository, and preserves all existing model
checkpoints. It does not use OpenAI/another language model.

## Run on your Mac

Stop any previously running Studio (Ctrl+C), then:

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'

# Preview actual missing questions and training/test blockers, without network:
python3 -m evolve v121-plan --max-topics 6

# Run the whole cycle with up to 6 robots-compliant HTTPS research requests:
python3 -m evolve v121-cycle --network --max-topics 6 --max-requests 6 --steps 120

# Inspect source-derived quiz lessons, model status and each past cycle:
python3 -m evolve v121-history
python3 -m evolve v120-readiness
python3 -m evolve model-audit
```

Use `v121-cycle` **without `--network`** for a fully offline
cycle. You can also run Studio and press **Run curiosity → research → learn →
test**. Explicitly enabling Studio automatic checks permits one limited
autonomous web cycle per UTC day (six requests maximum, and still within the
shared daily request budget), only while the local Studio server is running.

## What happens inside one cycle?

1. **Read the actual observed data.** Select short topics already saved in the
   `research_questions` and `concept_tasks` queues, or taken from literal
   article questions. No arbitrary focus word, random universe-wide keyword,
   or language-model-made-up premise is inserted.
2. **Ask a question.** For a missing concept, derive a bounded question such
   as "What is photon?" **from an observed word**. If an approved definition
   already exists, prefer asking source-backed review questions rather than
   wasting the research budget.
3. **Research (opt-in).** The unchanged existing safe HTTPS/robots engine
   investigates permitted public pages, stores citations and research
   excerpts, and honors the combined daily quota. Unavailable/blocked
   websites are logged, not bypassed. Retries are bounded and same-day
   duplicate topics are skipped.
4. **Learn from existing eligible evidence.** Build a SHA256-named immutable
   QA bank of *real literal sentences* from the already rights-approved
   corpus. Each question contains a source URL, source SHA256, topic and
   exact sentence SHA256. Two question variants are generated:
   "What is X?" and "Define X". These are *mechanically labeled*, not
   human-checked explanations.
5. **Self-test the source quotes.** Quiz the local retrieval system for each
   known subject and recompute the archived source SHA and literal sentence
   match. This measures **citation integrity**, not semantic reasoning.
6. **Train actual weights, if data permits.** Run the v1.20 PyTorch
   question/evidence ranker with both source-derived question variants and
   mismatched evidence, strict independent publisher-host train/validation/test
   splits, validation-only checkpoint promotion and held-out test reporting.
   The training step will be explicitly *skipped* if the test dataset is
   missing. The cycle never fabricates a third publisher, lowers the minimum,
   or promotes a model on the basis of its training score.
7. **Evaluate and save a report.** View the exact training status, research
   attempts, number of verified source quotes, evidence host counts and
   checkpoint SHA (if saved). Cycle reports are in the local SQLite database,
   and question manifests in `storage/learning/v121/`.
   An unchanged rights-approved corpus does not trigger repeated PyTorch
   training.

## What if real data has no held-out test source?

With the user's current dataset (4 USGS definitions, 7 NASA definitions,
zero NOAA test definitions), expect:

- Curiosity questions will be autonomously selected and researched, subject
  to website availability and the shared daily limit.
- Source-grounded lesson creation and source-integrity tests work offline.
- The neural evidence ranker reports
  `completed_with_training_blocker` / `needs_diverse_definitions`
  until two distinct testing topics are available from a permitted third
  independent source. **This is a correct result**, not a failed Python run.

Research is *not* an automatic license grant. An article fetched with
unknown rights remains research-only. Do not set `PUBLIC-DOMAIN` on a
website just because it is a government organization. Verify each
specific source before approving neural training with the existing
rights-review command. Existing NOAA restrictions and timeouts are not
circumvented.

## What counts as actual intelligence?

- **Source retrieval:** Find a sentence about the question.
- **Self-created QA examples:** Generate source-faithful question/answer
  pairs; this is supervision generation, but it is **not** semantic
  comprehension by neural weights.
- **Neural evidence learning:** Real PyTorch parameters learn to
  distinguish relevant evidence from irrelevant passages, evaluated on
  unseen source hosts. It does not compose explanations.
- **General autonomous reasoning and fluent new question answering:**
  **Not established by v1.21.** Reaching that requires stronger model
  pretraining, diverse human/external evaluation, verifiable tasks, and
  independent test sources. The output flags this explicitly.

Do not treat a 100% exact-quote integrity score as a 100% scientific-answer
score. These are different tests.
