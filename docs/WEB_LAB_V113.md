# BERHOT EVOLVE v1.13 — Corpus → Real Neural Learning, in One Command

## Scope of the milestone

One existing-project workflow now performs article curation, byte-text
deduplication, immutable dataset freeze, actual gradient-based PyTorch
training, validation-based checkpoint selection, heldout test evaluation,
arithmetic language-loss diagnostic, immutable checkpoint hashing, and
weight-only text generation. No web requests are made during training.
Data archive and previous models remain intact.

## Your current four collected Wikipedia pages

Based on the user's measured local corpus on November 4, 2026:
- Energy: 12,690 words, 81,384 characters — curated article candidate
- Portal:Current_events: dynamic index — excluded
- Main_Page: navigation — excluded
- Wikipedia:Contents: directory — excluded

Only one article and one host are suitable, so STRICT source-host heldout
training must SKIP. A separate, explicitly labeled within-document mode
permits an immediate small smoke test. Train/validation/test blocks have
distinct content hashes, but all originate from Energy. Results of that
experiment do NOT prove cross-article or cross-source generalization.

## Run on your Mac

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# 1. Read-only curation
python3 -m evolve web-inventory --include-unknown-rights

# 2. Full experimental pipeline, with existing archive; NO network calls
python3 -m evolve web-lab --steps 80 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document

# 3. Re-verify candidate checkpoint and frozen shards; no training
python3 -m evolve web-evaluate

# 4. Generate from weights only (unverified text continuation)
python3 -m evolve web-predict --prompt 'Energy is' --new-bytes 100

# 5. Inspect actual local checkpoint and learning metadata
python3 -m evolve model-audit
```

Pass `--device mps` for a compatible Apple Silicon installation if desired.
To perform STRICT host-level evaluation after collecting articles from
three or more distinct hosts, OMIT `--diagnostic-within-document`.
`web-freeze` also creates the immutable dataset without training.

## Actual training behavior

- Classifier removes wiki portal, directory, homepage and current-event
  pages and low-word-count content without deleting originals.
- Records script hint (Latin vs Arabic), word counts, URL and rights metadata;
  this is NOT reliable language identification or factual quality scoring.
- SHA256 exact normalized-text dedup is done by web corpus; 5-token ngram
  Jaccard high-overlap chunk filtering mitigates near-duplicate leakage.
- Frozen datasets live in storage/datasets/web_lab/<dataset-SHA>/ with
  train.bin, val.bin, test.bin, manifest.json. Every file is hash verified
  before training and re-evaluation; mutations are rejected.
- Strict mode groups every article from a source host into a single split,
  requiring three different hosts. Diagnostic mode explicitly splits
  non-overlapping chunks within article and labels source leakage true.
- A NEW tiny 1-layer byte Transformer with PyTorch autograd trains via
  actual cross-entropy gradients, AdamW, and scheduled optimizer steps.
  Training steps and before/after train/validation losses are recorded.
- A candidate checkpoint is stored in storage/models/web_lab/ independently
  from general and arithmetic checkpoints. SHA256 prevents silently
  using a tampered checkpoint. Validation alone chooses acceptance.
- Withheld test byte loss is measured only AFTER checkpoint selection.
- The arithmetic retention proxy compares loss on old heldout arithmetic
  snippets before/after training of the NEW model; it is NOT arithmetic
  problem accuracy, and this is NOT continual fine-tuning of the old model.
- Web-predict uses only model weights, NOT a database answer lookup.

## Rights and provenance

An UNKNOWN license does not prevent archiving and curation. For local
experimental training it requires the operator's explicit
`--include-unknown-rights` choice. The flag does not establish rights,
and training/redistribution must comply with applicable law and source
terms. Pages with publisher no-AI signals are excluded from this training
experiment. Robots/authentication/rate-limit controls remain intact.

## Limits and interpretation

The tests prove the SOFTWARE WORKFLOW, not any actual learned facts from
the user's Mac. Lower train/heldout byte loss is evidence of changed
language-model probabilities, not trustworthy general world knowledge,
reasoning or question answering. Training more web pages with robust
heldout source groups is required for independent conclusions.

## Dependencies

Data curation/freeze uses original BERHOT Python standard-library code.
Neural training still uses PyTorch and its external tensor/autograd backend;
this is NOT yet a fully independent BERHOT tensor implementation.
