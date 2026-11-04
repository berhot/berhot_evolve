# BERHOT EVOLVE v1.15 — Unicode Character Learning + Safe Continuation

## Why this exists

The local 400-step byte model beat a training-only byte unigram baseline
on the Energy article's diagnostic splits, but still emitted repetitive
pseudo-words and invalid UTF-8 when sampling. A reduced byte cross-entropy
does not mean factual understanding, grammatical sentences, or general
language ability.

This release adds a separately trained Unicode code-point model using an
original BERHOT stdlib-only tokenizer, longer context and optional real
checkpoint continuation. It does NOT claim this automatically fixes
language fluency or that PyTorch has been replaced.

## One-shot existing corpus test

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Train 128-character context, 64-dimensional, 2-layer Unicode model
python3 -m evolve char-lab --steps 400 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document

# Verify frozen corpus/tokenizer/model hashes and heldout character CE
python3 -m evolve char-evaluate

# Generate VALID Unicode characters without independently sampled UTF-8 bytes
python3 -m evolve char-predict --prompt 'Energy is' --new-chars 100 \
  --temperature 0.85 --top-k 30 --seed 42

# Optional: continue the accepted checkpoint, rather than starting again
python3 -m evolve char-lab --resume --steps 400 --seed 31415 \
  --device cpu --include-unknown-rights --diagnostic-within-document

# Audit both original byte and new char experiments
python3 -m evolve model-audit
```

## What's independent / what's reused

- Tokenizer: original BERHOT Python standard-library characters, learned
  ONLY from frozen training text. No pretrained tokenizer or HF library.
- Characters are Unicode code points; generated text never reconstructs
  partial UTF-8 bytes. Unknown ID and control characters are masked at
  generation time. This ensures valid Unicode representation, NOT correct
  or fluent words.
- Transformer: still uses torch.nn, PyTorch autodiff and AdamW.
- Data: same SHA256 verified frozen web train/validation/test partitions
  from v1.13, no hidden downloads or source text used for inference.
- Training objective: next Unicode character, cross entropy.
- Exact dataset, tokenizer and checkpoint digests are verified on read.
- Separate checkpoints under storage/models/char_lab/.
- Resume: only the accepted matching char model with the SAME frozen
  dataset, tokenizer and model config. Incompatible runs skip instead of
  silently resetting, and older accepted checkpoints remain on disk.
- Candidate selection: validation cross-entropy only; test sampled after
  validation acceptance. We report honest train-only char unigram losses.

## Evaluation caveats

The four archived Wikipedia pages yield only one substantive Energy
article. Use the explicit diagnostic flag to train now; the split shares
underlying source material, so it cannot prove cross-article generalization.
With three distinct substantive source HOSTS, omit the flag for strict
host-heldout evaluation. No source count alone proves independence of facts.

**Do not compare character cross-entropy numerically to byte cross-entropy:**
the target units are different. Compare within each experiment to its
own train-only unigram baseline, or construct a normalized bits-per-byte
metric on identical source content for later rigorous comparisons.

Explicit unknown-rights opt-in is only a local research decision; it does
not establish permission to train, redistribute or deploy a model on any
particular external content. Existing no-AI training exclusions remain.

## Main model preservation

The Unicode experiment never calls the old main/arith training APIs or
modifies old checkpoints. If a main checkpoint is present on disk, its
SHA256 is measured before and after each Unicode training run.

## Next capability gap

Even with a bigger character model and 800+ updates on one article, expect
low accuracy as a knowledge assistant. Larger, genuinely diverse corpus
data; source-host evaluation; more training; subword/BPE tokenization;
prompt/answer supervision and independent task benchmarks are separate
subsequent research requirements. A Unicode tokenizer does not itself
create language understanding.