# Your model, owned training and inference

**Archived experiment:** the user clarified that PyTorch is desired. The C
runtime described below is inactive. See [the active PyTorch workflow](pytorch-from-scratch.md).
Existing locally learned weights were preserved and restored into PyTorch.

The active Arabic model now uses project-owned Python/C code. It does not
load an external pretrained model, call a model API, or require torch, NumPy,
Playwright, Transformers, BLAS or a third-party ML runtime. The native library
is compiled locally from `evolve/owned_gru.c`.

The system foundations remain Python, its standard library, a local C compiler
and the operating system C/math runtime. On this Mac the compiler was already
installed. No package, model or dataset download was performed. Running Python
with `-S` disables site-package loading for this path. The existing packaging
metadata can be ignored; no pip installation is needed to run from the project.

## What changed

- The first-party C kernel implements embedding lookup, GRU gates, output
  projection, stable cross entropy, backpropagation through time, clipping and
  AdamW. Unit tests compare the gates against independent scalar equations,
  compare gradients with numerical derivatives, and verify optimizer resume.
- The 2,626,304 weights were imported from your own accepted checkpoint through
  a restricted standard-library archive reader. No torch import is needed to
  read that old file. It is not a downloaded pretrained model. Original files
  and an SQLite backup remain intact.
- New checkpoints use the project's `BEVGRU1` binary format, including weights
  and optimizer state. Older checkpoints contained weights only; the owned
  optimizer started with zero moments at migration. Later owned runs resume
  those moments instead of resetting them after every batch.
- Body extraction excludes structural navigation, tables of contents, sidebars,
  sharing controls and related-article blocks. The original raw/clean archive is
  unchanged. A new derived dataset is bound to the approved parent snapshots.
- Existing article splits remain attached to their parent snapshots. Equal
  cleaned bodies in conflicting partitions are excluded, never reassigned into
  the training set. Held-out text is not used for optimizer updates.
- Training now uses 256-byte windows and a persistent per-document cursor. It
  can continue through the same dataset across cycles. Its coverage counter
  records bytes used in optimizer attempts, not a claim that facts were learned.
- Candidate acceptance compares the same held-out windows before and after
  training. Test loss and generated-text heuristics are diagnostics only.
  Historical-answer accuracy is still unverified.

The first derived dataset has 443 training, 60 validation and 71 test documents.
It removed 906 occurrences of `جدول المحتويات`, leaving zero occurrences.
Its new files are under `storage/datasets/focus_plain/3461b7634411fb626c0d32d355c556f762458495ed49a29c33767b667a776ec1/`.
The original 585 source URLs and their archived HTML remain available.

## Run and inspect

```sh
cd /Users/farisalsaiari/Desktop/berhot_evolve

# Inspect runtime, code hash, model provenance and dataset.
python3 -S -m evolve owned-audit

# Rebuild or verify the derived body dataset locally.
python3 -S -m evolve owned-prepare

# Run one bounded training cycle.
python3 -S -u -m evolve owned-learn --steps 120 --context-bytes 256

# Continue local training until you stop it.
python3 -S -u -m evolve owned-learn \
  --steps 120 --context-bytes 256 --until-stopped

# Stop gracefully from another terminal, or use Studio's Stop button.
python3 -S -m evolve focus-stop

# Open the current accepted checkpoint's dataset and receipts.
open storage/datasets/focus_plain/active/saudi-history-ar
python3 -S -m evolve dataset-status

# Studio and /chat use the same accepted first-party model.
python3 -S -m evolve studio --port 8787
```

Do not start a second worker if one is already running. An OS lock prevents
duplicates. Ctrl+C and the Stop button preserve completed optimizer updates
through the candidate/evaluation procedure, with a receipt for the attempt.

Owned mode disables legacy model/search jobs in Studio and legacy commands
that require external ML runtimes. Automatic web discovery and model API
calls are off. This worker learns from existing local data; it does not gather
new pages. Legacy source files remain for historical compatibility, disabled
in the active mode. Already installed packages elsewhere on the Mac were not
uninstalled or changed.

New rejected weights are discarded after their measured result is recorded;
their receipts remain. Accepted snapshots and the older checkpoints are kept.
The learner stops with an error rather than saving another accepted snapshot
when less than 2GB of disk space remains.

## Interpreting progress

The first owned run made 120 real updates. On its fixed 256-byte held-out body
windows, loss improved from 1.571197 to 1.051075, so the checkpoint was accepted.
These scores use a new body/context evaluation setup and should not be compared
directly to old 48-byte/full-page scores.

Generated samples still repeat phrases and contain unreliable historical
content. Better next-byte loss is measurable progress, not evidence of a
working historical assistant. This pipeline has no instruction-answer training
or factual-answer evaluation yet. Your raw model response remains visible;
retrieved text or another model's output is never substituted.
