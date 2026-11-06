# Your from-scratch model, trained with PyTorch

PyTorch is the active training/inference framework. The model architecture,
learned weights, source data preparation and training loop are this project's
own. No external pretrained weights, hosted model or remote model API is used.
Using PyTorch does not change that model provenance.

The temporary C runtime is inactive. Its accepted local weights and Adam
moments were converted into a normal PyTorch checkpoint, so returning to
PyTorch did not discard that training. Original `.pt` and `.bin` files and an
SQLite backup are preserved. New checkpoints are PyTorch `.pt` files again.

The useful data/training improvements remain: article-body extraction,
parent-anchored approvals, existing article holdouts, 256-byte windows,
persistent optimizer state, a continuing dataset cursor and per-attempt
receipts. The active derived dataset has 443 training, 60 validation and
71 test documents, with table-of-contents interface text excluded.

## Commands

For automatic internet collection plus language and source-backed QA training,
use the [autonomous pipeline](autonomous-pipeline.md). The `focus-learn` command
below remains the offline learner for previously saved text.

```sh
cd /Users/farisalsaiari/Desktop/berhot_evolve

# Start continuous PyTorch training when no worker is already running.
python3 -u -m evolve focus-learn \
  --steps 120 --context-bytes 256 --until-stopped

# Stop from another terminal or use Studio's Stop button.
python3 -m evolve focus-stop

# Start Studio and test the current model at /chat.
python3 -m evolve studio --port 8787

# Inspect the active model's dataset and recorded optimizer attempts.
python3 -m evolve dataset-status
open storage/datasets/focus_plain/active/saudi-history-ar
```

Use normal Python, without `-S`, so the installed PyTorch package is available.
The model remains a from-scratch GRU; restoring PyTorch does not introduce
another company's trained model. This dedicated learner trains saved local
data. It makes no external model API calls. Generated historical answers are
still unverified, and training loss alone does not establish factual accuracy.
## Repeated or incoherent samples

The same prompt, checkpoint, decoding settings and seed intentionally reproduce
the same generated text. The chat defaults to seed 42 and now shows that seed
beside each response. Changing it changes sampling, not model knowledge.

Training now interleaves one window from each unfinished article before
returning to the next window in a long article. The earlier sequential cursor
is converted to per-document offsets, retaining attempted coverage and saved
optimizer state. Entire documents still keep their existing hold-out splits.

Candidate acceptance also checks repeated word trigrams in both fixed prompt
samples. Lower byte loss cannot by itself promote a candidate that materially
worsens these loops. This is a limited fluency check, not a history verifier.
The brain API reports pass/coverage counters separately from checkpoint steps.
Those counters describe optimizer attempts, not facts learned or retained.
