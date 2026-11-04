# BERHOT EVOLVE v1.18 — Learning Studio, source chat and controlled learning

## Open the app on your Mac

From the same existing `berhot_evolve` folder and active `.venv`:

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve studio --port 8787 --interval 3600
```

Open **http://127.0.0.1:8787** in your browser. Keep this terminal open.
Stopping it stops automatic checks and background training. The app binds to
**loopback only**, not to LAN, Wi-Fi or a public domain.

The dashboard is actual HTML/CSS/JS served by Python's standard-library
HTTP server. It is not a static mockup. It polls live local SQLite state every
2.5 seconds, displaying archive counts, source host count, training progress,
elapsed seconds, estimated remaining seconds, loss, checkpoints and job history.
Local-only POST endpoints require same-origin + an ephemeral secret header.
Do not expose this development server on the internet; there is no account
authentication, multi-user permissions, or production security hardening.

## Stage A — More diverse approved text

The dashboard's **Curriculum** panel shows how many eligible articles, hosts,
text characters and human-reviewed QA lessons the system currently has.
It can queue explicit HTTPS seed articles and run a bounded crawler with a
separate user click; existing robots/access protections and daily budgets
remain in effect.

The source archive distinguishes **UNKNOWN** rights from
`CC0-1.0`/`PUBLIC-DOMAIN`. The crawler never approves the former by itself.
For a new collected article, use the existing manual `v117-sources` and
`v117-review-rights` commands after checking the article, terms, authorship
and third-party rights. Never automatically classify a host as public domain
just because it ends in `.gov`.

Goals like **100 eligible articles** and **one million characters** are
progress indicators, not claims that this is enough for a fluent foundation
model. Train/validation/test partitions remain disjoint by source host.

## Stage B — Speak to BERHOT and teach reviewed answers

The chat panel has **four clearly labeled modes**:

1. **Approved source lookup (default):** extracts short, literal passages
   from archived CC0/public-domain source text and shows source links. This
   is *retrieval*, not a neural response or a demonstration of comprehension.
2. **Byte-BPE raw neural continuation:** loads the actual SHA-verified v1.17
   byte-BPE weights, encodes the prompt and samples 48 new tokens. Often
   incoherent because the model is tiny and data is scarce.
3. **Character raw neural continuation:** samples the SHA-verified v1.17
   Unicode character checkpoint. May be incoherent.
4. **Supervised QA neural answer:** samples a separate, locally trained
   256-byte-token answer-only Transformer, if an accepted checkpoint exists.

**Chatting does not train weights.** The dashboard has a separate **Teach
a question & answer** form for explicitly reviewed examples. The provided
answer must be an exact quotation present in an eligible SHA-verified archived
article. The source host determines the training, validation or held-out test
partition. To start answer-only training, prepare at least **8** reviewed
examples on train-host documents, **2** for validation-host documents, and
**2** for test-host documents. Short question + answer pairs must fit in
the 256-byte context window.

Then click **Train QA brain**, or run:

```bash
python3 -m evolve v118-qa-readiness
python3 -m evolve v118-qa-train --steps 60
python3 -m evolve v118-chat --mode qa --question "What is a volcano?"
```

The code performs real PyTorch backpropagation on the answer suffix only,
measures validation loss before and after, and selects an accepted QA
checkpoint based only on validation, not test. QA manifests are immutable
and SHA-verified, with distinct checkpoints in `storage/models/v118_qa`.
The model may remain unable to answer accurately even after acceptance.

The existing verified arithmetic answer-specialist remains available
independently through `math-finetune` and existing CLI commands; v1.18 does
not overwrite its weights.

## Stage C — Learn when new reviewed data becomes ready

**Enable auto checks** in Studio (off by default). The scheduler periodically
checks for a **different** strict SHA-frozen dataset with at least:

- 3 independent, rights-eligible source hosts;
- 12 eligible articles;
- 60,000 bytes of frozen **training** text.

If all conditions are met, the Studio launches one bounded 120-step-per-family
v1.17 char and BPE experiment. It does not collect new sites, grant training
rights, choose independent research objectives, or automatically modify
previous checkpoint files. A single scheduler process runs only while the
Studio terminal stays open. Manual training is always available with
**Train text brain**; it runs the standard v1.17 real training flow and
reports each ten-step progress update.

**Important:** the original v1.17 training still starts a *fresh*
small model on each dataset. This stage does **not** implement continuous
neural training or lossless optimizer-state transfer across new tokenizers
and datasets, and cannot promise monotonic intelligence improvements.
The previous v1.16 and v1.17 checkpoint artifacts remain on disk; Studio
training creates new timestamped, SHA-verified files. Auto training is an
experiment scheduler, not a self-aware AI.

## CLI examples

```bash
python3 -m evolve v118-plan
python3 -m evolve v118-chat --mode sources --question "What is a volcano?"
python3 -m evolve v118-chat --mode byte_bpe --question "The sun"
python3 -m evolve v118-qa-add \
  --question "What does this science passage explain?" \
  --answer "PASTE AN EXACT SENTENCE FROM THE APPROVED ARCHIVED ARTICLE" \
  --source-url "https://RIGHTS-REVIEWED-ARTICLE-URL"
python3 -m evolve v118-qa-readiness
python3 -m evolve v118-qa-train --steps 60
```

All data is stored under the existing local `storage/` directory, not
uploaded to an external service. You can inspect current jobs via the
browser and stop auto checks using the Stop button. Stopping the server
during training can interrupt the current run, so avoid terminating it
mid-checkpoint-save.

## Automated verification

```bash
python3 -m unittest discover -s tests -p 'test_v118_studio.py' -v
python3 -m unittest discover -s tests -v
```

Tests cover: strict source permission, quote-backed supervised examples,
duplicate rejection, cross-host separation, real QA CPU gradients,
validation-only acceptance, immutable QA manifest and SHA verification,
controller serialization, source chat provenance and local HTTP origin/CSRF/
Host checks. CI also runs the full suite with PyTorch.

## What remains for a production autonomous AI

EVOLVE is still a laboratory. A much larger rights-cleared corpus,
deduplicated licensed document bundles, a stronger tokenizer, scalable
compute, long-context pretraining, instruction tuning, human and automated
generalization evaluations, independent safety/reliability checks, model
promotion workflow, durable job queue, account security and a hardened web
service remain future engineering work. Today's chat is a helpful window
into the experiment, **not** general conversational intelligence.
