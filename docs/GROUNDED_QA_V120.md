# BERHOT EVOLVE v1.20 — Grounded Questions and a Learned Evidence Ranker

The original problem:

- Asking **sun** retrieved a valid but tangential NASA quote:
  "Our Sun is roughly midway through its main sequence stage."
- The v1.18 tiny byte generator emits broken UTF-8 and cannot reliably
  explain the Moon, the Sun, or volcanoes.
- 54 automatic "Continue: ..." examples practice text continuation,
  **not factual question answering**.

This release adds a deliberately *different* educational system. It gives
a readable definition **only when a sentence in an already-reviewed local
article explicitly defines the topic**. It can then learn a small neural
function to rank evidence candidates. It does **not** generate new prose,
verify the publisher's truth, or claim ChatGPT-level science understanding.

## Run it on your Mac

Stop the old Studio first with Ctrl+C, then:

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve studio --port 8787 --interval 60
```

Open http://127.0.0.1:8787 and refresh.

Choose **Curiosity** (default) and ask:

- `sun`
- `What is the Sun?`
- `What is a volcano?`

Curiosity now checks exact local *definition evidence* before falling back
to the v1.19 general-passage retrieval/research flow. If the approved NASA
article includes an explicit "The Sun is ..." source sentence meeting the
conservative definition gate, you will see that sentence and its NASA URL
instead of a tangential sentence about its lifetime. If not, the older
source-only path still applies; we do not manufacture an unsupported
definition.

Choose **Grounded Q&A** to search *only* local reviewed definition sentences
and never request the public internet. A missing definition is clearly
reported instead of fabricated. Why/how questions are deliberately
unsupported unless a future dedicated evidence verifier is implemented.

## Train evidence selection

In Studio click **Learn to select evidence · 120 steps** or run:

```bash
python3 -m evolve v120-inventory
python3 -m evolve v120-readiness
python3 -m evolve v120-ask --question "What is the Sun?"
python3 -m evolve v120-train --steps 120
python3 -m evolve model-audit
```

**What is learned?** A small 7-input, 12-hidden-unit PyTorch ranking model
learns to score matching question/evidence pairs more highly than mismatched
pairs. Its training examples come from exact, sourced definitional
sentences, with negatives drawn from different topics in the *same
partition*. It uses a reproducible frozen source-host partition from v1.17:

- **train**: articles from one source-host set only
- **validation**: independent source hosts, used for acceptance
- **test**: independent source hosts, reported only after training

All training material is sourced from articles previously marked CC0 or
public domain and not flagged no-AI. Unknown-rights research-only web
pages do **not** become training examples. Each example keeps source URL,
source SHA-256 and sentence SHA-256 in a content-addressed JSON manifest.
PyTorch state is saved in a new immutable `models/v120` checkpoint and
the checkpoint is verified on reload. Old v1.17/v1.18 models are untouched.

To train, the system requires at least **4 train, 2 validation, 2 test**
extracted definitions and **at least 2 distinct subject topics in every
partition**. If your approved corpus doesn't contain enough *explicitly
written definitions*, you will see `needs_diverse_definitions`; this is
not a bug or a reason to silently invent training examples. Expand
legally reusable science articles from independent source hosts, then
run readiness again.

The neural ranker is accepted only if it improves **held-out validation
pairwise loss** and achieves at least 50% pair accuracy on that small
synthetic comparison. The held-out test pair accuracy is recorded separately.
This limited metric is **not** a measure of scientific understanding,
human judgment, source reliability, or question answering accuracy on
unseen real questions.

When an accepted ranker exists, EVOLVE can use its weight output as a
small evidence-ranking tiebreaker. The answer remains an *exact source
quote* with a clickable citation; the neural model does not compose
the answer or edit any scientific claim. The earlier byte/character
generation modes remain available as raw experimental neural output.

## Automatic learning

When **Enable auto checks** is turned on, the local Studio process checks
approved source data at the configured interval. For a new approved corpus
revision it may run a bounded source-continuation experiment and separately
attempt a v1.20 evidence-ranker experiment; both require the relevant
data/host constraints. An unchanged revision does not retrigger the
ranker indefinitely.

Automatic checks do not browse new webpages except when the user asks a
new simple topic in Curiosity mode, and never approve rights or make
up factual labels. The Studio must be running for these checks.

## Interpretation

There are three different abilities. Do not conflate them:

1. **Finding an answer**: looking up source text and citing its URL.
2. **Ranking evidence with learned weights**: a tiny neural pairwise scorer,
   held out across independent publisher hosts.
3. **Generating meaningful explanations from neural weights**: **not achieved**.

This is a useful working stage of an evidence-grounded QA pipeline, not
a general reasoning language model. A production version would require
publisher-reviewed QA labels, harder negative examples, better passage
segmentation, semantic retrieval, large-scale pretraining, independent
held-out human-judged QA benchmarks, and potentially a much larger
conditional language generator.

If v1.20 finds too few definitions in your three existing approved
articles, the correct next action is to collect more diverse rights-cleared
source text and evaluate newly acquired evidence, **not** train 10,000
steps on the same 54 cloze examples.
