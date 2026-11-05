# BERHOT EVOLVE v1.16 — Stateful Resume & 400-to-800 Comparison

The user's Energy article character model was trained 400 steps, then
resumed 400 more (800 recorded), with lower train/validation losses.
However, v1.15's resume restored only model weights; it RESET AdamW
optimizer moments and the Python batch-sampling sequence.

v1.16 checkpoints now persist and restore **optimizer state, Python
sampler RNG state, and CPU torch RNG state**. On CPU, tests compare
15+15 resumed updates against 30 uninterrupted updates for identical
weights. Older v1.15 checkpoints remain readable and resumable, but
are transparently flagged `legacy_weights_only`, not exact continuation.

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Compare your prior 400-step checkpoint to the latest 800-step model
# Using identical frozen data, heldout windows and generation seeds:
python3 -m evolve char-compare \
  --reference-sha256 ca1902a0ddaa907cea23bd6b4a22402a02580fe8c82e77e57b72bc44b4e2574a \
  --prompt 'Energy is' --new-chars 100 --temperature 0.85 --top-k 30 --seed 42

# Generate directly from the latest 800-step checkpoint:
python3 -m evolve char-predict \
  --prompt 'Energy is' --new-chars 100 --top-k 30 --seed 42

# Continue 800->1200: because the existing 800-step model was created
# by v1.15, the first v1.16 resume reports legacy_weights_only.
# Its NEW checkpoint will include optimizer+RNG state for future exact resumes.
python3 -m evolve char-lab --resume --steps 400 --seed 31415 \
  --device cpu --include-unknown-rights --diagnostic-within-document
python3 -m evolve model-audit
```

All old and new Unicode checkpoints are separate files under
`storage/models/char_lab`; `char-compare` checks both SHA256 digests,
tokenizer, model architecture and dataset identity and does NOT train.
Comparison test scores do not prove independent generalization because
the source is still one Energy article.

PyTorch remains the neural backend; this release doesn't claim fluent
English, independent reasoning, external-data rights or a proprietary
neural tensor framework.

## Previous version — v1.15
# BERHOT EVOLVE v1.15 — Original Unicode Tokenizer, Resumable Neural Experiment

The real 400-step Energy model outperformed a training-only byte
unigram baseline, but greedy output still looped through fragments
and byte sampling generated invalid UTF-8. v1.15 adds a fully
BERHOT-authored **Unicode character tokenizer** and a SEPARATE longer
context neural experiment, with safe character-level generation and
resume-from-accepted checkpoint support. The Transformer still uses PyTorch.

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# One-shot Unicode-char training on frozen Energy article:
python3 -m evolve char-lab --steps 400 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document

# Validate hashes and independently repeat heldout CHAR loss:
python3 -m evolve char-evaluate

# Test actual generated Unicode text, NOT database retrieval:
python3 -m evolve char-predict --prompt 'Energy is' --new-chars 100 \
  --temperature 0.85 --top-k 30 --seed 42

# Continue accepted weights instead of restarting:
python3 -m evolve char-lab --resume --steps 400 --seed 31415 \
  --device cpu --include-unknown-rights --diagnostic-within-document

python3 -m evolve model-audit
```

Train/validation/test data remain derived from the same Energy article;
this is an explicitly labeled diagnostic, **not independent source
generalization or verified knowledge**. Character cross-entropy cannot
be directly compared with v1.14's byte cross-entropy; both experiments
should be compared to their own train-only frequency baselines.

The character tokenizer and persistence logic use the Python standard
library; **PyTorch is still used for tensors, autograd, and optimization**.
Existing general, numeric, and web-byte checkpoints are not overwritten.
Unknown-rights data requires explicit opt-in and no-AI signals remain
excluded from experimental model training.

See docs/UNICODE_CHAR_LAB_V115.md for technical details.

## Previous version — v1.14
# BERHOT EVOLVE v1.14 — Measure Web-Language Learning vs Byte Frequency

The user's 80-step Energy-article experiment reduced next-byte loss but
greedy generation collapsed to spaces and `e`. v1.14 checks that result
against a matched, **training-only 256-byte unigram predictor** rather
than interpreting loss reduction as factual comprehension.

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# No training/network calls or checkpoint changes:
python3 -m evolve web-diagnose --prompt 'Energy is' --new-bytes 80 \
  --seed 42 --temperature 0.9

# Compare same checkpoint with both decoding modes:
python3 -m evolve web-predict --prompt 'Energy is' --new-bytes 100 \
  --decoding sample --temperature 0.85 --seed 42

# Optional fresh longer experiment; separate checkpoint, NOT a resume:
python3 -m evolve web-lab --steps 400 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document
```

The diagnostic reports matched-window neural vs train-only unigram
cross entropy in nats, perplexities, greedy and sampled continuation,
space fraction, repetition and unique-byte counts. A lower neural
loss than a basic unigram matters more than merely beating the
randomly initialized neural baseline.

Sampling makes output variable but cannot improve neural weights.
The current same-article splits still cannot prove independent
generalization; require multiple host-heldout articles for that.
No existing checkpoint, source archive or model registry is mutated
by `web-diagnose`; PyTorch remains an external neural dependency.

More detail: docs/WEB_DIAGNOSTICS_V114.md.

## Previous version — v1.13
# BERHOT EVOLVE v1.13 — Complete Corpus-to-Neural Learning Experiment

**One-shot command using the current archived Energy article**:

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve web-inventory --include-unknown-rights
python3 -m evolve web-lab --steps 80 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document
python3 -m evolve web-evaluate
python3 -m evolve web-predict --prompt 'Energy is' --new-bytes 100
python3 -m evolve model-audit
```

The single qualifying Energy article does NOT provide independent website
holdouts. This explicitly labeled same-article **DIAGNOSTIC** experiment
allows real weight updates now; it is not proof of generalization. Once
at least three distinct hosts with substantive articles have been archived,
run without `--diagnostic-within-document` for source-host heldout tests.

One-shot web-lab handles article curation (navigation excluded),
chunk dedup, rights metadata and explicit unknown-rights research opt-in,
SHA256 frozen datasets, actual PyTorch gradient descent, validation-only
checkpoint selection, heldout test byte loss, arithmetic language-loss
transfer diagnostic, SHA256 checkpoint persistence, and weight-only
inference through web-predict. Existing general and arithmetic checkpoints
remain untouched. No network access is needed after collection.

**Tests do not establish genuine web knowledge.** A reduced byte loss
only measures improved next-byte predictions on those samples. The new
tiny model is initialized independently of the arithmetic model, so
the arithmetic proxy is not a true continual-learning forgetting test.
Training uses PyTorch; BERHOT has not yet built a fully native tensor engine.

See docs/WEB_LAB_V113.md for the complete experimental design and safeguards.

## Previous version — v1.12.1
# BERHOT EVOLVE v1.12.1 — Correct Audit & Isolated Corpus Crawl

## Fixes from actual Mac diagnostic output

- A general language checkpoint with `promoted=1` was incorrectly
  reported as `accepted_by_validation=false` because the audit read
  an absent `accepted` field. `model-audit` now explicitly reports
  `promoted: true` from the original general registry and uses
  `accepted_by_validation: null` when that particular decision
  was never recorded. SHA256 verification remains unchanged.
- `corpus-crawl` used to generate research questions from an older
  pending-concept backlog, even when zero pages were fetched due to
  a spent UTC daily quota. It now runs in corpus-only mode with
  `questions_created: 0` and
  `question_review_mode: disabled_for_corpus_only`.
  Normal discovery and autonomous research cycles retain their
  existing backlog-review behavior.
- No model checkpoint or existing corpus/SQLite data is erased.

## Mac commands

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve model-audit
python3 -m evolve research-health
python3 -m evolve corpus-crawl --limit 5
python3 -m evolve corpus-status
```

The corpus-crawl command still respects the shared UTC daily budget
and fetches zero pages until quota is available.
The example.org and example.net domains in earlier examples are
reserved illustrative domains, not real educational datasets.
Queue real public HTTPS pages whose content you wish to study.
Fetching, collecting, and training are separate actions.

## Previous version — v1.12
# BERHOT EVOLVE v1.12 — Broad Public Web Corpus + Honest Model Audit

## Major correction: collection is not neural training

The old public explorer could read many HTTPS sites but usually stored
only metadata without recognized CC0/public-domain declarations. This
made the research DB active while preserving little reusable text.

v1.12 adds a SEPARATE public HTML corpus that can collect any site
accessible under HTTPS, public DNS, robots.txt, size/rate bounds and
normal request rules. **The collection/archive phase does not require
a CC0/public-domain license.** It records absent license as UNKNOWN
instead of rejecting text solely for that reason. Decoded HTML snapshots and cleaned
Unicode are saved under storage/web_corpus/; content-addressed SHA256
deduplicates exact normalized copies and stores source metadata and
publisher no-AI signals. The collector uses only Python standard library.

**This is not universal browser access**, an auth/paywall bypass,
semantic dedup, or permission to redistribute/training on any material.
Collected web corpus is NOT silently inserted into existing neural
training; a stored page is NOT learned knowledge.

### Commands

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Read the ACTUAL local checkpoint, data and dependency audit first:
python3 -m evolve model-audit

# Old research observations were metadata only. New sites get full archive:
python3 -m evolve corpus-seed --urls https://example.org/ https://example.net/

# Check shared UTC request budget; do not bypass rate limits:
python3 -m evolve research-health

# When remaining requests are available, crawl up to five queued pages:
python3 -m evolve corpus-crawl --limit 5

# Shows unique cleaned text count, rights metadata, raw source counts:
python3 -m evolve corpus-status

# Optional: run existing bounded automated cycles while process stays up:
python3 -m evolve daemon
```

Multiple URLs may be queued in one command; cross-domain follow is on
by default for corpus-seed and may be disabled with --same-host.
Each website is still subject to its own robots.txt and access policies.

**Important:** 96 arithmetic documents and 21 research observations
do not imply 96 new facts or 21 learned websites. If concept-inventory
reports zero eligible text definitions, the concept model has not learned
any from the local training set. Use model-audit to see general and
specialized checkpoints rather than assuming they ran.

The original MIT LICENSE, PyTorch neural dependencies, existing research
verifier and neural checkpoint registry remain unchanged.
See docs/OWNERSHIP_AND_DEPENDENCIES.md for the roadmap to a BERHOT-owned
native tensor/optimizer/Transformer implementation.

## Previous version — v1.11
# BERHOT EVOLVE v1.11 — Targeted Research & Offline Source Planning

## Why this change

Your Mac reported 10 open supported definition questions and 20 unsupported
questions, with 0 verified text training documents. Offline source planning
showed two candidate publisher groups for every concept. However, only
**energy** had already yielded a definition from one publisher in a previous
run; the remaining candidates have not been corroborated or rights checked.
Research queue order would spend limited requests on English before energy.

## Exact-concept research, without bypassing safeguards

````bash
git pull origin main
python3 -m pip install -e '.[train]'

# No network calls, no quota usage and no DB writes:
python3 -m evolve research-plan --concept energy
python3 -m evolve research-health

# Only when health reports remaining quota and cooldown eligibility:
python3 -m evolve research --concept energy --limit 1
python3 -m evolve concept-inventory
````

`research-plan` prints real candidate URLs in the existing queue, publisher
group hints, which URLs fall inside the configured per-question window,
and whether each URL was previously observed. It **does not download** any
page, verify rights, or claim independent corroboration.

`research --concept energy` performs the existing bounded research process
only on an open, cooldown-eligible `What is energy?` question. It does not
prioritize or alter other questions, expand the total request budget, change
robots restrictions, or loosen the requirement for two independently
agreeing sources and at least one recognized CC0/public-domain page.

Since the shared UTC daily quota was 60/60 on November 4, 2026, the next
automatic reset was 2026-11-05T00:00:00Z (03:00 Saudi time). The default
retry cooldown separately applied until about 20:52 Saudi time on November 4.
Do not retry while blocked.

## Honest experimental boundary

Two candidate publisher groups are not two verified definitions. The
per-page license, text extraction, agreement and actual training admission
must still succeed. Even if one definition is verified, v1.9 needs at
least two admitted *train-split* definitions before training.

## Previous version — v1.10
# BERHOT EVOLVE v1.10 — Research Budget & Cooldown Diagnostics

## Why your Mac returned zero attempts

User's evidence revealed 96 arithmetic documents, 21 research observations,
30 open questions (10 definition format, 20 currently unsupported),
zero verified text answers, and a fully consumed 60/60 UTC-day crawl budget.
After a prior research run, all 10 eligible definition questions were
also on the default 3600-second retry cooldown. Running `research --limit 3`
returned 0 attempts, 0 requests, 0 new eligible definitions.

**v1.10 does not fabricate verified facts, force a retry, increase the
crawl budget, or weaken reusable-rights requirements.** Instead it fixes
the queue's misleading behavior and makes the operational limits visible.

### New read-only command

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve research-health
```

`research-health` shows: current UTC quota and next automatic UTC reset,
supported versus unsupported open questions, how many are currently
blocked by cooldown, next eligible retry time, major source hosts,
and number of rights-admitted definition training documents. Its
`source_qualification_preview` enumerates candidate URL counts and
distinct publisher groups for each supported concept, WITHOUT fetching
pages. Candidate URLs are not validated sources, and their reuse rights
remain unknown until the normal per-page checks succeed.

### Safer research behavior

With exhausted quota, `research` now returns
`status: blocked, stop_reason: daily_budget_exhausted` without marking
more questions as attempted or extending their cooldown. Once a
nonempty quota remains, but all supported questions are cooling down,
it reports `stop_reason: retry_cooldown_active` without extra requests.
If the quota runs out partway through a batch, later unattempted
questions are left untouched and the report counts them separately.
The existing stricter per-page definition and two-publisher verification
pipeline, explicit CC0/PUBLIC-DOMAIN page rights declarations,
source security, and training admission remain unchanged.

When `research-health` reports `ready` and remaining quota, use a small
batch and inspect the actual findings:

```bash
python3 -m evolve research --limit 3
python3 -m evolve research-health
python3 -m evolve concept-inventory
```

A queue being ready is NOT evidence of verified text data. v1.9 neural
text training remains blocked until two distinct approved
training-split definition records have been independently verified.

## Previous version — v1.9
# BERHOT EVOLVE v1.9 — Verified Text → Neural Recall

## Why move beyond arithmetic?

The user's v1.8 Mac study showed Adam-trained heads with much better
500–999 extrapolation than plain SGD, but neither result showed natural
language understanding. v1.9 tests one limited step toward the project
goal: acquire concepts from **eligible, independently corroborated text**
and answer held-out question phrasings from neural weights alone.

## What v1.9 actually does

The existing research pipeline first persists `verified_answers` with
two independent publisher groups and explicit CC0/public-domain evidence,
plus one `verified:definition` training document. v1.9 reads ONLY verified,
rights-admitted, `split='train'` definition documents that exactly match
their persisted proof record. All research-only pages, unlicensed answers,
non-definition documents, and documents reserved in the original
global `val` split are excluded. v1.9 never crawls websites implicitly.

A separate tiny **byte-level causal Transformer** learns to produce
definition text from supervised question-answer sequences. Its loss
covers **answer bytes only**; question bytes are masked. It uses three
training phrasings per known concept and a separate validation phrasing
to select checkpoint weights. The final two **held-out phrasings of
those same concepts** are evaluated only AFTER selecting weights.
It reports before/after neural raw completions, exact-match accuracy,
token-overlap F1, per-concept breakdown, incorrect cross-concept answers,
and a never-trained unknown-concept negative-control prompt.

Most importantly, `concept-predict` reads ONLY its SHA256-verified saved
neural checkpoint. It does not search verified answers, documents,
skills or memory, and it never replaces its raw completion with a known
definition. Poor or zero exact accuracy must remain visible.

This is **same-concept prompt-paraphrase testing**, not
generalization to entirely new facts. Synthetic local CI sources verify
the admission workflow but are NOT falsely marketed as real public
research, and are never uploaded into the user's persistent database.

### Run on your Mac

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# First verify that your own existing DB contains eligible definitions:
python3 -m evolve concept-inventory

# If at least TWO train-split concepts are available, train:
python3 -m evolve concept-lab --steps 300 --seed 31415 --max-concepts 8

# Generate the answer from saved model weights, without retrieval:
python3 -m evolve concept-predict --question 'Describe volcano.'

# Re-evaluate saved model after process restart (read-only):
python3 -m evolve concept-evaluate

# Or inspect unaccepted candidate with honest labeling:
python3 -m evolve concept-evaluate --candidate
python3 -m unittest discover -s tests -v
```

For a concise report that avoids copying the full output:

```bash
python3 -m evolve concept-lab --steps 300 | python3 -c '
import json,sys
r=json.load(sys.stdin)
print("Status:",r["status"],"Reason:",r.get("reason"))
if r["status"]=="trained":
 print("Concepts:",r["concepts"])
 print("Accepted:",r["accepted"],"Steps:",r["actual_steps"])
 print("Val loss:",r["validation_answer_loss_before"],"->",r["validation_answer_loss_after"])
 print("Heldout before:",r["before_heldout"]["exact_accuracy"])
 print("Heldout after:",r["after_heldout"]["exact_accuracy"])
 for x in r["after_heldout"]["examples"]:
  print(x["question"],"=>",repr(x["raw_completion"]),"correct:",x["exact_match"])
'
```

If `concept-inventory` says `insufficient_verified_facts`, **do not
fabricate a training set**. Use the existing `explore --seed-url` and
`research` commands to gather independent, reusable source claims.
A minimum of two admitted `train`-split definitions is required; because
the existing store reserves its first (and each fifth) document for
validation, this can require three or more verified definition records.
Any rights/verification failures must remain research-only.

Checkpoints are stored separately under `storage/models/concept_lab/`,
and accepted concept candidates are **not promoted** into the general
language-model registry. If additional eligible definitions arrive,
the next run can warm-start from the last accepted concept checkpoint.
No original models or user evidence are overwritten.

**Limitations:** Two agreeing source groups can still be wrong. Passing
a byte-level loss threshold is not equivalent to correct factual recall.
The model may overfit and may hallucinate on unseen concepts; we must
inspect exact raw held-out answers and not just training loss.

## Previous version — v1.8
# BERHOT EVOLVE v1.8 — Gradient-Only Head Training (Adam vs SGD)

## Grounding in real Mac results

Your v1.7 head-calibration study reported **100%** across three operations
and every listed OOD range, through operands 500..999, for all three seeds.
The original saved v1.5 checkpoints scored poorly on large magnitudes.
That showed the original bilinear model was expressive enough, given precise
supervised head fitting; it did not show spontaneous arithmetic discovery.

## Controlled v1.8 question

Can ordinary **gradient descent alone** achieve the precision of v1.7's
analytic head calibration with the same saved v1.5 hidden features?

Both variants start from the same **original v1.5 output-head weights**:

- **Adam** uses first/second gradient moments, default learning rate 0.01.
- **SGD** uses plain gradients, default learning rate 0.05, no momentum.

The original v1.5 hidden projections remain strictly frozen. The module
runs on CPU in **float64** for numerical precision; only the existing
output-head parameters receive gradients. No polynomial basis vector,
ideal coefficients, or least-squares solution is used for optimization.
Both train only on the original **35 unordered 0–9 training pair groups**
with all three supervised operations. Disjoint validation groups select
the best optimizer step and accepted/rejected status; test/OOD scores and
the read-only v1.7 reference are accessed **only afterward**.

### Mac commands

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Train Adam on frozen features from the saved v1.5 seed-31415 model:
python3 -m evolve gradient-lab --seed 31415 --optimizer adam --steps 2000 --count 20

# Train SGD from the same saved original v1.5 output head:
python3 -m evolve gradient-lab --seed 31415 --optimizer sgd --steps 2000 --count 20

# Report every seed and both optimizers (short summary):
python3 -m evolve gradient-study --seeds 31415 42 1001 --optimizers adam sgd --steps 2000 --count 20

# Predict with the saved Adam candidate; even rejected models are labeled:
python3 -m evolve gradient-predict --a 999 --b 998 --operator '*' --optimizer adam --seed 31415

python3 -m unittest discover -s tests -v
```

The compact `gradient-study` result includes validation MSE, whether each
checkpoint was accepted, in-range exact match, the hardest-range
500–999 exact match, and per-operator means for all OOD groups.
It does **not** select one lucky optimizer or seed by test accuracy.

Stored checkpoints are isolated under `storage/models/gradient_lab/` and
include the original v1.5 source SHA256. Existing v1.5, v1.7, JSM-style
language model, and website research state are unchanged.

**Scientific boundary:** This is supervised fine-tuning of a fixed
head on learned bilinear features. The multiplication-capable architecture
and integer magnitudes were supplied. Even perfect 500–999 scores would
not show general reasoning or independent web-knowledge acquisition.
The repeated OOD problems are development diagnostics, not untouched
confirmatory benchmarks.

## Previous version — v1.7
# BERHOT EVOLVE v1.7 — Frozen Bilinear Feature Calibration

## What was established by the user's v1.6 Mac runs

The degree-2 polynomial reference scored 100% exact-match accuracy for
addition, subtraction and multiplication on all reported held-out groups,
including operands 500..999, for all three saved seeds (31415, 42, 1001).
In contrast, the earlier v1.5 bilinear network failed many large-number
tests. This proves the *provided basis* is expressive and fitted precisely,
not that a model autonomously discovered arithmetic.

## Controlled v1.7 question

Can we repair the ORIGINAL v1.5 model by keeping both learned bilinear
hidden operand projections completely frozen, and refitting ONLY its
operator-conditioned output head to the original training data?

Yes, this is a separate proposed experiment. A newly trained v1.7 candidate
is never silently substituted for your saved v1.5 model.

The procedure is precise:

1. Verify the original SHA-256 fingerprint of the saved v1.5 checkpoint.
2. Copy its left and right hidden projections into the exact same architecture
   in **float64** rather than the old float32 arithmetic.
3. Freeze hidden projections; collect their learned bilinear activations
   from the original 35 unordered training pair groups, including both
   operand orientations and all three supervised operations.
4. Fit each output head using float64 least squares on **training labels**.
   No explicit [1,x,y,xy] basis or target mathematical coefficients are
   supplied. The original v1.5 interaction prior is retained.
5. Select accepted/rejected based only on disjoint validation MAE.
6. AFTER that, measure held-out and OOD accuracy, side by side with the
   unchanged v1.5 checkpoint on the same problems.

### Run on Mac (zsh)

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Calibrate the original saved v1.5 model for one seed:
python3 -m evolve calibration-lab --seed 31415 --count 20

# Predict from new saved calibration without touching original weights:
python3 -m evolve calibration-predict --a 999 --b 998 --operator '*' --seed 31415

# All three original seeds, report failures as well as successes:
python3 -m evolve calibration-study --seeds 31415 42 1001 --count 20

python3 -m unittest discover -s tests -v
```

Each trial stores a separately SHA-256-verified checkpoint under
`storage/models/calibration_lab/` and tracks its v1.5 source fingerprint.
No neural gradient training occurs in this step. All training examples
remain from 0..9, and test/OOD labels are consulted only after fitting
and validation-based acceptance are decided.

**Caveat:** The original bilinear interaction already contains a strong
multiplication prior. An accurate calibrated model would demonstrate that
better head fitting and float64 precision fix v1.5's parameter error in
this finite supervised task, NOT that the general EVOLVE language model
can reason, learn from the web, or invent arithmetic.

## Previous version — v1.6
# BERHOT EVOLVE v1.6 — Audit Learned Coefficients vs Polynomial Basis

## Actual Mac result from v1.5

The learned-bilinear network achieved 100% on in-range and 10–19 pairs
for all three operations, but the 50–99 three-seed mean accuracy fell to
**0% addition, 64.0% subtraction, and 26.3% multiplication**.
For seed 31415, it predicted **27 × 28 = 755**, not 756.

The v1.5 network has a very useful mathematical property: because its
hidden interaction multiplies *affine functions* of both operands, the
entire neural model is exactly a degree-2 expression:

    c0 + c1*(a/9) + c2*(b/9) + c3*(a/9)*(b/9)

That function class can represent +, - and * perfectly. Thus its larger
number failures can result from **small learned coefficient errors**
magnified at large inputs, not lack of representational capacity.

## Two controlled tools

1. `basis-audit` reads your existing v1.5 checkpoint without retraining.
   It algebraically extracts the four effective coefficients per operation,
   compares them to exact mathematical reference coefficients, and breaks
   down how each coefficient contributes to a sample prediction error.
   Reference coefficients are part of the *audit*, never injected into
   neural weights.
2. `basis-lab` independently fits a trainable linear neural output layer
   using the explicit [1,x,y,x*y] **polynomial feature basis** and
   **double-precision least squares** on ONLY the 35 train groups.
   This is supervised parameter fitting, NOT neural gradient descent,
   and the polynomial basis explicitly contains a multiplication feature.
   It is an idealized numerical reference, not emergent reasoning.

Validation is disjoint; the test and OOD evaluation are performed only
after fitting. Along with the old ranges, two further groups of operands
100–199 and 500–999 are included to expose tiny coefficient errors.
Saved v1.5 checkpoints are evaluated read-only on exactly matching probes.

### Mac commands

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# First diagnose v1.5's original learned coefficients (does not train):
python3 -m evolve basis-audit --seed 31415 --a 27 --b 28

# Fit an explicit polynomial-basis baseline from only 0–9 examples:
python3 -m evolve basis-lab --seed 31415 --count 20

# Compare the new model and frozen v1.5 across every predeclared seed:
python3 -m evolve basis-study --seeds 31415 42 1001 --count 20

# Read an actual prediction from the saved fitted reference:
python3 -m evolve basis-predict --a 999 --b 998 --operator '*' --seed 31415

python3 -m unittest discover -s tests -v
```

New checkpoints are saved independently under `storage/models/basis_lab/`
with SHA256 integrity checks. No previous model weights, training rights
or web observations are changed.

**Interpretation:** A highly accurate polynomial reference demonstrates
that the correct rule fits within the architecture and that a stable fit
can recover its coefficients from supervised examples. It does not prove
that EVOLVE's general language model learned multiplication, invented
the basis autonomously, or extrapolated with a neural learning algorithm.
Repeated OOD probes are development diagnostics, not pristine benchmarks.

## Previous version — v1.5
# BERHOT EVOLVE v1.5 — Learned Bilinear Interactions

## What the Mac's v1.4 results show

The shared smooth-MLP arithmetic model learned most held-out 0–9 combinations
of addition, subtraction and multiplication, but generalization to unseen
magnitudes was unequal. Across three seeds, average exact-match OOD results
for **both 20–29 operands** were **0% addition, 50.9% subtraction, 0%
multiplication**. The single prediction for 27*28 was 538 instead of 756.
All 73 local tests passed. Those tests validate the software, not reasoning.

## Controlled v1.5 interaction experiment

A new shared model receives ordered numeric magnitudes, learns separate
projections of the left and right operand, and **multiplies their hidden
activations elementwise** before a learned three-output head. This is a
**strong bilinear inductive bias**: it puts product-like relationships into
the model's hypothesis class. The coefficients and operation responses
are learned from supervised examples; no arithmetic answer is substituted
during inference. It must NOT be called autonomous discovery of multiplication.

It uses the exact same 35/10/10 unordered-pair training/validation/test split
for each seed as v1.4, with forward/reverse operand orientations kept together.
All three operations (+,-,*) share one neural model. Checkpoints are chosen
using validation only. Frozen v1.4 models are re-evaluated, read-only,
only where the saved seed/split match. New independent development probe
ranges **both 30–49** and **both 50–99** are added alongside the old
10–29 probes; the network does not train on any of these OOD problems.

### Run on Mac

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve interaction-lab --seed 31415 --steps 800 --rank 8 --count 20
python3 -m evolve interaction-predict --a 27 --b 28 --operator '*' --seed 31415
python3 -m evolve interaction-predict --a 27 --b 28 --operator '+' --seed 31415
python3 -m evolve interaction-predict --a 27 --b 28 --operator '-' --seed 31415
python3 -m evolve interaction-study --seeds 31415 42 1001 --steps 800 --rank 8 --count 20
python3 -m unittest discover -s tests -v
```

The study prints in-range and OOD accuracy separately by operation and
every seed. All candidate weights are stored under
`storage/models/interaction_lab/`, SHA-verified and labeled accepted or
unaccepted based on validation improvement. Your v1.4 and v1.3 checkpoint
files, neural language model, web observations, and source-rights policies
remain untouched.

**Scientific boundary:** If multiplication improves beyond 0–9, it would
demonstrate that supplying a bilinear interaction allows this small supervised
model to represent products. It would not demonstrate emergent universal
arithmetic, carrying algorithms, or independently validated web knowledge.
Repeated probes and shared pairs across seeds are not independent
confirmatory benchmarks. An entirely new held-out task is still needed.

## Previous version — v1.4
# BERHOT EVOLVE v1.4 — Shared Three-Operation Neural Study

## From the v1.3 Mac results

The non-saturating linear specialist achieved **100% on all 180 scored OOD
addition examples** across seeds 31415, 42, and 1001. The original tanh MLP
failed most extrapolation tests. The v1.3 coefficients correspond to an
addition rule because that rule belongs to the supplied linear function class.
Those were repeated problem sets across seeds, not 180 independent problems.

## The new experiment

One small supervised neural network now learns **addition, subtraction,
and multiplication**. It uses ordered numeric operands, a one-hot operation
indicator and a shared MLP with three learned output heads. It does not
call Python's calculator in the forward pass.

Important supplied priors: familiar magnitudes, three known operations,
operation-specific output scales, and smooth MLP behavior. Each operator
uses the **same operand-pair groups**: 35 train, 10 validation, and 10 test
unordered pairs. For nonidentical operands, both directions are kept in
the same split, including subtraction where order changes the answer.
Checkpoints are chosen using validation MSE; in-range test and out-of-range
questions are evaluated only after selection. All three operations are
reported separately, including poor or zero accuracy.

Three OOD groups are unchanged from v1.2/v1.3: one operand 10–19, both
10–19, or both 20–29. The v1.3 linear specialist is compared **only for
addition**, because it was never trained on the other operations.

### Run on Mac

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve multiop-lab --seed 31415 --steps 800 --count 20
python3 -m evolve multiop-predict --a 27 --b 28 --operator '+' --seed 31415
python3 -m evolve multiop-predict --a 27 --b 28 --operator '-' --seed 31415
python3 -m evolve multiop-predict --a 27 --b 28 --operator '*' --seed 31415
python3 -m evolve multiop-study --seeds 31415 42 1001 --steps 800 --count 20
python3 -m unittest discover -s tests -v
```

Each trial stores an isolated SHA-verified candidate at
`storage/models/multiop_lab/`. When held-out validation does not improve,
the model remains unaccepted but is available for labeled inspection.
Old web observations, language-model checkpoints, arithmetic specialists
and rights/training eligibility controls are unchanged.

**Scientific boundary:** Good performance for these three explicitly
taught operations does not establish general numerical reasoning,
compositional carry handling, or learning concepts autonomously from the web.
Earlier extrapolation problems have now been used repeatedly as
development diagnostics, not untouched confirmatory benchmarks.

## Previous version — v1.3
# BERHOT EVOLVE v1.3 — Learned Linear Extrapolation Baseline

## The actual Mac result

Your v1.2 frozen MLPs averaged **15%** exact accuracy when one operand
was outside 0..9, and **0%** when both operands were in 10..19 or 20..29.
Across three saved seeds, 9/180 scored OOD problems were correct (the
probe pairs are shared across seeds). A seed-31415 model predicted about
30 for 27+28=55, consistent with saturation of the tanh network.

## The controlled v1.3 experiment

A new **trainable linear layer** maps two normalized numerical operands to
a real-valued output. It has no hardcoded addition coefficients, but the
LINEAR function class and continuously ordered number magnitudes are strong
explicit priors. It is designed to test whether non-saturating structure
alone explains the difference—not whether EVOLVE has discovered mathematics.

The model receives exactly the original 35 training pairs (0..9), has 10
validation pairs for checkpoint selection, and 10 single-digit test pairs.
Only after training/selection do we evaluate the same three OOD groups as
v1.2 (one 10..19 operand, both 10..19, both 20..29). The already-saved
tanh MLP is loaded and scored read-only on exactly matching questions.

### Run on Mac

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve linear-lab --seed 31415 --steps 500 --count 20
python3 -m evolve linear-predict --a 27 --b 28 --seed 31415
python3 -m evolve linear-study --seeds 31415 42 1001 --steps 500 --count 20
python3 -m unittest discover -s tests -v
```

Each run includes the **learned coefficients and bias**, single-digit
validation/test accuracy, exact-match OOD scores, signed error, and a
frozen-checkpoint tanh comparator with source labels. No existing model
or web-training data is overwritten. The new specialist is stored under
`storage/models/linear_lab/`, with per-seed checkpoint hashes.

**Limits:** A learned linear layer will readily represent addition because
linearity is built into its hypothesis class. Accurate extrapolation is
evidence for that inductive bias, not emergent general reasoning. In
particular this test does not teach carry handling, multilingual knowledge,
self-directed learning, or arbitrary mathematical operations.

## Previous version — v1.2
# BERHOT EVOLVE v1.2 — Frozen Model Extrapolation Audit

## Why v1.2?

The actual v1.1 Mac experiment achieved **10/10** on withheld digit
combinations using seed 31415. Across seeds **31415, 42, 1001** the
model scored **10/10, 10/10, and 8/10**, averaging **93.33%**.
All 56 local unit tests passed.

That tests interpolation with known operands (0..9). It does **not**
demonstrate the ability to add new magnitudes or two-digit numbers.

v1.2 audits **unchanged, already trained v1.1 checkpoints** on:

- `one_new_operand`: one number 0..9 and one number 10..19
- `both_10_to_19`: both numbers 10..19
- `both_20_to_29`: both numbers 20..29

These probes never appear in the 0..9 training or validation data.
Each reports exact-match accuracy after rounding the neural output,
within-one accuracy, mean absolute error, signed bias and example failures.

### Run on your Mac (no GPU retraining required)

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Compare the actual saved checkpoints for all three seeds:
python3 -m evolve magnitude-extrapolation-study --model-seeds 31415 42 1001 --count 20

# Inspect all predictions and mistakes for one checkpoint:
python3 -m evolve magnitude-extrapolate --model-seed 31415 --count 20
```

The audit uses saved SHA256-verifiable registered checkpoints when possible.
Earlier v1.1 per-seed files are recovered from `storage/models/magnitude_lab/`
and explicitly labeled `local_archive_hash_not_registry_verified`; their
old acceptance status may not be recoverable. Future training runs register
each model seed separately for stronger provenance.

Importantly, **no weights are updated or checkpoints promoted** by these
commands. Out-of-distribution answers are scored only after the existing
model has already been selected. Repeated audits of the same numbers
are development diagnostics, not a fresh confirmatory test.

A model may show high within-0..9 accuracy and fail these probes because
numeric order and magnitude are input priors, while extrapolation beyond
the normalized training range remains an unsolved challenge.

## Previous version — v1.1
# BERHOT EVOLVE v1.1 — Neural magnitude ablation

## Why this experiment?

On the Mac, v0.9's explicit-addition model scored 10/10 unseen pairs,
while v1.0's generic categorical digit-pair classifier memorized 35/35 training
pairs but scored 0/10 on validation and 0/10 on the test set.

v1.1 tests a middle ground: a generic MLP that receives the **numeric
magnitudes** of both operands. No summation is hard-coded in its forward pass.
Normalized ordinal inputs, the continuous regression objective, and
commutative pair canonicalization are strong priors—not autonomous reasoning.

### Run on macOS (zsh)

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve magnitude-lab --seed 31415 --steps 800
python3 -m evolve magnitude-predict --a 9 --b 8
python3 -m evolve magnitude-study --seeds 31415 42 1001 --steps 800
python3 -m unittest discover -s tests -v
```

Each seed uses 35 training, 10 validation and 10 unseen test unordered
digit pairs. Matching-seed v0.9 and v1.0 baselines are compared read-only.
The study reports **every** seed, not just the best. The last study seed
becomes the latest magnitude candidate; run `magnitude-lab --seed 31415`
again to restore that seed for subsequent prediction.

Models save separately to `storage/models/magnitude_lab/` and never overwrite
prior checkpoints or research observations. These supervised finite-digit
tests do not establish general arithmetic, model self-improvement or
real-world web knowledge.

## Previous version — v1.0
# BERHOT EVOLVE v1.0 — Operator-Blind Neural Arithmetic Comparison

## What the previous experiment really demonstrated

The v0.9 experiment achieved **10/10 correct** on its ten withheld
single-digit addition pairs and **47/47 unit tests** on a Mac with Apple MPS.
The learned digit parameters closely matched the numerical values 0..9.

**Important:** v0.9 explicitly combined two learned scalar representations
by addition; this was an architectural inductive bias, not spontaneous
discovery of arithmetic. It was also supervised on Python-checked labels
rather than knowledge taken from web exploration.

## New in v1.0 — Can a neural network learn without the explicit sum?

The new \`rule-lab\` experiment receives the same digit tokens, but its forward
pass has **no numeric summation**:

\`\`\`
digit a, digit b -> learned embeddings -> generic MLP -> 19 answer classes
\`\`\`

The network is supervised on known examples. The input domain (0..9), output
range (0..18), and **commutative pair canonicalization** are still supplied.
These assumptions make the experiment easier; it is not a general mathematics
model or a self-improving foundation model.

Both experiments use 55 unordered pairs. The split for each seed is:

- **35 pairs** for training.
- **10 pairs** for validation/checkpoint selection.
- **10 pairs** reserved for final testing, never used in training or selection.

Reversed operand pairs are in the same group, preventing leakage. The
black-box model's test accuracy is **measured, not assumed**. If its
validation cross-entropy does not improve, its checkpoint is **not promoted**.
Rejected candidates are nevertheless saved for inspection. Use
`rule-predict --candidate` to inspect them, and note the returned
`checkpoint_status: unaccepted_candidate` instead of confusing a failed
experiment with an accepted neural model; even a low score
is scientifically useful and highlights the value of structural priors.

### Fair head-to-head on your Mac

Run from the existing Git-cloned project with your \`.venv\` active:

\`\`\`bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve numeric-lab --steps 400 --seed 31415
python3 -m evolve rule-lab --steps 800 --seed 31415
python3 -m evolve rule-predict --a 9 --b 8 --candidate
python3 -m unittest discover -s tests -v
\`\`\`

Use the **same seed** in both commands. The rule-lab output contains
\`explicit_additive_baseline\` only if a matching-seed v0.9 model exists.
The new classifier's \`test_examples\` include the true number, predicted
number, correctness and predicted-class confidence. The v0.9 checkpoint
remains untouched. To see how split selection affects results, repeat
both experiments on another seed, for example \`--seed 1001\`; do not
select the best-performing seed using the final test score.

### Interpretation guide

- High v0.9 accuracy, lower rule-lab accuracy: the explicit addition prior
  is doing important work. The digit-pair classifier has not generalized as
  well from sparse labels.
- High accuracy in both: the generic network learned the 0..9 addition
  mapping on this finite domain. It **does not** prove extrapolation to
  multi-digit arithmetic or new operations.
- Low accuracy in both: inspect the train/validation loss and split,
  model size, optimization and number of *independent* training pairs.

All checkpoints are isolated in \`storage/models/rule_lab/\`,
\`storage/models/numeric_lab/\` or existing model folders. No research-only
webpage text is added to gradient training. Web concepts, independent
evidence gathering and knowledge verification remain a separate,
unfinished research track.

## v0.9 — Overfitting Protection and Structured Numeric Learning

## What the real macOS results showed

- Mixed arithmetic byte-based specialist: 400 training steps, training answer loss
  **2.6185 → 0.0065**, but held-out answer loss **2.5784 → 3.3191**
  and **0/20** correct on unseen math; **17/20** outputs were numeric.
- Single-digit addition byte-based specialist: held-out loss **2.3033 → 1.9399**,
  **0/10** correct even within the training operand range, and **0/10** on
  two-digit extrapolation.
- This is **overfitting** and a failure to generalize; formatting improvement
  should not be confused with real arithmetic.

## v0.9 changes

1. **Safer checkpoint acceptance:** every new math specialist, including the
   **first**, must lower held-out answer loss relative to its pre-training
   baseline or previous accepted checkpoint. An initial checkpoint with
   held-out loss rising from 2.58 to 3.32 now fails promotion.
   The trainer checks held-out loss throughout optimization, retains the
   best validation step, and stops early after repeated non-improvement.
   Existing old checkpoints are not deleted or silently changed.
2. **Separate structured numerical experiment:** 
   \`numeric-lab\` trains ten **learned scalar digit embeddings** with
   an explicitly additive composition layer. This strong, task-specific
   architectural assumption makes the relation easier to learn than from
   byte-level text alone. It is not a general language model or an
   autonomously discovered algorithm.
3. **Strict evaluation split:** 55 unordered pairs from 0..9 are divided
   into **35 training**, **10 validation**, and **10 unseen test** groups.
   Reverse operand order never leaks between them. Test labels are used
   only for final scoring, not for gradient updates or selecting a model.
4. **Isolated checkpoint:** model and split metadata are saved under
   \`storage/models/numeric_lab/\` and indexed separately. This does not
   overwrite the existing general, mixed-math, or single-digit byte models.

### Run on macOS

\`\`\`bash
git pull origin main
python3 -m pip install -e '.[train]'
# A new isolated, fully neural addition experiment:
python3 -m evolve numeric-lab --steps 400 --seed 42
# Query its learned digit parameters (not the Python verifier):
python3 -m evolve numeric-predict --a 9 --b 8
# Previous byte models remain available for honest comparison:
python3 -m evolve assess --skill single-digit-add --distribution both --questions 10
python3 -m evolve status
\`\`\`

The structured model is expected to perform better **within single-digit addition**
because its architecture encodes the compositional form of addition. That
cannot establish general arithmetic reasoning or web-based learning. To
expand the genuine language model, we will need a numerically aware
representation, larger **independent** corpora, and rigorous transfer tests.
Do not merge these metrics with general byte-level validation loss.

## v0.8 — Answer-Focused Training & Honest Skill Evaluation

## Why v0.8 exists

The first real macOS neural run achieved **5.60 → 1.66 byte-level validation loss**, yet
scored **0/10** on unseen two-digit arithmetic. Every parsed answer was `null`.
This means training captured common byte patterns, not usable arithmetic. v0.8
addresses the mismatch between the training objective and the actual skill test;
it **does not** assume the result will be correct.

### New tools

- `assess` now displays each **raw generated completion** (`raw_completion`),
  `numeric_output_rate`, and exact-match accuracy. It supports both
  unseen **in-range** questions and harder **extrapolation** questions.
  Questions already in training **or** validation documents are excluded from probes.
- `math-finetune` adds a separate answer-only objective: ignore loss on repeated
  boilerplate, and optimize only the numeric answer and final newline.
  Math checkpoints are stored separately in `storage/models/math/`;
  the general model's checkpoints remain untouched.
- `practice --skill single-digit-add` creates a bounded curriculum of unique,
  Python-checked single-digit addition problems. Existing mixed arithmetic practice
  remains available by default.

**Important:** Answer-focused training can improve numeric formatting and teacher-forced
answer loss without improving calculation accuracy. Unseen exact-match accuracy is
the real signal. The general model and math specialist are still small neural
sequence predictors, not algorithmic solvers.

## macOS: diagnose your existing checkpoint and train a math answerer

From the cloned repository with your virtual environment active:

```bash
git pull origin main
python3 -m pip install -e '.[train]'
# Diagnose your v0.7 general checkpoint: raw text vs missing numeric output
python3 -m evolve assess --questions 10 --distribution both --model-source general
# Reuse your existing 32 independently verified mixed arithmetic documents
python3 -m evolve math-finetune --skill mixed --steps 400
# Compare against the same deterministic unseen problems
python3 -m evolve assess --questions 10 --distribution both --model-source math
```

To study a much narrower, measurable first skill:

```bash
python3 -m evolve practice --skill single-digit-add --limit 64
python3 -m evolve math-finetune --skill single-digit-add --steps 400
python3 -m evolve assess --skill single-digit-add --distribution both --questions 10
```

The `in-range` assessment excludes saved problems while sampling operands within the
practice range. `extrapolation` tests numbers outside it. Tests use greedy
decoding so comparisons are reproducible for a fixed checkpoint and seed.
`math-finetune --force` intentionally trains longer on the same corpus;
otherwise the minimum-new-data threshold prevents unnecessary repeat runs.
The checkpoint is only accepted for the math track if its held-out
answer-token loss improves over the previously accepted math model.
These metrics do not compare directly with the original **generic byte loss**.

## Changes in v0.7

This is an **experimental learning platform**, not a self-conscious agent or general autonomous intelligence. It introduces a way to **explore public HTTPS pages without requiring a declared training license at each page**, while **separating observations from training data**. The project still uses a randomly initialized, tiny byte-level PyTorch Transformer. Concept selection is heuristic, not yet intelligent self-directed research.

## Changes in v0.7

The observed macOS run found 21 websites and 51 topics, but no independently corroborated definitions suitable for training. The tests were correct; live evidence availability, not PyTorch, was the blocker.

- **Better research triage:** unsupported question types no longer consume the limited source-fetch slots. New discovery queues only `What is ...?` questions, which the current lexical verifier can attempt. Older unsupported questions remain in the database as open tasks.
- **Better evidence candidates:** looks for topic-related URLs already present in the crawler frontier and tries independent publisher-domain candidates before redundant same-site pages. Uses normal safe HTTPS/robots restrictions.
- **Visible uncertainty:** `research` and `learn` output per-question reasons such as `no_definition_found`, `single_publisher_only`, `definitions_do_not_agree` or `daily_budget_exhausted`.
- **Verified arithmetic practice:** `practice` generates independently checked mathematical QA examples from Python (no web content, no external license assumptions), giving the small PyTorch model real incremental training material while web corroboration remains unresolved.
- **Real skill benchmark:** `assess` tests the promoted model's *exact-match answers to unseen arithmetic questions* with operands outside the practice range. A low or zero score is a valid result; declining byte-level loss is not arithmetic understanding.
- **Mac warning fixed:** the optional `[train]` dependency now installs NumPy alongside PyTorch.

## Run v0.7 on macOS

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve research --limit 10
python3 -m evolve questions --limit 20
```

To make a first verified-only neural checkpoint without pretending Worldometer text is training-eligible:

```bash
python3 -m evolve learn --limit 10 --practice-arithmetic --practice-limit 32
python3 -m evolve assess --questions 10
python3 -m evolve status
```

For a second, distinct batch and incremental training round, repeat `learn --practice-arithmetic`. The generated arithmetic material is explicitly marked `verified:arithmetic`; the `research_questions_verified` count remains zero unless genuine two-source web corroboration succeeds.

Pure web-only learning remains available via `python3 -m evolve learn --limit 10`. For ongoing, optional internal arithmetic practice, set `practice.auto_enabled` to `true` in `config/settings.json`, or run `python3 -m evolve once --practice-arithmetic` per cycle. This arithmetic work is a separately engineered learning curriculum, **not a claim that the crawler spontaneously learned mathematics from webpages**.

Run all tests with `python3 -m unittest discover -s tests -v`. GitHub Actions also tests the neural training/assessment pipeline on CPU.

## Changes in v0.6

- Added an independent-source **research step**: supported `What is ...?` questions are checked against definitions on two distinct publisher-domain groups. Their URLs and content hashes remain in SQLite as an audit trail.
- Added answer checking for substantial lexical overlap, contradictory numbers and explicit negation. Other questions and disputed definitions remain open; no answer is fabricated.
- Added a reusable-source admission gate: only corroborated answers with at least one explicit CC0/public-domain supporting page can create compact verified training examples. Unlicensed research pages remain excluded from the gradient-training dataset.
- Added incremental **verified-only PyTorch training**, train/validation loss diagnostics, and a second training round once new verified examples arrive.
- Added `research`, `answers`, `learn`, and `train --verified-only`. The `once` / `daemon` workflow now runs bounded research before verified-only training when exploring public sites.
- Added offline regression tests and a GitHub Actions CPU-PyTorch integration job. Existing observations, concepts and training records are migrated in place.

**Verification boundary:** Matching definitions from two distinct websites is *source corroboration*, not a guarantee of factual truth or model comprehension. This initial verifier supports short English definitional sentences only; it cannot yet resolve disputed claims, time-sensitive facts, abstract questions or complex scientific propositions. Do not use its outputs as independently certified answers in high-stakes settings.

## Changes in v0.5

- EVOLVE now persists **open research questions** connected to each discovered topic and source URL.
- New `questions` command lists the question queue; `status` separates open questions from verified questions.
- The previous `reviewed` label was misleading: a generated question does not mean learned knowledge. Old `reviewed` concepts are requeued once, preserving observations and other data, so their questions become durable.
- No answers are fabricated and no research-only pages enter neural training. Future versions must gather and validate evidence before marking questions verified.
- The question generator is still a simple bootstrap heuristic, not autonomous factual reasoning.

## Changes in v0.4

- No user-selected focus word. EVOLVE extracts candidates from each visited page and ranks observed terms using its title, headings and repeated words.
- The old `--focus` option is accepted for compatibility but ignored; it cannot create unrelated learning topics.
- New `python -m evolve concepts --limit 20` command lists candidates and their source URLs.
- SQLite task-priority fields are migrated in place so previous data is preserved.

## Open-web exploration

- New **`explore`** command: paste a seed URL; no `--license` or `--rights-evidence-url` required for read-only discovery.
- Optional **`--cross-host`**: follow ordinary public HTTPS links to other sites within request, depth, and frontier budgets.
- Stores research observations as **URL, title, hash, character count, and candidate concepts**. It **does not** insert full unclear-rights pages into the neural training dataset.
- A page explicitly declaring CC0 or public-domain status through supported machine-readable links can still enter the training dataset automatically, unless its metadata signals `noai`.
- Keeps source references private and retained for deduplication, verification, corrections, and deletion requests.
- Retains the v0.2 training, arithmetic verification, persistent skills and checkpoints, and long-running daemon.
- The old `web-start` exists for specifically permitted training sources. A content URL can no longer be used as its own `AUTHORIZED-TRAINING` grant evidence.

## macOS (zsh) / Linux (bash) — setup

Open Terminal in the cloned repository folder. On macOS, use `python3` and `source` — **not** Windows commands such as `py -3` or `.\\.venv\\Scripts\\Activate.ps1`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[train]'
python3 -m evolve --help
```

If you already ran `python3 -m venv .venv`, **do not recreate it**: just run `source .venv/bin/activate` and continue with the install commands. An activated shell usually displays `(.venv)` at the start of the prompt.

When returning to the project in a new Terminal window:

```bash
cd /path/to/berhot_evolve
source .venv/bin/activate
```

Use the actual repository location in place of `/path/to/berhot_evolve`. To exit the environment, run `deactivate`.

## Windows PowerShell — setup

Open PowerShell in the cloned repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
python -m evolve --help
```

If activation is blocked, first run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in the same session, then retry activation. Do not use `source` on Windows PowerShell.

**Platform rule:** In the commands below, use `python3 -m evolve` on macOS/Linux and `python -m evolve` on Windows PowerShell. Neither platform needs a Wikimedia contact email for `explore`.

## Explore from one website — no API, no license parameter

The following URL is the user's example. The page is **not treated as licensed neural-training material**:

```bash
# macOS / Linux
python3 -m evolve explore --seed-url "https://www.worldometers.info/languages/english-alphabet/" --limit 8 --cross-host
python3 -m evolve concepts --limit 20
python3 -m evolve questions --limit 20
python3 -m evolve research --limit 5
python3 -m evolve answers --limit 20
python3 -m evolve status
python3 -m evolve discover --limit 8
```

```powershell
# Windows PowerShell
python -m evolve explore --seed-url "https://www.worldometers.info/languages/english-alphabet/" --limit 8 --cross-host
python -m evolve concepts --limit 20
python -m evolve questions --limit 20
python -m evolve research --limit 5
python -m evolve answers --limit 20
python -m evolve status
python -m evolve discover --limit 8
```

Any ordinary publicly accessible HTTPS HTML website may be attempted, provided automated access is permitted. If its `robots.txt`, server, or access policy blocks access, this crawler stops. **No stealth, captcha bypass, paywall bypass, logins, or evasion.** Some sites require APIs or prohibit crawling.

Continue scheduled observation and eligible training, if available:

```bash
# macOS / Linux
python3 -m evolve once
python3 -m evolve daemon
```

```powershell
# Windows PowerShell
python -m evolve once
python -m evolve daemon
```

Stop daemon using **Ctrl+C**. No GPU is necessary for crawling. Run `python -m evolve train` only if there are sufficient eligible training documents. If all pages are research-only, the model will not train from their full text; the system still records concepts and source pointers.

## Troubleshooting: one attempted URL, zero concepts

If `explore` reports `"attempted": 1`, `"errors": 1` and `"research_observed": 0`, then the seed fetch failed before any content was processed. This is **not** a sign that the model rejected the topic. The current response includes `error_details` with the URL and failure reason.

To inspect failures stored in the existing SQLite database **without deleting your discoveries**, run:

```bash
# macOS / Linux
python3 -m evolve errors --limit 10
python3 -m evolve status
```

```powershell
# Windows PowerShell
python -m evolve errors --limit 10
python -m evolve status
```

- `Website disallows automated access`: respect the website's policy and use a different permitted source.
- `Robots availability uncertain (HTTP 403)`: the crawler could not verify `robots.txt`; do not bypass that restriction.
- `Redirect refused`: the crawler deliberately rejects HTTP redirects rather than trusting an unknown destination.
- `HTTP 403`, `HTTP 429`, or a connection timeout: inspect access policy and rate limits; do not evade access controls.
- `Unable to resolve source host`: inspect local DNS/internet connectivity.
- `Too little educational text`: the server returned a page without enough extractable text.

Failed URLs remain recorded as errors; `discover` does not repeatedly hammer blocked pages. Register another permitted seed if necessary. A page being readable in a regular browser does not guarantee automated crawling is permitted.

## macOS SSL certificates — CERTIFICATE_VERIFY_FAILED

If discovery returns `Cannot verify robots.txt: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`, Python cannot validate the server's TLS certificate. Do **not** disable TLS verification.

1. Check which Python installation you're using:

   ```bash
   which python3
   python3 --version
   python3 -c 'import ssl; print(ssl.get_default_verify_paths())'
   ```

2. If you installed Python from python.org, open the version-matched `/Applications/Python 3.x/Install Certificates.command` included with the installer. Python's macOS documentation recommends this certificate-installation step.

3. Alternatively, for a session-specific CA bundle with certificate **verification still enabled**, install the maintained `certifi` bundle inside the active virtual environment:

   ```bash
   python3 -m pip install --upgrade certifi
   export SSL_CERT_FILE="$(python3 -m certifi)"
   python3 -c 'import urllib.request; print(urllib.request.urlopen("https://www.python.org/", timeout=10).status)'
   ```

   An HTTP status of 200 verifies that this Python session can establish a trusted HTTPS connection to python.org. If the original site still fails, check its certificate chain or a network TLS-inspection proxy. Do not assume the site is safe or try to bypass access controls. The environment setting lasts for the current shell session unless you configure it persistently.

4. Retry only the previously failed seed URL after fixing the trust store:

   ```bash
   python3 -m evolve errors --limit 10
   python3 -m evolve retry-url --url "https://www.worldometers.info/languages/english-alphabet/"
   python3 -m evolve discover --limit 8
   python3 -m evolve concepts --limit 20
   ```

   `retry-url` requeues one failed URL, retains its error history, and will **not** override an explicit robots.txt exclusion. If the same site remains blocked by `robots.txt`, HTTP restrictions, or redirects, use a different eligible source.

## Research and continual learning on macOS

From an activated virtual environment in your cloned project:

```bash
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve discover --limit 8
python3 -m evolve questions --limit 20
python3 -m evolve research --limit 5
python3 -m evolve answers --limit 20
python3 -m evolve train --verified-only
python3 -m evolve status
```

To combine question research and eligible-training into one bounded step:

```bash
python3 -m evolve learn --limit 10
```

The `once` and `daemon` controllers now also run bounded research and train from verified-only QA examples for research-only profiles. These commands **do not bypass robots.txt or site restrictions**. Evidence retrieval consumes the same daily request budget as crawling and unsuccessful questions have an attempt cooldown. Additional facts and independently accessible sources are needed to generate enough train/heldout documents. An output such as `Need >= 6 documents; have 0` is an expected, truthful training skip—not a bug.

`status` separately reports `research_questions_verified`, `verified_answers_training_eligible`, `verified_neural_documents`, and the latest promoted model. In `learn`, `training.status=trained` means PyTorch executed optimization; only the actual train/heldout loss and controlled evaluations can establish model progress. This tiny byte-level model is not a reliable general knowledge answerer.

Run unit tests:

```bash
python3 -m unittest discover -s tests -v
```

For implementation details, limitations and provenance rules, see [Evidence and Learning](docs/EVIDENCE_LEARNING.md).

## Protections against malicious sites

- Only public HTTPS names on port 443 (no IP-literal, local, or private network addresses); public DNS resolution screening is best-effort.
- Honors robots.txt (RFC 9309), with bounded per-cycle and daily requests, limited crawl depth and per-page links, per-request timeouts, and response size caps.
- No browser JavaScript, HTML active content, user cookies, attachments, program execution, or document macros. Only `text/html` is fetched and parsed as text.
- Redirects are refused, which also avoids unexpected destination changes.
- Untrusted webpage text and markup are **data**, never privileged instructions. Do not grant its contents shell, GitHub, payment, or filesystem actions.
- Production deployment should also add network egress ACLs, container isolation, malware scanning for any future media downloads, content quotas and monitoring. This prototype does **not** guarantee immunity to attacks or DNS rebinding.

## Learning limits and realistic goals

- `explore` now ranks **page-grounded** lexical candidates using observed headings, title and word counts. It **does not** yet truly understand these topics, independently explain them, or verify broad factual claims.
- The tiny Transformer does not become intelligent merely because the crawler discovers many pages or generates questions. Its knowledge depends on usable data, training objectives, feedback, evaluation, and compute.
- Public content is not necessarily permitted for copying or model training; the permissibility depends on content, jurisdiction, and relevant terms. The program does not claim to make legal determinations. This version merely removes training-license checks **from the act of observing public HTML**.
- For later image/audio/video/3D and social media: use dedicated opt-in/authorized connectors, scan files, minimize personal data, and run decoders in sandboxed workers.

## Quality assurance

macOS / Linux:

```bash
python3 -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
python -m unittest discover -s tests -v
```

Tests use local HTML fixtures and have **no external network dependency**. Live browsing of the Worldometer URL was not verified in the packaging environment.

## Roadmap

1. Verified factual memory and vocabulary graph from independently supported sources (not just word counts).
2. Autonomous concept-question generation and priority selection, with a bounded task queue and failure-aware retries.
3. Mathematical and programming experiments with independently executable verifiers.
4. Continual neural training on eligible and verifiable learning data; comparison against fixed curriculum using hidden holdout tests.
5. Strictly isolated multimodal ingestion with malware/quarantine scanning and provenance controls.

## Files

- `evolve/web_discovery.py`: safe public HTML reading, source profile, concept discovery.
- `evolve/store.py`: SQLite data, research observations, deduplication, frontier, experiences.
- `evolve/runtime.py`: autonomous cycles.
- `evolve/training.py`: small PyTorch model and candidate evaluation.
- `evolve/lab.py`: independently checked arithmetic problems.
- `tests/test_open_exploration.py`: research-only and cross-host tests.

## Content-grounded learning queue

For an alphabet tutorial, the system will prioritize terms actually seen in that tutorial, such as `alphabet`, `letters` and `vowels`, rather than inventing an unrelated operator-supplied subject. Discovered topics are ranked heuristically, stored with their page source and processed under queue limits. The `concepts` command shows their current status. This is a bootstrap toward intelligent self-directed research, not a demonstrated autonomous reasoning agent.

For setup, learning commands, and platform-specific tests, use the sections above.

## Check the curiosity queue on existing installations

From the repository directory, update without deleting `storage/`:

```bash
# macOS / Linux
source .venv/bin/activate
git pull origin main
python3 -m evolve discover --limit 8
python3 -m evolve questions --limit 20
python3 -m evolve status
```

The first discovery cycle after upgrading v0.4 automatically requeues older `reviewed` concept entries that had only logged questions. The open-question list does not contain answers. If `research_questions_verified` remains zero, this is accurate: evidence checking and real learning are separate future milestones.


## BERHOT EVOLVE v1.17 — independent host + char/BPE benchmark

The v1.16 Energy-only character model lowered loss after 800 steps but did not
produce coherent English. v1.17 adds a separate **strict three-host** language
experiment, train-only original BERHOT byte-BPE versus Unicode characters,
quality-proxy generation probes, immutable model history, and isolated synthetic
three-host CPU smoke tests. Existing checkpoints are kept intact.

Run `python3 -m evolve v117-smoke --steps 8` for an immediate synthetic
end-to-end test. Run `python3 -m evolve v117-plan` and
`python3 -m evolve v117-prepare` to audit whether the real corpus has enough
eligible distinct source hosts. Full guide: [v1.17 language benchmark](docs/LANGUAGE_BENCH_V117.md).

These are controlled evaluations, not autonomous self-improvement or verified
factual understanding. Real-site training still requires rights checks and
independent sources; a single article must not be scored as heldout.


## BERHOT EVOLVE v1.18 — Live Learning Studio

The **three-stage integrated laboratory** builds on the actual v1.17 source-host
checkpoints, without replacing or claiming to improve them automatically:

- **A — Expanded curriculum:** live source/rights inventory, new HTTPS article
  queue and bounded manual crawl, frozen independent-host training data.
- **B — Chat and supervised answers:** real locally generated character/BPE
  continuations, clearly labeled source quotation answers, and a separate
  training path for human-reviewed, exact-source QA examples.
- **C — Automatic experiments:** opt-in new-dataset checks, bounded CPU training,
  live step counts, elapsed time, ETA estimates, loss and separate checkpoint
  history. The daemon operates only while the local server remains running.

**Launch the browser dashboard:**

```bash
python3 -m pip install -e '.[train]'
python3 -m evolve studio --port 8787 --interval 3600
```

Visit **http://127.0.0.1:8787**. The dashboard runs entirely on the local
Mac using the existing Python/SQLite project (no OpenAI or third-party chat
API). Existing model checkpoints and data stay on disk. Raw small-model neural
text remains incoherent; readable source lookup is not neural reasoning.

[Full setup, controls and limitations](docs/LEARNING_STUDIO_V118.md).


## BERHOT EVOLVE v1.19 — Curiosity Loop and source-based self-study

**The source mode was only reading three saved science articles**. Asking
`moon` returned nothing because there was no moon article. V1.19 adds a
deliberately bounded *research from a user question* pathway.

```bash
python3 -m evolve studio --port 8787
```

Open <http://127.0.0.1:8787> and use **Curiosity** chat (now the default).
Ask **moon**. The system first searches rights-reviewed local articles, then
checks its previously recorded research. If still unknown, it queues a
background research job: up to two HTTPS pages, honoring robots, public DNS
checks, timeouts and daily quota, with NASA Moon Facts and the corresponding
Wikipedia topic as initial candidates. On completion, it shows a short literal
excerpt and a source link; it does not invent an answer or claim neural
understanding. It caches provenance and the excerpt locally, and keeps any new
archive with `UNKNOWN` rights excluded from training unless independently
reviewed. This is public information retrieval and cache, **not** a neural
weight update.

**Train from already rights-approved books** with **Teach from approved text**.
The system mechanically creates literal reading-completion examples, with
host-based train/validation/test splits, an immutable training manifest,
candidate checkpoints and validation checks. It does not pretend these
automatically generated exercises are verified factual question-answer
examples. The supervised model still may produce nonsensical text.

Alternatively:

```bash
python3 -m evolve v119-research --question "moon"
python3 -m evolve v119-inventory
python3 -m evolve v119-self-study
python3 -m evolve v118-qa-readiness
python3 -m evolve v119-study-train --steps 60
python3 -m evolve model-audit
```

The existing **Enable auto checks** option (off until you click it) now also
automatically processes a new approved-source corpus revision into reading
exercises and, if there are at least eight training, two validation and two
test examples, can train a separate QA candidate. It never auto-approves
newly researched pages, trains on unlicensed content, removes your existing
checkpoints or claims that lower loss proves independent reasoning.

Full guide: [Curiosity Loop v1.19](docs/CURIOSITY_V119.md).


## v1.20 — Grounded science definitions and actual evidence-ranking training

BERHOT EVOLVE can now show a **readable answer** when a previously
rights-reviewed science article explicitly contains a definitional
sentence. Example: asking "What is the Sun?" searches for an actual
"The Sun is ..." source quote, prefers it to an unrelated fact about
its lifetime, and shows the clickable publisher URL. It does **not**
invent an explanation. Curiosity still performs bounded research
when necessary. Grounded Q&A mode never makes network requests.

It also introduces a separate, small **PyTorch neural evidence ranker**
trained on exact-source question/evidence pairs with **independent
publisher-host train/validation/test splits** and SHA-verified immutable
checkpoints. A learned ranker chooses between existing passages;
**it is not a generative language model**. When there are not enough
explicit definitions per host, training safely skips and explains why.

```bash
python3 -m evolve v120-inventory
python3 -m evolve v120-readiness
python3 -m evolve v120-ask --question "What is the Sun?"
python3 -m evolve v120-train --steps 120
python3 -m evolve studio --port 8787 --interval 60
```

Full [v1.20 Grounded Q&A guide](docs/GROUNDED_QA_V120.md).


### v1.20.1: Diagnose missing real QA definitions

If `v120-readiness` says `needs_diverse_definitions`, run
`python3 -m evolve v120-diagnose` to see **exactly which URLs/source
hosts are missing usable examples**, or add
`--question "What is the Sun?"` to inspect nearby literal passages.
This version also recognizes publisher question titles plus exact
on-topic definitional sentences, with source/rights checks unchanged.
See [the QA corpus gap guide](docs/QA_GAPS_V1201.md).


## v1.21 — one-command autonomous curiosity, source QA and neural ranker evaluation

The bounded agent chooses real source-observed terms, asks its own short questions, optionally researches permitted HTTPS pages, creates immutable exact-source QA lessons, quizzes citation integrity, and attempts the PyTorch evidence-ranking model **only when strict independent-source train/validation/test data are ready**. No external LLM and no automatic rights approval.

```bash
python3 -m evolve v121-plan
python3 -m evolve v121-cycle --network --max-topics 6 --max-requests 6 --steps 120
python3 -m evolve v121-history
```

[Full autonomous curiosity guide](docs/AUTONOMOUS_CURIOSITY_V121.md). This does **not** demonstrate general understanding or fluent generated answers. Missing held-out NOAA/test data will still block ranker training.


## v1.21.1 — Learning quality and compact progress

A correct citation test is not a guarantee of a meaningful lesson: the extraction filter now refuses low-information subjects such as `result` and metalinguistic phrases such as `word photosphere`. Use `python3 -m evolve v121-report` for a short read-only summary of which questions were researched, citation quiz results, whether neural weights actually trained and which independent-source examples were missing *at the last cycle*. It does not rewrite historical manifests or give rights to new pages.

## v1.22 — New-topic curiosity, rights qualification, and honest held-out benchmarking

EVOLVE now advances to **unresearched** source-observed questions instead of spending subsequent cycles on the same already-researched words. `v122-evidence` inspects existing SHA-verified archives for per-host rights and candidate definition coverage **without** approving unknown training licenses or changing the stored corpus. When a genuine three-host neural ranker experiment becomes eligible, its checkpoint records an apples-to-apples lexical-vs-neural pairwise comparison on the same frozen validation/test samples. Easy mechanically labeled tests are not evidence of fluent reasoning.

```bash
python3 -m evolve v121-plan --max-topics 8
python3 -m evolve v122-evidence
python3 -m evolve v121-cycle --network --max-topics 8 --max-requests 6 --steps 120
python3 -m evolve v121-report
```

[Full v1.22 upgrade guide](docs/CURIOSITY_PROGRESSION_V122.md).


## EVOLVE v1.22.1 — Cleaner autonomous topics and source-host evidence diagnosis

The source-observed curiosity queue now rejects obvious organizational acronyms and low-information standalone adjectives, while preserving legitimate compound concepts (such as `natural selection`). It uses a conservative plural English question form for topics such as `continents` and `emissions`; `gas` remains singular. The `v122-evidence --host oceanservice.noaa.gov` report inspects the already-archived third-source testing publisher without network requests or rights changes. New research results clearly label rights metadata separately from actual neural training admission.

```bash
python3 -m evolve v121-plan --max-topics 8
python3 -m evolve v122-evidence --host oceanservice.noaa.gov
python3 -m evolve v120-readiness
```

The missing independent test data still cannot be replaced by repeated training-source quotations or unreviewed website text.

## v1.22.3 — Target one publisher in the existing crawl queue

`corpus-crawl` normally picks the **oldest pending frontier URLs across every
publisher**, not the latest two seeds. To inspect the required NOAA held-out
publisher without spending requests on unrelated URLs:

```bash
python3 -m evolve corpus-seed --urls \
  "https://oceanservice.noaa.gov/facts/current.html" \
  "https://oceanservice.noaa.gov/facts/estuary.html" --same-host

python3 -m evolve corpus-crawl --host oceanservice.noaa.gov --limit 2

python3 -m evolve v122-evidence --host oceanservice.noaa.gov
python3 -m evolve errors --limit 10
python3 -m evolve v120-readiness
```

The crawl report now lists `attempted_urls` and `archived_urls` separately.
The host filter chooses **only already-pending URLs on that exact host**.
It does not change website robots policy, HTTPS verification, per-day request
limits, source rights, existing done/error queue statuses, or the independent
train/validation/test gate. If robots.txt is denied, the source remains
unavailable and is **not** bypassed. The `retry-url` command refuses an
explicit robots exclusion; never treat a 403 robots failure as permission.

## v1.23 — Critical source-label integrity fix

The previous NOAA evaluation used **incorrect answers**: the title-heading
fallback accepted a navigation fragment like `Toggle navigation ... What is a
current?` as if it defined the topic. That invalidates the earlier claim of
100% useful held-out definition-ranking accuracy, even though the PyTorch
optimization and source-host partitioning did run.

The title extractor now requires a self-contained, subject-anchored
**declarative definition**; it will not use heading questions, menus, or
search/navigation text as answers. It can still accept literal source
definitions such as `The term "current" describes ...` and `Estuaries
and their surrounding wetlands are bodies of water ...`, if those exact
sentences exist in the previously rights-reviewed archived content.

Old model checkpoints and metrics are preserved for historical auditing
but **automatically disabled** for grounded answers, because their label
policy does not match the corrected extraction policy. A changed evidence
policy triggers a new readiness check and training attempt when eligible.
No artificial test examples, source-rights bypass, or blind-test claim.

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve v120-ask --question "What is a current?"
python3 -m evolve v120-ask --question "What is an estuary?"
python3 -m evolve v120-readiness
python3 -m evolve v121-cycle --max-topics 8 --steps 120
python3 -m evolve v121-report
python3 -m evolve model-audit
```

This is still a tiny source-derived evidence ranker; it has not demonstrated
general scientific reasoning or outperformance over lexical heuristics.

## v1.23.1 — Preserve exact excerpts while removing transcript page labels

The v1.23 run revealed an additional source-extraction artifact:
`Transcript Estuaries and their surrounding wetlands are bodies of water ...`.
This version strips an **exact leading transcript-content label** for
title-anchored science definitions, so the displayed/cited answer begins with
`Estuaries` rather than `Transcript`. The chosen answer must still be a
**contiguous substring of the immutable, SHA-verified archived text**; the
archive, URL and rights metadata are not rewritten. The subject/identity and
anti-navigation checks still apply.

The extraction-policy revision changes to `v1231_strict_subject_definition_transcript_clean`,
so the previous ranker (trained with a different label policy) is disabled
for new inference until an eligible new model is trained. Old history is
preserved. There are new offline regression tests using the precise
NOAA-style flattened transcript prefix.

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve v120-ask --question "What is an estuary?"
python3 -m evolve v120-readiness
python3 -m evolve v121-cycle --max-topics 8 --steps 120
python3 -m evolve v121-report
python3 -m evolve v120-ask --question "What is an estuary?"
```

A separate `ocean .` whitespace artifact remains in the immutable NOAA
source text. This version deliberately leaves exact punctuation untouched
rather than silently editing quoted evidence. The learned ranker still
has not outperformed the lexical topic baseline, and the NOAA sample is
previously inspected rather than a new blind evaluation set.
