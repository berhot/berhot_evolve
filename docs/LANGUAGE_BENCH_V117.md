# BERHOT EVOLVE v1.17 — strict multi-source language benchmark

v1.16 successfully preserved stateful checkpoints and compared 400 vs 800
Unicode-character steps. The measured local Energy-article results improved
per-character CE, but output remained incoherent. **Do not keep adding steps
to that same article and call it understanding.**

v1.17 is a separately checkpointed experiment implementing all five requested
milestones, without changing old v1.15/v1.16 models or training registries.

## 1. Independent source-host evaluation

`python3 -m evolve v117-plan` audits eligible article count, rights metadata
and source-host gap. `python3 -m evolve v117-prepare` freezes SHA256-verified
`train/val/test` shards using the existing cross-document near-copy filtering
and **strict host-disjoint partitioning**. No within-article diagnostic mode is
permitted in v1.17. This requires **at least three distinct eligible substantive
source hosts**. A single Energy article must cause a clear `status: skipped`.

At least three source hosts is necessary for this test, not sufficient to
prove subject/topic independence, factual accuracy or broad generalization.
Test evaluations should be interpreted as exploratory and not used repeatedly
for checkpoint selection.

## 2. Expand the real corpus responsibly

Use the existing source collection workflow:

```bash
python3 -m evolve corpus-seed --urls https://YOUR_PERMITTED_SITE_1/articles/example https://YOUR_PERMITTED_SITE_2/articles/example https://YOUR_PERMITTED_SITE_3/articles/example
python3 -m evolve corpus-crawl --limit 12
python3 -m evolve web-inventory
python3 -m evolve v117-plan
python3 -m evolve v117-prepare
```

Replace the placeholders with **actual** eligible article URLs that you have
a right to use. Robots/auth/access/rate limits remain enforced by the existing
collector. The v1.17 freeze admits only metadata-declared `CC0-1.0` and
`PUBLIC-DOMAIN` pages without a `noai` signal; `UNKNOWN` sources, even if
previously opted into diagnostics, are **not** used. Excluded reasons are
reported. Source metadata and automated license recognition are not legal
advice or conclusive license verification; manually review rights and
publisher conditions before real training. v1.17 does not accept third-party
`AUTHORIZED-TRAINING` claims as a shortcut through this gate.

All corpus content is stored outside GitHub; no harvested text is committed
to the repo, and a URL alone cannot be considered training authorization.

## 3. Learn a BERHOT byte-BPE tokenizer from TRAIN only

```bash
python3 -m evolve v117-train --steps 120 --seed 31415 --device cpu --bpe-vocab 320
```

One command independently trains two small, randomly initialized PyTorch
Transformers, both using the same frozen source-host split and number of
optimizer updates:

- **Unicode character baseline:** original train-only Unicode tokenizer,
  with an explicit unknown character ID on unseen text.
- **BERHOT byte-BPE:** 256 raw byte tokens plus a deterministic,
  frequency-based sequence of pair merges fitted on TRAIN bytes only.
  Valid source bytes including Arabic round-trip exactly; byte-level
  *generation* can still contain partial UTF-8 sequences.

No external pretrained tokenizer, embeddings, or model weights. External
PyTorch `torch.nn`, `torch.optim.AdamW` and autograd are still used.
Different tokenizations have different token counts and compute per update:
this is a **small comparison**, not compute-matched model benchmarking.

## 4. Genuine testing and qualitative output

```bash
python3 -m evolve v117-evaluate
```

The command SHA-verifies frozen data, both tokenizer manifests and both
checkpoint files and evaluates deterministic source-prefix windows. It
prints train/validation/test cross entropy **per token** within each family,
and a normalized estimate of bits per predicted UTF-8 byte.

**Normalized estimates have a boundary caveat:** the first predicted token
differs between tokenizers and character-unknown positions lose information.
Do not interpret tiny differences as scientifically decisive. Compare under
a larger, controlled test with matched bytes, model compute, and proper
heldout subjects in future work.

It also generates independent outputs for:
- `Energy moves through`
- `Describe a distant forest`
- `What causes rain?`

The questions are **prompt probes**, not proof that their full phrasing was
absent from training. Automated repetition, diversity, replacement-Unicode,
punctuation and word-like-span counts are **nonsemantic proxies**, not a
validator of sentences, knowledge or reasoning. Read the text and judge
coherence yourself.

Validation, not the held-out test, decides whether a candidate is accepted.
We report test after both validation decisions are frozen.

## 5. Immutable checkpoints and lineage

V1.17 saves each family under `storage/models/v117/`, separately from
`storage/models/char_lab/`, the older byte lab and the general model.
Tokenizers live under `storage/datasets/v117/<dataset-sha>/`. A SHA-verified
`v117_history` append-only run index is maintained in the existing SQLite
store; previous records and checkpoint digests are checked before and after
training. The older char model's SHA and accepted pointer remain unchanged.

No previous checkpoint is loaded as initial neural weights for v1.17.

## One-command test RIGHT NOW, with no additional real sources

```bash
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve v117-smoke --steps 8
```

**Important:** This is an explicitly SYNTHETIC, self-authored,
three-fictional-host isolated experiment. It trains real PyTorch weights
and exercises the full freeze → train-only tokenize → train twice →
SHA-verify → evaluate → generate pipeline inside a temporary directory.
It does **not** train on the user's real single-source corpus or establish
real-world generalization. No existing checkpoint files are touched.

To test your *real* corpus today as it stands:

```bash
python3 -m evolve v117-plan
python3 -m evolve v117-prepare
```

The expected outcome with only the Energy source is `skipped` because there
is no independent holdout. This is an intentional quality gate.

## Run offline automated tests

```bash
python3 -m unittest tests.test_v117_language -v
python3 -m unittest discover -s tests -v
```

The new regression suite covers: deterministic Unicode/Arabic byte BPE
round-trip; train-only tokenizer digests; invalid/forward merge rejection;
one-source skip; distinct source hosts; real CPU gradients for both families;
separate checkpoint lineage; and refusal to read tampered checkpoints.
