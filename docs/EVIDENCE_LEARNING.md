# BERHOT EVOLVE v0.6 — Evidence, verification and continual learning

This is an experimental, **bounded** pipeline. The verifier is an English-language
definition corroborator: it does not establish a general ability to fact-check
all claims or understand what is read.

## The research cycle

1. Open `research_questions` identifies a supported `What is ...?` question.
2. Candidate URLs are drawn from the source page, pages whose concept metadata
   mentions the term, and (if cross-host discovery is enabled) one Wikipedia
   article title derived from the term. URLs pass the existing public HTTPS,
   robots, redirect and DNS protections.
3. Relevant *definition clauses* are extracted in memory, not whole raw web
   bodies inserted into neural training.
4. The verifier requires matching claims on two distinct publisher-domain hints;
   it uses a conservative shared-content-word threshold and detects mismatched
   numbers and explicit negation. The method is recorded as
   `two_source_definition_overlap_v1`. Shared hosting, syndication,
   hallucinations and matching misinformation can still produce false positives.
5. The SQL record stores the answer, source URLs, license indicators, evidence
   hashes and verification method. Unsupported or disputed questions stay open.
   A cooldown reduces repeated requests; evidence requests share the crawler
   daily budget.
6. If **one of the matching evidence pages independently declares CC0 or public
   domain**, a *single concise QA example* based only on that reusable page
   can be added as `topic=verified:definition`,
   `license=SELF-GENERATED-VERIFIED`.
7. `train --verified-only` selects only that dataset. The training code
   continues from the last accepted checkpoint if compatible, records candidate
   checkpoints and train/heldout loss, and only promotes a later checkpoint when
   heldout byte-prediction loss improves by the configured margin.

Two-source definition corroboration is a useful early, **fallible** answer
check; it is not independent confirmation of reality. Do not assume the
result is safe for medicine, finance, law, research claims or other high-stakes
uses. Evidence diversity currently uses domain hints, not actual publisher
ownership.

## Local commands (macOS zsh)

```bash
source .venv/bin/activate
git pull origin main
python3 -m pip install -e '.[train]'
python3 -m evolve questions --limit 20
python3 -m evolve research --limit 5
python3 -m evolve answers --limit 20
python3 -m evolve train --verified-only
python3 -m evolve status
```

The combined `learn --limit 10` command researches up to ten open questions
and then attempts verified-only training. `once` and `daemon` also run the
research step for a registered public web profile.

For a research-only site such as Worldometer, verification might succeed while
training remains skipped if no eligible reusable evidence was found. With only
six pages originally discovered from a single site, there might not yet be
enough independent definitional support. This is expected, not a license
bypass or training failure.

## Tests and measurable boundaries

```bash
python3 -m unittest discover -s tests -v
```

The `tests/test_verified_learning.py` fixture verifies:
- Cross-host matching definitions can become source-backed answers.
- Unlicensed definitions can be researched but cannot become PyTorch examples.
- Disagreeing numbers, negation, and sources under the same publisher-domain hint
  do not pass the verifier.
- Unsupported questions remain open.
- Six independently corroborated, reusable QA examples support a first CPU
  training run, and six more support a second run, with separate train and heldout
  losses and checkpoint files.
- The code skips retraining without enough new eligible examples.

These tests use synthetic, offline webpages whose rights declarations are
fixtures. They do not demonstrate that any given live public website offers
training rights or that the tiny model is factually proficient. Real-world
evaluation requires a held-out factual question-answer benchmark and
human-reviewed sources, not only lower byte-level validation loss.
