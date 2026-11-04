# BERHOT EVOLVE v0.8 — Answer-Focused Training & Honest Skill Evaluation

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
