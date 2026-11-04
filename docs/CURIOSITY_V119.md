# BERHOT EVOLVE v1.19 — How the new Curiosity Loop works

This update follows v1.18's local Learning Studio. It does not use an
external language-model API and does not change existing trained weights
unless a separate, bounded neural training job is started.

## 1. Start the UI

```bash
git switch main
git pull --ff-only origin main
python3 -m pip install -e '.[train]'
python3 -m evolve studio --port 8787 --interval 3600
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787).

In **Talk to EVOLVE**, the default is now **Curiosity**. Enter **moon**:

1. BERHOT checks previously approved, SHA-verified local article passages.
2. If no useful passage exists, it checks earlier research records (SQLite).
3. If still missing, it creates a bounded research job in the background.
4. The public HTTPS fetcher attempts a maximum of two subject-specific
   candidate URLs; `moon` tries NASA Moon Facts and a public encyclopedia.
5. It obeys the same robots, DNS protections, request budget, response limits
   and no-redirect policy as the existing crawler.
6. It extracts a short, source-linked passage (not an invented answer) and
   marks it *research memory, NOT neural learning*.
7. On a later question about that topic, it can reuse the recorded passage.
   A failed research attempt is cooldown-limited and never fabricated.

The NASA Moon Facts candidate was checked: https://science.nasa.gov/moon/facts/.
A public page may still be blocked by robots, timeouts or availability.
If both attempts fail, EVOLVE says it could not find an answer and shows no
fake confidence. The search is intentionally narrow and lexical. Broad or
complex questions may be unsupported.

The newly found article can be *archived for further inspection*, but the
`UNKNOWN` rights marker is never a training grant. **Research memory is not
training-ready data**. Existing manually reviewed CC0/public-domain rights
metadata is not overwritten when revisiting an already archived article.
For newly found content, manually inspect authorship, rights and restrictions
using `v117-sources` and `v117-review-rights` before allowing it into a
training dataset.

## 2. Let BERHOT create reading exercises without typing each lesson

Click **Teach from approved text · 60 steps**, or use:

```bash
python3 -m evolve v119-self-study
python3 -m evolve v118-qa-readiness
python3 -m evolve v119-study-train --steps 60
```

The program extracts original sentences from your already-approved local
articles. It builds deterministic, exact-source reading practice:

- Original article: `Some scientific observation continues with ...`
- Question: `Continue: Some scientific observation`
- Target answer: the exact remaining words from that article

No external AI writes the example, no web copyright is silently assumed, and
the answer is SHA-tied to the archived, reviewed source. Sentences are
screened for navigation, low quality and repeats across hosts. At most 18
examples per source host are built by default, with a bounded global source
scan. Training still uses the original v1.18 SHA-frozen QA manifest, source-
host partition, small PyTorch model, held-out validation and test, and a new
immutable checkpoint path.

**Important:** These exercises teach *text continuation*, NOT the factual
meaning of the source material. They do not make the model able to answer
arbitrary questions like ChatGPT. The source-lookup chat path stays separate
from neural generation.

If there are not enough usable examples (minimum 8 training, 2 validation,
2 test), the neural training step is skipped honestly. The UI reports what
was extracted and whether a new model was trained. It never counts newly
fetched unknown-rights pages as automatically approved examples.

## 3. Autonomous checks without repeated manual commands

Click **Enable auto checks** in Learning Studio.

While Studio runs, the loop checks approved archived data at its configured
interval. Once a *new approved corpus revision* is detected, it attempts
one bounded self-study pass. It will not endlessly repeat a rejected model
on an unchanged source revision. The earlier v1.18 rule still checks a new
large strict-host data split (12 eligible articles, 60 KB training text)
before running a fresh character/BPE experiment.

For explicit user questions in Curiosity mode, read-only research runs
without requiring you to supply a source URL. It uses fixed, safe URL
candidates and no broad unrestricted search engine or autonomous browsing.

The loop does NOT:
- approve intellectual-property rights on its own;
- learn unknown-rights web pages directly into weights;
- invent factual answers when no source is found;
- rewrite existing v1.16/v1.17 or v1.18 checkpoints;
- discover arbitrary new scientific goals or prove general reasoning;
- learn while the Mac is shut down or the Studio process has stopped.

## 4. Tracking

The Studio now tracks separately:

- real source and document counts;
- automatically prepared source-reading examples versus human QA examples;
- research job status and research attempts;
- the active job, its elapsed seconds, steps, loss and optional ETA;
- actual trained model checkpoint SHA-256 values;
- whether neural weights were updated or a page was merely cached.

Commands:

```bash
python3 -m evolve v119-research --question "moon"
python3 -m evolve v119-inventory
python3 -m evolve v119-self-study
python3 -m evolve v119-study-train --steps 60
python3 -m evolve v118-chat --mode byte_bpe --question "Moon"
python3 -m evolve model-audit
```

The `v118-chat` CLI keeps its historical modes; the new question-driven
internet research mode is available via `v119-research` or the Studio.
The model is still very small and will usually produce poor English from
raw weights.

## 5. Security and quality

All network access is controlled by the project's public-HTTPS crawler.
Research quotas share the daily crawler request budget. The local web
server remains bound to 127.0.0.1 and enforces Host/Origin and anti-CSRF
checks for its private local mutations. No provider API tokens are required.

Read-only source snippets are not independent factual verification. A single
citation says where the text came from, not that it is true. The extraction
heuristic favors more explanatory sentences over short image captions, but
cannot always locate the right paragraph. The human remains responsible for
evaluating the model's factual reliability.

Recommended next steps after this version: increase source diversity,
enable more than two safe candidate pages for well-grounded research, create
meaningfully independently checked QA labels (not cloze alone), use a much
larger model/corpus for coherent language, and evaluate with a frozen external
benchmark before promoting any neural checkpoint.
