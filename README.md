# BERHOT EVOLVE v0.3 — Open-Web Research Explorer

This is an **experimental learning platform**, not a self-conscious agent or general autonomous intelligence. It introduces a way to **explore public HTTPS pages without requiring a declared training license at each page**, while **separating observations from training data**. The project still uses a randomly initialized, tiny byte-level PyTorch Transformer. Concept selection is heuristic, not yet intelligent self-directed research.

## The change from v0.2

- New **`explore`** command: paste a seed URL; no `--license` or `--rights-evidence-url` required for read-only discovery.
- Optional **`--cross-host`**: follow ordinary public HTTPS links to other sites within request, depth, and frontier budgets.
- Stores research observations as **URL, title, hash, character count, and candidate concepts**. It **does not** insert full unclear-rights pages into the neural training dataset.
- A page explicitly declaring CC0 or public-domain status through supported machine-readable links can still enter the training dataset automatically, unless its metadata signals `noai`.
- Keeps source references private and retained for deduplication, verification, corrections, and deletion requests.
- Retains the v0.2 training, arithmetic verification, persistent skills and checkpoints, and long-running daemon.
- The old `web-start` exists for specifically permitted training sources. A content URL can no longer be used as its own `AUTHORIZED-TRAINING` grant evidence.

## Windows PowerShell — setup

Open PowerShell in the extracted project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train]"
```

If activation is blocked, first run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in the same session.

## Explore from one website — no API, no license parameter

The following URL is the user's example. The page is **not treated as licensed neural-training material**:

```powershell
python -m evolve explore --seed-url "https://www.worldometers.info/languages/english-alphabet/" --focus school --limit 8 --cross-host
python -m evolve status
python -m evolve discover --limit 8
```

Any ordinary publicly accessible HTTPS HTML website may be attempted, provided automated access is permitted. If its `robots.txt`, server, or access policy blocks access, this crawler stops. **No stealth, captcha bypass, paywall bypass, logins, or evasion.** Some sites require APIs or prohibit crawling.

Continue scheduled observation and eligible training, if available:

```powershell
python -m evolve once
python -m evolve daemon
```

Stop daemon using **Ctrl+C**. No GPU is necessary for crawling. Run `python -m evolve train` only if there are sufficient eligible training documents. If all pages are research-only, the model will not train from their full text; the system still records concepts and source pointers.

## Protections against malicious sites

- Only public HTTPS names on port 443 (no IP-literal, local, or private network addresses); public DNS resolution screening is best-effort.
- Honors robots.txt (RFC 9309), with bounded per-cycle and daily requests, limited crawl depth and per-page links, per-request timeouts, and response size caps.
- No browser JavaScript, HTML active content, user cookies, attachments, program execution, or document macros. Only `text/html` is fetched and parsed as text.
- Redirects are refused, which also avoids unexpected destination changes.
- Untrusted webpage text and markup are **data**, never privileged instructions. Do not grant its contents shell, GitHub, payment, or filesystem actions.
- Production deployment should also add network egress ACLs, container isolation, malware scanning for any future media downloads, content quotas and monitoring. This prototype does **not** guarantee immunity to attacks or DNS rebinding.

## Learning limits and realistic goals

- `explore` extracts lexical candidate concepts and records observations. It **does not** yet independently explain the word `school`, formulate good web searches, or verify broad factual claims.
- The tiny Transformer does not become intelligent merely because the crawler discovers many pages. Its knowledge depends on usable data, training objectives, feedback, evaluation, and compute.
- Public content is not necessarily permitted for copying or model training; the permissibility depends on content, jurisdiction, and relevant terms. The program does not claim to make legal determinations. This version merely removes training-license checks **from the act of observing public HTML**.
- For later image/audio/video/3D and social media: use dedicated opt-in/authorized connectors, scan files, minimize personal data, and run decoders in sandboxed workers.

## Quality assurance

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
