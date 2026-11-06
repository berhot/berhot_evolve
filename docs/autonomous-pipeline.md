# Autonomous local learning

This worker owns the cycle: choose a pending source question, discover public
article/reference URLs, read permitted pages, archive the raw HTML/text, clean
and deduplicate article bodies, prepare language and supervised datasets, train
the project's from-scratch PyTorch GRU, evaluate, and serve an accepted model.
It does not use downloaded model weights or external model APIs.

Start one worker (do not start a second while the dashboard says active):

```sh
cd /Users/farisalsaiari/Desktop/berhot_evolve
python3 -u -m evolve focus-autonomous --network --until-stopped \
  --pages-per-cycle 5 --train-steps 120 --interval-minutes 2
```

Collection and training alternate: up to five permitted page attempts, then
120 optimizer updates, evaluation, and a two-minute pause. Network timing and
page sizes vary, so there is no fixed completion time. The Mac must stay awake.
Studio's Start button uses this pipeline once it is configured. Stop with
`python3 -m evolve focus-stop` or Studio's Stop button. Stop does not erase data.

## What runs without human labeling

The planner ranks article-title questions by the local model's answer loss.
Public Arabic MediaWiki search supplies candidate articles and their external
references, permitting discovery beyond Saudipedia. A catalog is a discovery
source, not a full index of every website. Search results are never treated as
training answers. Destinations retain public-DNS, HTTPS, robots, publisher
pacing, content checks, and source backoff.

Scheduling favors URLs with Arabic language hints for four of every five
attempts; every fifth favors unfamiliar publishers. These hints do not grant
admission: each fetched body still passes language and topical checks. Article
query IDs are retained, and common tracking parameters are removed. Activity
reports the actual search query and how much text in each batch was eligible;
an all-rejected collection batch can still train the previously saved corpus.

Supervision uses three question templates per eligible article, with an exact
paragraph excerpt as the answer. Every example records its URL, article hash,
original approved snapshot hash, answer offset and hash, and article split.
This teaches source-backed answer behavior without an outside AI or a person
writing answer labels. Excerpts can contain source errors; they are not
independently verified historical facts. The worker does not rewrite its code.

Initially three language updates alternate with one answer-supervised update.
Question bytes are masked out of the supervised loss; answer and end-marker
bytes are targets. Document coverage and per-example exposure survive dataset
growth. The worker never trains on the validation/test partitions. URL splits
remain pinned on refresh; exact cross-split body/QA duplicates are excluded.

Promotion requires improved held-out answer loss, improved combined language
and answer loss, language loss within tolerance, and passing repetition guards.
Test loss is diagnostic and never selects a checkpoint. A rejection keeps the
accepted checkpoint and reduces learning rate; the language/QA balance adjusts
according to which validation loss worsened. Recoverable failures back off and
retry. These are programmed feedback rules, not proof of human-like reasoning
or autonomous factual correction. All decisions are recorded.

## Inspect the real files

```sh
# Raw HTML/extracted text and article-body clean text, per successful batch
open storage/corpus_runs

# Current immutable article-body training data and lifecycle receipts
open storage/datasets/focus_plain/active/saudi-history-ar
python3 -m evolve dataset-status

# QA examples with source evidence and train/validation/test splits
open storage/datasets/supervised_qa

# Live status and generation from accepted weights
open http://127.0.0.1:8787/
open http://127.0.0.1:8787/chat
```

Each populated collection batch has `raw.txt`, `raw_html/`, `raw_pages/`,
`clean.txt`, `clean_pages/`, `corpus.txt`, `documents.jsonl`, `sources.json`, and
`manifest.json`. The raw files retain what was read; the training corpus includes
only eligible unique text. A no-read cycle creates an audit record instead of
empty dataset files. Training preparation may exclude low-quality bodies.

QA folders contain readable `train.txt`, `val.txt`, `test.txt` and provenance-rich
`train.jsonl`, `validation.jsonl`, `test.jsonl`, plus their checksummed manifest.
The worker stores the current QA descriptor under `focus_supervised_qa_latest`
in SQLite. Its event history identifies the exact folder for each cycle.

`/chat` automatically uses the question/answer template after a supervised
checkpoint is accepted. Raw continuation remains selectable. Both modes run
the local model; neither substitutes a retrieved paragraph for model output.
Responses remain unverified. A small GRU trained on this archive may still
produce incoherent or inaccurate answers after many updates.
