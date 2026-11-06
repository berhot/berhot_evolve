# Plain-text extraction and training corpus

Each focus collection batch now follows this sequence:

1. Read permitted HTML and extract text.
2. Save the pre-cleaning extracted text and decoded HTML snapshot immediately.
3. Normalize Unicode to NFC, remove invisible/control characters, and collapse whitespace.
4. Deduplicate whole cleaned documents by SHA-256 while retaining every source URL.
5. Prepare approved, relevant text as immutable UTF-8 train/validation/test files.
6. Verify shard and document hashes, then train using whole documents from `train.txt` only.

## Files for every extraction run

`storage/corpus_runs/<worker-run>-batch-000001/` contains:

```text
raw.txt                 All extracted text, including repeated pages
clean.txt               Unique normalized text from the batch
corpus.txt              Unique batch texts eligible for the active training workflow
raw_pages/*.txt         Separate original text extract for each successful read
raw_html/*.html.gz      Decoded HTML snapshots for each successful read
clean_pages/*.txt       Cleaned content-addressed documents
sources.json            Durable extraction index, updated after each read
documents.jsonl         Source metadata, eligibility and UTF-8 byte offsets
manifest.json           Counts, hashes and the cumulative dataset location
```

Raw and clean text files update during collection. Finalization rebuilds their
aggregates from verified individual files and writes the final manifest. Failed
requests cannot produce text. Live collection creates a bundle only after a
successful extraction; idle/failed-only cycles write an audit record in
`storage/collection_audit/no_extraction_cycles.jsonl` instead. Previously
created zero-page bundles are preserved under `collection_audit/empty_batches/`.
`corpus_runs/LATEST_WITH_TEXT` points to a completed populated bundle, and
`corpus_runs/COLLECTED_TEXT` opens the latest combined saved-article export.
Off-topic and no-AI reads can appear in the raw research bundle, but
are excluded from the training corpus. Ordinary `corpus-crawl`/`explore` runs also
produce raw/clean extraction bundles.

## Cumulative training dataset

`storage/datasets/focus_plain/<dataset-revision>/` contains:

```text
corpus.txt              All approved unique texts, including held-out documents
train.txt               Training documents
validation.txt          Validation documents
test.txt                Test documents
documents.jsonl         URL, cleaned SHA-256, split, byte offset and length
manifest.json           Per-file SHA-256, byte counts and split policy
```

The trainer loads these prepared shards and slices whole documents using the
verified offsets. It does not sample across document boundaries. `corpus.txt`
is a convenient combined export; training uses `train.txt`, and acceptance
uses validation only. Test documents do not select the checkpoint.

Identical cleaned text is assigned only once, so aliases cannot leak into a
different split. The existing persisted article or publisher split policy is
retained. Existing shards are verified and never silently overwritten if
modified. Source approval is checked before preparing/using a revision.

## Export already collected sources

```sh
python3 -m evolve corpus-export --profile saudi-history-ar
```

This requires no network access. It reconstructs the raw text from saved HTML
and labels the bundle `reconstructed_from_archived_html`; it does not claim to
recover the original HTTP wire bytes or an unsaved historical extraction.
Existing per-page archives remain intact. New batches are saved as live runs.

`python3 -m evolve corpus-status` reports the latest export/dataset locations.
The worker's saved activity includes `corpus_batch_prepared` events before
training. Model records identify the exact prepared dataset used.

Training now also writes `training_status.json` and `training_receipts/` beside
each immutable dataset. Readable folder links under `ready/`, `training/`,
`trained/accepted/`, `trained/rejected/`, and `failed/` track its lifecycle.
`active/saudi-history-ar/` opens the dataset used by the active accepted model.
Historical candidates lacking saved acceptance records appear in
`trained/unclassified/`. These metadata files do not change shard hashes.

See [the inspection and chat guide](check-your-data-and-chat.md) for commands,
file meanings and how to test `/chat` against the actual accepted checkpoint.
