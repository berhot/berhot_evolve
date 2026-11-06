# Check collected text, datasets and the active model

The active learner uses [PyTorch with your own from-scratch model](pytorch-from-scratch.md).
Its article-body dataset is a new derived revision; older raw/clean exports
remain unchanged. Use the `active/saudi-history-ar` folder below to inspect
exactly what the current accepted model trained on. Use normal `python3`
without `-S` so the installed PyTorch framework is available.

Run all commands from the project directory:

```sh
cd /Users/farisalsaiari/Desktop/berhot_evolve
```

## 1. Open extracted website text

```sh
open storage/corpus_runs/COLLECTED_TEXT
```

`COLLECTED_TEXT` opens a combined export reconstructed from saved article HTML.
Refresh it without using the network with
`python3 -m evolve corpus-export --profile saudi-history-ar`.
`open storage/corpus_runs/LATEST_WITH_TEXT` opens the latest completed nonempty
batch/export. The ordinary `<worker-run>-batch-000001` folders are individual
collection batches. The filenames
mean:

| File or folder | Contents |
| --- | --- |
| `raw.txt` | Extracted plain text before cleaning, including repeated reads |
| `raw_pages/` | A separate plain text file for each successful website read |
| `raw_html/` | Compressed decoded HTML snapshots |
| `clean.txt` | Normalized text with exact duplicate documents removed within that batch |
| `clean_pages/` | Individual cleaned documents, named by SHA-256 |
| `corpus.txt` | Approved, relevant, unique text eligible for training from this batch |
| `sources.json` | Article URLs, titles, timestamps and individual file paths |
| `documents.jsonl` | Eligibility and UTF-8 byte offsets for each extracted document |
| `manifest.json` | Final counts, hashes and cumulative training dataset reference |

Open a `.txt` file in TextEdit. For a Terminal preview, run `head -c 2000`
followed by a specific file path, or `less` followed by that path (`q` exits).
Raw and clean files update while collection runs. `corpus.txt` and the final
manifest are completed at the end of the batch. An empty file during a live
batch does not mean that no text was read. A failed request has no article
text to save. Idle or failed-only cycles no longer create empty text datasets.
They are recorded in `storage/collection_audit/no_extraction_cycles.jsonl`.
Previously created zero-page folders are preserved under
`storage/collection_audit/empty_batches/`, with relocation records beside them.
An empty `corpus.txt` can still be correct when raw reads exist but none qualify
for the approved Arabic training corpus.

The permanent per-article archive is also retained in
`storage/web_corpus/raw/` and `storage/web_corpus/clean/`.

## 2. Open datasets that have already been trained

```sh
open storage/datasets/focus_plain
python3 -m evolve dataset-status
```

Dataset folders are grouped with readable folder links:

```text
storage/datasets/focus_plain/
  ready/<revision>/                 Prepared; no completed optimizer run recorded
  training/<revision>/              Optimizer run in progress
  trained/accepted/<revision>/      Candidate trained and accepted by validation
  trained/rejected/<revision>/      Candidate trained, but its weights were rejected
  trained/unclassified/<revision>/  Historical training evidence; promotion record unavailable
  failed/<revision>/                Training attempt failed
  active/saudi-history-ar/           Dataset used for the current accepted checkpoint
  <revision>/                       Original immutable dataset files
```

These folder links open the original files; they do not copy or move them.
The long revision is a SHA-256 identity. Preserve the original revision
folders and their text files. Removing or editing shards can break verified
training. A cumulative dataset can include text also present in an earlier
revision, so summing revision sizes does not count unique learned data.

## 3. Verify what was actually trained

```sh
open storage/datasets/focus_plain/active/saudi-history-ar
```

Inside the dataset:

- `train.txt`: text used for optimizer updates.
- `validation.txt`: held-out text used to accept or reject a candidate.
- `test.txt`: held-out text evaluated for diagnostics.
- `corpus.txt`: combined export of all three splits; the optimizer does not
  train on this combined file.
- `manifest.json`: document counts, split policy, file sizes and SHA-256 hashes.
- `documents.jsonl`: original URL, cleaned hash, split and byte offsets.
- `training_status.json`: current training state and last receipt location.
- `training_receipts/`: optimizer steps, acceptance, losses, checkpoint hash
  and path for each recorded attempt.

For a short readable summary in Terminal:

```sh
python3 -m evolve dataset-status > /tmp/evolve-dataset-status.json
open -a TextEdit /tmp/evolve-dataset-status.json
```

`python3 -m evolve dataset-organize` rebuilds the folder views from existing
saved training evidence if needed. New training runs manage these folders
automatically. Old model runs without an original plain-text snapshot cannot
be reconstructed as an exact historical dataset; their checkpoints remain
in `storage/models/focus/saudi-history-ar/`.

## 4. Test your active model

Open [the chat page](http://127.0.0.1:8787/chat), or click **Test active brain**
in Studio. If Studio is closed, start it:

```sh
python3 -m evolve studio --port 8787
```

1. Confirm the page shows a ready checkpoint, parameter count and accepted steps.
2. Enter an Arabic prefix such as `تأسست الدولة السعودية الأولى`.
3. Click **Generate continuation** (or the Send button shown on the page).
4. Compare the returned checkpoint hash with the active model metadata.

Each prompt is tested independently. This is raw local byte-model text
continuation, without web search, retrieved passages or a remote model.
The model was trained to predict text, not to follow chat instructions, and
its continuation can be repetitive or inaccurate. Sending prompts does not
train the model or save chat history. The active checkpoint remains readable
while the independent worker trains its next candidate.

Generation settings allow a fixed seed for comparisons, temperature `0` for
greedy decoding, and a maximum output size in UTF-8 bytes. Arabic characters
usually use multiple bytes. Sampling excludes malformed UTF-8 and nonprinting
control bytes. The continuation itself comes directly from the model logits.
