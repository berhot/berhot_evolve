# Train a complete pass of the saved corpus

```sh
python3 -u -m evolve focus-full-pass --resume-collection
```

Stop an existing focus worker first with `python3 -m evolve focus-stop`.
The full pass prioritizes saved-data training. When it finishes, the flag
above resumes the already-authorized collection/training pipeline. Without
that flag it finishes after the pass. No external model or downloaded weights
are involved; the existing project GRU and Adam optimizer use PyTorch.

The command freezes the current approved article-body dataset and its QA
examples. It starts a new coverage counter and trains every next-byte target
in every **training** article exactly once, plus every training QA example
once. It spreads QA examples across the language pass. Validation and test
articles remain excluded. Raw archive totals include menus, duplicates and
evaluation text, so they exceed actual training bytes. Existing article
assignments, approvals, source license metadata and raw files are preserved.

This mode does not discard the candidate after 120 steps. It saves current
weights, Adam moments, document offsets, QA position and counters every 120
steps. A graceful Stop saves the current position too. Re-run the same
command to resume an interrupted pass. A crash can replay at most the work
since the previous checkpoint. Approved source snapshots are rechecked on
resume. Do not run two workers against the same profile.

The active chat model remains separate. Evaluation happens after the full
pass, comparing held-out language/answer loss and repetition against the
accepted model. A rejected final candidate remains saved for inspection;
rejection does not falsely label it as the active chat model. Historical
accuracy is not established by these loss/fluency checks.

Live progress is recorded under `focus_full_pass_progress` in SQLite,
included in `/api/status` and `/api/brain`, and shown in Studio's Arabic
full-pass progress row. It reports steps done/remaining, training byte targets
processed and an estimate based on measured speed. The estimate changes with
load and is not a promised finish time. Its checkpoint pointer is separately
stored under `focus_full_pass_checkpoint`; optimizer and cursor there always
belong to the same saved checkpoint. Only the most recent private candidate
is retained during a pass; accepted checkpoints are protected.

The exact frozen text files are `train.txt`, `validation.txt`, `test.txt` and
`documents.jsonl` in the SHA-256 dataset directory referenced by the progress
record. Per-attempt receipts remain under its `training_receipts/` directory.
