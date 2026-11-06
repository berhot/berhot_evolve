# Approving saved training text

To approve the currently archived corpus for this local project:

```sh
python3 -m evolve corpus-approve-training --reason "Operator approved the current archived corpus"
```

Approval is recorded for each URL and cleaned SHA-256 digest. It preserves
publisher license metadata, verifies the archived files, and excludes no-AI
records. To persist approval for future sources and updated snapshots:

```sh
python3 -m evolve corpus-approve-training --include-future --reason "Operator approves present and future collections"
```

The archive transaction records each new snapshot under that standing policy.
Use `--no-include-future` to stop approving future snapshots; existing approvals
remain recorded. With no policy enabled, new sources need explicit approval.
Approval is an operator training decision, not a public-domain declaration.

All corpus training consumers use the same approval predicate. Dataset
manifests record operator approval alongside the original rights metadata.
The Saudi-history Arabic model normally requires three independent publishers
for train/validation/test partitions. To train the existing single-publisher
corpus as an explicitly experimental article-held-out model:

```sh
python3 -m evolve focus-training-policy --allow-article-holdout
```

Use **Train saved Arabic corpus** in Studio. Whole cleaned article digests are
assigned once to train/validation/test. Exact copies never cross partitions,
and future pages do not move existing assignments. Checkpoints and Studio label
this evaluation as experimental, without claiming independent publisher
generalization. New approved data permits another bounded training run that
resumes compatible accepted weights. An unchanged corpus is not retrained in
a loop. `--no-allow-article-holdout` restores the independent-publisher policy.

# Recovering previously refused redirects

With the focus worker stopped:

```sh
python3 -m evolve focus-retry-redirects
```

This requeues saved redirect failures from the latest run without restarting
the worker or erasing historical failures. During a run, same-host HTTPS
redirect destinations are queued as separate requests. Each destination gets
normal DNS, robots, format and publisher-pacing checks. Redirect chains are
bounded, Unicode/escaped aliases are deduplicated, and cross-host or HTTP
redirects remain refused.

Missing pages, unavailable DNS, unverifiable or oversized robots files and
publisher server failures remain distinct failures. The discovery filter
rejects whitespace-only paths such as `https://www.spa.gov.sa/%20`.

# Increasing Arabic model capacity

Stop the focus worker before changing dimensions:

```sh
python3 -m evolve focus-stop
python3 -m evolve focus-model-capacity --embedding-dim 256 --hidden-size 768
```

Wait for the worker to stop before running the capacity command. This selects
a 2,626,304-parameter byte GRU, up from the legacy 79,168-parameter model.
Dimensions are persisted per profile; they are separate from the general
Transformer configuration in settings.json.

The next Arabic training job widens compatible accepted weights, preserving
the old prediction function initially. Added units learn during subsequent
optimizer updates. Architecture metadata is included in checkpoint manifests
and dataset revisions, so the unchanged corpus can train once after an upgrade.
A candidate must improve its measured validation loss to become accepted.
The previous checkpoint file and accepted history remain available. Studio
distinguishes configured capacity from the previously accepted model until
the larger checkpoint is accepted. Resume the normal focus worker after the
upgrade; it uses the persisted dimensions for future collection/training cycles.
