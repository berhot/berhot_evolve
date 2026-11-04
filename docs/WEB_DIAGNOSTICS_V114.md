# BERHOT EVOLVE v1.14 — Why did it generate only spaces and e?

## Observed evidence, from the user's real Mac

- Web training completed 80 gradient steps on the single Energy article.
- Training byte CE: 5.73087 -> 3.61366 nats (36.9% reduction).
- Same-article validation byte CE: 5.67458 -> 3.72873 (34.3% reduction).
- Same-article test byte CE after training: 3.49065.
- Greedy text continuation: many spaces and repeated 'e' characters.
- Checkpoint SHA256 verified; web-specific checkpoint accepted by validation.

**A decrease in byte CE is real next-byte likelihood learning but does not
prove fluent English, factual answers, cross-article generalization or
conversational ability.** The tiny model had 32 embedding dimensions,
one Transformer layer, 64-byte context and only 80 optimizer updates.

## Next measured experiment

```bash
git pull origin main
python3 -m pip install -e '.[train]'

# Read-only diagnostic on the EXACT same saved checkpoint and frozen data:
python3 -m evolve web-diagnose --prompt 'Energy is' --new-bytes 80 \
  --temperature 0.9 --seed 42

# Compare greedily selected characters vs stochastic decoding:
python3 -m evolve web-predict --prompt 'Energy is' --new-bytes 100 \
  --decoding greedy
python3 -m evolve web-predict --prompt 'Energy is' --new-bytes 100 \
  --decoding sample --temperature 0.85 --seed 42

# OPTIONAL: a new-from-scratch larger-step candidate with the same frozen data
python3 -m evolve web-lab --steps 400 --seed 31415 --device cpu \
  --include-unknown-rights --diagnostic-within-document
python3 -m evolve web-diagnose --prompt 'Energy is' --new-bytes 80
```

`web-diagnose` compares the saved neural model against a
Laplace-smoothed byte-frequency unigram trained ONLY on the training shard,
with both models scored on precisely the same validation/test byte windows.
Lower CE is better. Negative `neural_minus_unigram_nats` means the neural
model beats this elementary baseline; positive means it has not.

Greedy and sampled continuations also expose the space fraction, unique
bytes, most common byte fraction and maximum repeated-byte run.
Sampling changes selection from existing neural logits; it cannot add
knowledge, make facts true, or improve model weights. Seeds make it
reproducible. No dataset or weight is changed by diagnosis.

## Measurement limits

All current Energy chunks originate from a single article, so test and
validation are not independent source holdouts. The baseline is a lexical
model, not a meaningful QA baseline. The original tiny neural model is
separate from the arithmetic model, so comparisons with arithmetic-byte
loss are NOT evidence of continual skill retention or forgetting.

Training for longer may help or may overfit one article. Prefer gathering
many substantive articles from multiple source-host groups and conducting
the strict heldout experiment before claims of general knowledge.

PyTorch is still the neural backend; the BERHOT-owned tensor engine remains
unimplemented. Explicit unknown-rights opt-in remains for local research;
that flag does not determine lawful training permissions.