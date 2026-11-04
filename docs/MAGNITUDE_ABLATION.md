# Neural magnitude ablation: evaluation protocol

## Scientific question

Can a generic neural network generalize from 35 known single-digit addition
pairs when numeric ordinal magnitude is supplied explicitly, without
hard-coding the sum into the neural forward pass?

## Models being compared

- **Explicit structural prior (v0.9):** ten learned scalar embeddings
  combined by an actual addition operator. Learns digit parameters, but
  its architecture already provides the solution structure.
- **Categorical MLP (v1.0):** learned digit tokens, generic 19-way
  classifier. The Mac run showed 100% training but 0% unseen accuracy.
- **Magnitude MLP (v1.1):** numeric operands mapped to scaled ordinal
  input features; generic MLP regressor predicts a scalar. It does not
  add the operand values. Ordered magnitude, continuous output regression
  and commutativity remain supplied inductive biases.

## Controls

1. Per seed, generate 55 unordered pairs (0 <= a <= b <= 9).
2. Shuffle deterministically and use 35/10/10 for train/validation/test.
3. Never place a reversed pair in a different partition.
4. Use labels from the isolated Python arithmetic checker only for
   supervised training and evaluation; the model forward pass has no
   access to that checker.
5. Select neural weights by validation MSE, not test MSE or accuracy.
6. Evaluate the test set only after choosing the weights.
7. Compare stored v0.9 and v1.0 checkpoints only if seed and full pair
   split match. A mismatched or missing checkpoint is not a comparison.
8. Predeclare multiple seeds; present mean, minimum, maximum and every
   seed's result rather than reporting only a favorable seed.

## Limits

This study is narrow: known digits 0..9, one operation, 55 possible
unordered examples. Numeric magnitude is already an informative
representation. No demonstrated transfer to multi-digit arithmetic,
unseen operators, abstract reasoning, independent web-based learning,
or general model self-improvement follows from a high score.

Repeated runs and study seeds are development evaluations rather than
an untouched confirmatory dataset. Further claims require a new
preregistered holdout task and broader test distributions.

Commands (macOS, from the repository with `.venv` active):

```bash
python3 -m evolve magnitude-lab --seed 31415 --steps 800
python3 -m evolve magnitude-study --seeds 31415 42 1001 --steps 800
python3 -m evolve magnitude-predict --a 9 --b 8
```
