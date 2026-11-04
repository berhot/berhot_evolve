# BERHOT EVOLVE — ownership, third-party software, and learning audit

## Factual position (v1.12)

**This project is NOT a literal zero-third-party-library implementation.**
The new open-web collector uses the Python standard library only. The
neural training engine still depends on PyTorch's tensor kernels, torch.nn,
autograd and optimizers; NumPy is declared as a training dependency.
Python, SQLite, setuptools, Docker, platform libraries and drivers are
also external runtimes or tools.

The repository's LICENSE is currently MIT. That grants third parties broad
reuse rights under its terms. A later license change does not undo earlier
MIT licenses already granted. Exclusive title cannot be established by
looking at a GitHub repository; contributors and input data may have
separate rights. The original BERHOT application logic can be independently
owned without claiming ownership of all infrastructure and web content.

## What actually learns

- The custom Python application can discover and archive web pages, keep
  research questions, and execute bounded once/daemon cycles.
- Saving a webpage or experience DOES NOT update neural weights.
- Neural weight updates only occur in explicit training steps or a running
  daemon cycle that invokes train_candidate.
- The byte-level general Transformer is built using PyTorch layers.
- The specialized arithmetic labs have saved numerical models; v1.7
  calibrated heads by least squares, v1.8 Adam/SGD tuned heads via gradients.
- v1.9 has a concept-text neural lab, but actual user-supplied data so far
  contains 0 verified text definitions and hence no concept model training.
- Saved data counts are NOT measures of verified model knowledge.

## New collection behavior

The corpus-seed and corpus-crawl commands accept any PUBLIC HTTPS HTML
sites that pass safe DNS, robots.txt, network, size and bounded crawl rules.
No CC0/PD declaration is needed merely to extract and archive text.
An absent declaration becomes rights_status=UNKNOWN. The crawler records
publisher no-AI flags instead of hiding them. Raw HTML and cleaned text
are stored separately with SHA256 digests and URL-level provenance.
Normalized content is deduplicated by exact hash; semantic deduplication
and near-copy detection are NOT yet implemented.

The collected corpus is separated from automatically approved neural
training data. This is not a claim that any public website automatically
grants commercial reuse/training rights or that copyrighted material
can be redistributed. No bypasses for logins, robots, paywalls or bans.

Historical observation records contain page hashes and metadata only:
they cannot reconstruct older text without refetching those pages.

## What a BERHOT-owned neural framework would actually require

1. Freeze reference benchmarks and checkpoint formats.
2. Implement original tensor and CPU matrix kernels for tiny experiments.
3. Implement BERHOT-owned reverse-mode autodiff and optimizers.
4. Implement neural layers: embedding, normalization, attention, MLP,
   causal cross-entropy, tokenizer and safe weight serialization.
5. Reproduce current train/validation/OOD results without PyTorch/NumPy.
6. Implement native high-performance CPU and GPU kernels if required.
7. Remove external ML imports only when all regression and quality
   tests pass. GPU drivers, hardware and OS remain external.

Zero external dependencies in the literal sense, including Python and
operating systems, is not a meaningful claim for this application.
Rebranding an imported neural library as BERHOT code is not independence.

## Checks

Run python3 -m evolve model-audit to inspect real persisted model records,
dataset counts, installed external ML libraries and the actual MIT license.
Run python3 -m evolve corpus-status to inspect collected content separately.
