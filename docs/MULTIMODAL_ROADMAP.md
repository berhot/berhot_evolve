# Autonomous discovery across modalities: future design, not enabled

Each external connector would emit a **SourceCandidate** with license ID, evidence, URL, modality, and proof of training authorization. A rights/policy gate must pass *before* any bytes are admitted to training. Separate consent/lawful-basis reviews are necessary for personal data.

| Future input | Discovery method | Model / evaluator | Status |
| --- | --- | --- | --- |
| Text | Licensed archives, publishers' open API, verified CC0/public-domain collections | Byte/token LM, verifiable factual eval | Wikidata CC0 only operational |
| Code | Licensed repos and official source APIs respecting license and terms | Compiler, unit tests, sandbox | Planned |
| Images | CC0 image libraries and authorized APIs with per-asset rights | Image encoder/generator, image benchmark | Planned |
| Audio | Consent-based, training-authorized speech collections | ASR/TTS quality, speaker consent | Planned |
| Video | Rights-cleared footage with temporal metadata | Temporal quality, motion/world consistency | Planned |
| 3D | CC0/explicitly authorized model archives, clean mesh/UV metadata | Mesh validity, render comparisons, rig usability | Planned |
| Social platforms | Official authorized APIs, rights-holder permissions, platform-compliant limits and privacy protections | Permissioned analysis, not automatic model training | Disabled |

**Never** treat public profile visibility or the ability to view a post as permission to scrape, retain, analyze personal data, and train a commercial model. For social connectors, separate public-knowledge extraction from content copying, default to non-personal aggregates where possible, and support deletion/withdrawal workflows.

For multimodal use, store asset bytes, manifest, provenance, SHA-256, validation checks, and frozen modality-specific shards separately. Use separate models and verifiers, not one text-only training process applied to video or 3D assets.
