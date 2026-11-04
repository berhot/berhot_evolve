# QA source coverage (v1.20.1)

The user currently has only five qualifying definitions: 4 train, 1 validation, 0 test. Training correctly skips rather than violating the host holdout.

Run:

```bash
python3 -m evolve v120-diagnose
python3 -m evolve v120-diagnose --question "What is the Sun?"
python3 -m evolve v120-diagnose --question "What is a volcano?"
```

Diagnostics list archived URLs, status of rights review, definitions per page, and verbatim nearby text for optional topics. Nearby text is only a candidate and must not be mistaken for proven QA data.

The extractor also recognizes exact publisher questions in article titles, such as "What is a current?", where the page contains an on-topic, definition-like sentence. The answer is kept as an exact quote with its original source URL and SHA256. A title alone is never an answer. A source must have existing approved rights metadata and no AI exclusion before any candidate can enter learning.

## Suggested starting science articles

- NASA Sun Facts: https://science.nasa.gov/sun/facts/
- NASA Planet: https://science.nasa.gov/exoplanets/what-is-a-planet/
- NOAA Current: https://oceanservice.noaa.gov/facts/current.html
- NOAA Estuary: https://oceanservice.noaa.gov/facts/estuary.html
- USGS Volcano: https://www.usgs.gov/media/audio/what-volcano

These URLs are for discovery, **not automatic ML training approval**. Collect them with the existing research-only corpus-seed and corpus-crawl commands, respecting robots rules and quotas. Inspect the archived articles and the publisher's actual usage terms before making any manual rights decision. Keep uncertain material as UNKNOWN.

Then rerun `v120-readiness`. Only if the source-host-held-out dataset actually satisfies the minimum definitions and topic counts should you run `v120-train --steps 120`. The ranker learns evidence selection, not fluent new explanations.
