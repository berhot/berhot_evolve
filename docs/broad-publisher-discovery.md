# Broad publisher discovery

The Saudi-history focus has no permanent website allowlist. The earlier
frontier nevertheless became dominated by Saudipedia because DuckDuckGo's
automated result route was unavailable, a large sitemap backlog was drained
before other publishers were checked, and opaque external citations lacked
topic words in their URLs.

The worker now:

- Starts with 18 candidate articles across 11 hosts, including education,
  research, institutional and newspaper sources. These are discovery entry
  points, not verified historical evidence or a fixed list of permitted sites.
- Reserves link slots for independent publishers, including references named
  “المصدر” on topical pages. New publishers can be added by those references
  or search results without editing the profile.
- Prioritizes publishers with fewer saved articles or attempts; origin pacing
  and retry deadlines remain independent constraints.
- Scans saved article links each cycle and explores publisher sitemaps when
  fewer than three publishers have ready work. A pending retry no longer
  prevents other discovery. Fresh publishers precede cached sitemap backlog;
  pending links from other publishers are retained.
- Alternates open-ended topic queries with independent-source variants when
  one publisher dominates the observations.
- Extracts the published article body from ASP.NET/SharePoint document forms
  marked with hidden server state. Ordinary input forms, scripts and hidden
  field values remain excluded. Previously these wrappers reduced whole
  institutional articles to their titles.

`--browser-search-provider auto` selects the official Brave Search API when
`BRAVE_SEARCH_API_KEY` is present in the worker environment, otherwise it tries
the existing policy-checked DuckDuckGo route. An unavailable engine is reported
as unavailable; article links, archive recovery and publisher sitemaps continue.
Brave is optional, requires the operator's API subscription, and is not called
without a configured key. Authentication is sent only to the fixed API host,
redirects are refused, and keys are never stored in corpus files or diagnostics.
See the [official API documentation](https://api-dashboard.search.brave.com/app/documentation/web-search).

```sh
python3 -u -m evolve focus-overnight \
  --network --profile saudi-history-ar --until-stopped --no-fixed-quotas \
  --browser-search-provider auto --interval-minutes 2 \
  --pages-per-cycle 30 --train-steps 120
```

Every destination must still pass HTTPS, public DNS, robots, access, MIME,
size, Arabic-language and topic checks. Collection does not treat search
snippets as source text. Existing cleaning, deduplication, operator approval,
per-batch raw exports, dataset shards and training holdouts are unchanged.
The default worker does not execute publisher JavaScript; empty shells,
403 responses and inaccessible publishers remain visible failures rather
than invented collected data.
