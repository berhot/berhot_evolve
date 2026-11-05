"""SQLite-backed durable frontier, provenance, experience, and progress log."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "evolve.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS frontier (
                entity_id TEXT PRIMARY KEY, depth INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, discovered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL, license TEXT NOT NULL,
                modality TEXT NOT NULL DEFAULT 'text', split TEXT NOT NULL,
                topic TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL,
                prompt TEXT NOT NULL, prediction TEXT, expected TEXT,
                success INTEGER NOT NULL, verified_by TEXT NOT NULL,
                model_sha256 TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
                instructions TEXT NOT NULL, evidence_count INTEGER NOT NULL,
                verified_by TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
                status TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL,
                sha256 TEXT NOT NULL, train_documents INTEGER NOT NULL,
                val_loss REAL NOT NULL, baseline_val_loss REAL,
                promoted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS web_frontier (
                url TEXT PRIMARY KEY, depth INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, discovered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS web_corpus_texts (
                sha256 TEXT PRIMARY KEY, clean_path TEXT NOT NULL,
                characters INTEGER NOT NULL, words INTEGER NOT NULL,
                quality_score REAL NOT NULL, split TEXT NOT NULL,
                first_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS web_corpus_sources (
                url TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
                raw_html_sha256 TEXT NOT NULL, raw_path TEXT NOT NULL,
                title TEXT NOT NULL, host TEXT NOT NULL,
                rights_status TEXT NOT NULL, rights_evidence_url TEXT,
                noai INTEGER NOT NULL DEFAULT 0,
                observed_at TEXT NOT NULL,
                FOREIGN KEY(sha256) REFERENCES web_corpus_texts(sha256)
            );
            CREATE INDEX IF NOT EXISTS web_corpus_sources_sha_idx
              ON web_corpus_sources(sha256);
            CREATE TABLE IF NOT EXISTS page_observations (
                url TEXT PRIMARY KEY, title TEXT NOT NULL, sha256 TEXT NOT NULL,
                characters INTEGER NOT NULL, extracted_concepts TEXT NOT NULL,
                training_status TEXT NOT NULL, observed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_tasks (
                term TEXT PRIMARY KEY, source_url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT NOT NULL,
                question TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_title TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                UNIQUE(concept,question,source_url)
            );
            CREATE TABLE IF NOT EXISTS verified_answers (
                question_id INTEGER PRIMARY KEY REFERENCES research_questions(id),
                concept TEXT NOT NULL, answer TEXT NOT NULL,
                evidence_json TEXT NOT NULL, verification_method TEXT NOT NULL,
                training_eligible INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            -- Append-only worker activity: no fixed history retention cap.
            CREATE TABLE IF NOT EXISTS focus_activity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                profile TEXT NOT NULL,
                run_id TEXT NOT NULL,
                at_utc TEXT NOT NULL,
                kind TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                url TEXT
            );
            CREATE INDEX IF NOT EXISTS focus_activity_profile_id_idx
              ON focus_activity_events(profile,id);
            -- Durable focus-only corpus membership. The latest-run counters
            -- reset on restart; these source facts must never do so.
            CREATE TABLE IF NOT EXISTS focus_archived_sources (
                profile TEXT NOT NULL, url TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                clean_utf8_bytes INTEGER NOT NULL DEFAULT 0,
                source_questions INTEGER NOT NULL DEFAULT 0,
                recorded_at_utc TEXT NOT NULL,
                PRIMARY KEY(profile,url)
            );
            CREATE INDEX IF NOT EXISTS focus_sources_profile_sha_idx
              ON focus_archived_sources(profile,source_sha256);
        """)
        # Migrate existing v0.3 databases without losing discoveries.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(concept_tasks)")}
        if "priority" not in columns:
            self.db.execute("ALTER TABLE concept_tasks ADD COLUMN priority REAL NOT NULL DEFAULT 0")
        if "source_title" not in columns:
            self.db.execute("ALTER TABLE concept_tasks ADD COLUMN source_title TEXT NOT NULL DEFAULT ''")
        # v0.4 'reviewed' meant only that questions were logged; requeue
        # those topics once so persistent open questions can be created.
        self.db.execute("UPDATE concept_tasks SET status='pending' WHERE status='reviewed'")
        question_columns = {row[1] for row in self.db.execute("PRAGMA table_info(research_questions)")}
        if "attempts" not in question_columns:
            self.db.execute("ALTER TABLE research_questions ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "last_attempt_at" not in question_columns:
            self.db.execute("ALTER TABLE research_questions ADD COLUMN last_attempt_at REAL")
        self.db.commit()

    def record_focus_archive(self, profile, url, source_sha256,
                             source_questions=0):
        """Associate archived source with focus corpus across worker restarts.

        Record the size of the cleaned immutable UTF-8 text, not the size
        of the raw HTML, extracted pre-normalization text, or training bytes.
        Never open files outside this store's clean corpus directory.
        """
        if not profile or not url or not source_sha256:
            raise ValueError("Focus source identity required")
        row = self.db.execute(
            "SELECT clean_path FROM web_corpus_texts WHERE sha256=?",
            (source_sha256,)).fetchone()
        if row is None:
            return False
        root = (self.root / "web_corpus" / "clean").resolve()
        candidate = Path(row["clean_path"]).resolve()
        size = 0
        if candidate.is_relative_to(root) and candidate.is_file():
            size = candidate.stat().st_size
        self.db.execute(
            "INSERT INTO focus_archived_sources "
            "(profile,url,source_sha256,clean_utf8_bytes,source_questions,"
            "recorded_at_utc) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(profile,url) DO UPDATE SET "
            "source_sha256=excluded.source_sha256,"
            "clean_utf8_bytes=excluded.clean_utf8_bytes,"
            "source_questions=excluded.source_questions,"
            "recorded_at_utc=excluded.recorded_at_utc",
            (profile, url, source_sha256, size,
             max(0, int(source_questions)), now()))
        self.db.commit()
        return True

    def backfill_focus_archives(self, profile, observations):
        """One-time idempotent recovery for pre-index source observations."""
        indexed = {r["url"]: r["source_sha256"] for r in self.db.execute(
            "SELECT url,source_sha256 FROM focus_archived_sources "
            "WHERE profile=?", (profile,))}
        changed = 0
        for item in observations or []:
            if not isinstance(item, dict):
                continue
            url, digest = item.get("url"), item.get("source_sha256")
            if not url or not digest or indexed.get(url) == digest:
                continue
            if self.record_focus_archive(
                    profile, url, digest, len(item.get("curiosity") or [])):
                changed += 1
        return changed

    def focus_archive_totals(self, profile):
        """Saved cross-session totals; unique bytes dedup by clean SHA."""
        row = self.db.execute(
            "SELECT COUNT(*) AS source_urls,"
            "COUNT(DISTINCT source_sha256) AS unique_texts,"
            "COALESCE(SUM(source_questions),0) AS source_questions "
            "FROM focus_archived_sources WHERE profile=?", (profile,)
        ).fetchone()
        size = self.db.execute(
            "SELECT COALESCE(SUM(size),0) FROM ("
            "SELECT MAX(clean_utf8_bytes) AS size "
            "FROM focus_archived_sources WHERE profile=? "
            "GROUP BY source_sha256)", (profile,)
        ).fetchone()[0]
        return {
            "source_urls": int(row["source_urls"]),
            "unique_texts": int(row["unique_texts"]),
            "source_questions": int(row["source_questions"]),
            "unique_clean_utf8_bytes": int(size),
            "scope": "saved_focus_archive_all_runs",
            "training_bytes": 0,
        }

    def close(self):
        self.db.close()

    def add_frontier(self, entity_id, depth=0, maximum=5000):
        if not isinstance(entity_id, str) or not entity_id.startswith("Q") or not entity_id[1:].isdigit():
            return False
        if self.count("frontier") >= maximum:
            return False
        cur = self.db.execute(
            "INSERT OR IGNORE INTO frontier(entity_id,depth,discovered_at) VALUES(?,?,?)",
            (entity_id, int(depth), now()),
        )
        self.db.commit()
        return cur.rowcount > 0

    def next_entity(self):
        return self.db.execute(
            "SELECT * FROM frontier WHERE status='pending' ORDER BY depth ASC, discovered_at ASC LIMIT 1"
        ).fetchone()

    def mark_entity(self, entity_id, status, error=None):
        self.db.execute(
            "UPDATE frontier SET status=?, attempts=attempts+1, last_error=? WHERE entity_id=?",
            (status, (str(error)[:300] if error else None), entity_id),
        )
        self.db.commit()

    def add_web_url(self, url, depth=0, maximum=5000):
        if self.db.execute('SELECT COUNT(*) FROM web_frontier').fetchone()[0]>=maximum:return False
        cursor=self.db.execute('INSERT OR IGNORE INTO web_frontier(url,depth,discovered_at) VALUES(?,?,?)',
                               (url, int(depth), now()))
        self.db.commit()
        return cursor.rowcount>0

    def next_web_url(self, host=None):
        """Select oldest pending URL, optionally limited to one exact HTTPS host.

        Use a hostname-validated filter from discover_web(); matching the
        slash after the hostname prevents prefix-host confusion.
        """
        if host is None:
            return self.db.execute(
                "SELECT * FROM web_frontier WHERE status='pending' "
                "ORDER BY depth,discovered_at LIMIT 1").fetchone()
        return self.db.execute(
            "SELECT * FROM web_frontier WHERE status='pending' "
            "AND url LIKE ? ORDER BY depth,discovered_at LIMIT 1",
            ("https://" + host + "/%",)).fetchone()

    def mark_web_url(self,url,status,error=None):
        self.db.execute('UPDATE web_frontier SET status=?,attempts=attempts+1,last_error=? WHERE url=?',
                        (status,str(error)[:300] if error else None,url))
        self.db.commit()

    def retry_failed_web_url(self, url):
        """Requeue one failed seed after the underlying connection issue is fixed.

        Never silently retry explicit robots exclusion.
        Keep previous error text and attempts for troubleshooting.
        """
        row = self.db.execute(
            "SELECT status,last_error,attempts FROM web_frontier WHERE url=?", (url,)
        ).fetchone()
        if row is None:
            return {"status": "not_found", "url": url}
        if row["status"] != "error":
            return {"status": "not_failed", "url": url, "current_status": row["status"]}
        reason = row["last_error"] or ""
        if "Website disallows automated access" in reason:
            return {"status": "blocked_by_robots", "url": url, "reason": reason}
        self.db.execute("UPDATE web_frontier SET status='pending' WHERE url=?", (url,))
        self.db.commit()
        self.log("web_manual_retry", "queued", {"url": url, "previous_error": reason})
        return {"status": "requeued", "url": url, "previous_error": reason}


    def add_concept_task(self,term,url,max_total=60,priority=0.0,source_title=''):
        # Additional source evidence can raise priority without duplicating tasks.
        existing = self.db.execute('SELECT priority FROM concept_tasks WHERE term=?',(term,)).fetchone()
        if existing:
            if float(priority)>existing['priority']:
                self.db.execute('UPDATE concept_tasks SET priority=?,source_url=?,source_title=? WHERE term=?',
                                (float(priority),url,source_title[:300],term))
                self.db.commit()
            return False
        if self.pending_concepts()>=max_total:return False
        if self.db.execute('SELECT COUNT(*) FROM concept_tasks').fetchone()[0]>=100_000:return False
        cursor=self.db.execute('INSERT OR IGNORE INTO concept_tasks '
                               '(term,source_url,created_at,priority,source_title) VALUES(?,?,?,?,?)',
                               (term,url,now(),float(priority),source_title[:300]))
        self.db.commit()
        return cursor.rowcount>0

    def save_observation(self, url, title, text, concepts, training_status="research_only"):
        """Record page metadata and lexical discoveries, not a training text copy.

        Keep an internal source pointer to permit later correction and provenance.
        """
        self.db.execute(
            "INSERT OR REPLACE INTO page_observations "
            "(url,title,sha256,characters,extracted_concepts,training_status,observed_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (url, title[:300], hashlib.sha256(text.encode('utf-8')).hexdigest(),
             len(text), json.dumps(concepts, ensure_ascii=False), training_status, now()),
        )
        self.db.commit()

    def next_concept_task(self):
        return self.db.execute("SELECT * FROM concept_tasks WHERE status='pending' "
                               "ORDER BY priority DESC,created_at,term LIMIT 1").fetchone()

    def list_concept_tasks(self,limit=20):
        return [dict(row) for row in self.db.execute(
            "SELECT term,status,ROUND(priority,1) AS priority,source_title,source_url "
            "FROM concept_tasks ORDER BY CASE WHEN status='pending' THEN 0 ELSE 1 END, "
            "priority DESC,created_at LIMIT ?",(max(1,min(int(limit),100)),))]

    def concept_sources(self,term):
        """Use research observations as evidence, not only training documents."""
        matches=[]
        for row in self.db.execute('SELECT url,title,extracted_concepts FROM page_observations'):
            if term in json.loads(row['extracted_concepts']):
                matches.append({'url':row['url'],'title':row['title']})
        return matches

    def mark_concept_task(self,term,status):
        self.db.execute('UPDATE concept_tasks SET status=? WHERE term=?',(status,term))
        self.db.commit()

    def add_research_question(self, concept, question, source_url, source_title='', maximum=5000):
        """Queue a source-grounded research question, NOT a verified claim."""
        if self.db.execute('SELECT COUNT(*) FROM research_questions').fetchone()[0] >= maximum:
            return False
        cur = self.db.execute(
            "INSERT OR IGNORE INTO research_questions "
            "(concept,question,source_url,source_title,created_at) VALUES (?,?,?,?,?)",
            (concept, question[:500], source_url, source_title[:300], now()))
        self.db.commit()
        return cur.rowcount > 0

    def list_research_questions(self, limit=20):
        return [dict(row) for row in self.db.execute(
            "SELECT id,concept,question,status,source_title,source_url "
            "FROM research_questions ORDER BY CASE WHEN status='open' THEN 0 ELSE 1 END,"
            " id LIMIT ?", (max(1, min(int(limit), 100)),))]

    def open_questions_count(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM research_questions WHERE status='open'").fetchone()[0]

    def next_open_questions(self, limit=5, cooldown_seconds=3600,
                            supported_only=False, concept=None):
        """Choose eligible questions; optional concept selection is exact and read-safe."""
        import time
        cutoff = time.time() - max(0, float(cooldown_seconds))
        conditions = ["status='open'",
                      "(last_attempt_at IS NULL OR last_attempt_at<=?)"]
        params = [cutoff]
        if supported_only:
            conditions.append("question LIKE 'What is %?'")
        if concept is not None:
            conditions.append("LOWER(concept)=?")
            params.append(str(concept).strip().casefold())
        params.append(max(1, min(int(limit), 50)))
        return [dict(row) for row in self.db.execute(
            "SELECT id,concept,question,source_url,source_title FROM research_questions "
            "WHERE " + " AND ".join(conditions) + " ORDER BY attempts, id LIMIT ?",
            params)]

    def unsupported_question_count(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM research_questions "
            "WHERE status='open' AND question NOT LIKE 'What is %?'").fetchone()[0]

    def related_frontier_urls(self, concept, limit=12):
        """Find additional already-discovered URLs with meaningful concept slugs."""
        from urllib.parse import unquote, urlsplit
        import re
        key = re.sub(r"[^a-z0-9]+", "-", concept.casefold()).strip("-")
        if len(key) < 3:
            return []
        found = []
        for row in self.db.execute(
            "SELECT url FROM web_frontier WHERE status IN ('pending','done') "
            "ORDER BY depth,discovered_at LIMIT 5000"):
            slug = unquote(urlsplit(row['url']).path).casefold()
            slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
            if key in slug.split("-") or key in slug:
                found.append(row['url'])
                if len(found) >= max(1, min(int(limit), 50)):
                    break
        return found

    def mark_question_attempt(self, question_id):
        import time
        self.db.execute(
            "UPDATE research_questions SET attempts=attempts+1,last_attempt_at=? "
            "WHERE id=? AND status='open'", (time.time(), int(question_id)))
        self.db.commit()

    def save_verified_answer(self, question_id, concept, answer, evidence, method, training_eligible=False):
        """Persist a machine-corroborated answer; require two distinct host groups."""
        publishers = {entry.get("publisher") for entry in evidence if entry.get("publisher")}
        if len(publishers) < 2 or len(answer) > 300 or method != "two_source_definition_overlap_v1":
            raise ValueError("Verification requires two distinct source groups and bounded answer")
        question = self.db.execute(
            "SELECT status,concept FROM research_questions WHERE id=?", (int(question_id),)
        ).fetchone()
        if question is None or question["status"] != "open":
            return False
        if question["concept"].casefold() != concept.casefold():
            raise ValueError("Answer concept does not match source question")
        if training_eligible and not any(item.get("license") in ("CC0-1.0", "PUBLIC-DOMAIN") for item in evidence):
            raise ValueError("Training requires reusable evidence")
        self.db.execute(
            "INSERT INTO verified_answers "
            "(question_id,concept,answer,evidence_json,verification_method,training_eligible,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (int(question_id), concept, answer, json.dumps(evidence, ensure_ascii=False),
             method, int(training_eligible), now()))
        self.db.execute("UPDATE research_questions SET status='verified' WHERE id=?", (int(question_id),))
        self.db.commit()
        return True

    def list_verified_answers(self, limit=20):
        results = []
        for row in self.db.execute(
            "SELECT question_id,concept,answer,evidence_json,verification_method,"
            "training_eligible,created_at FROM verified_answers ORDER BY question_id DESC LIMIT ?",
            (max(1, min(int(limit),100)),)):
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            item["training_eligible"] = bool(item["training_eligible"])
            results.append(item)
        return results



    def recent_web_errors(self, limit=10):
        """Return recorded crawl failures without exposing scraped page contents."""
        limit = max(1, min(int(limit), 100))
        return [dict(row) for row in self.db.execute(
            "SELECT url, depth, attempts, last_error "
            "FROM web_frontier WHERE status='error' "
            "ORDER BY rowid DESC LIMIT ?", (limit,))]

    def pending_concepts(self):
        return self.db.execute("SELECT COUNT(*) FROM concept_tasks WHERE status='pending'").fetchone()[0]

    def save_doc(self, source_id, text, url, license_id, topic="general", modality="text"):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.db.execute(
            "SELECT id FROM documents WHERE source_id=? OR sha256=?", (source_id, digest)
        ).fetchone()
        if existing:
            return False
        # A stable held-out set is established from the first insertion and every 5th document.
        split = "val" if self.count("documents") % 5 == 0 else "train"
        self.db.execute(
            "INSERT INTO documents(source_id,text,sha256,url,license,modality,split,topic,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (source_id, text, digest, url, license_id, modality, split, topic, now()),
        )
        self.db.commit()
        return True

    def documents(self, split=None, verified_only=False):
        where = []
        params = []
        if split:
            where.append("split=?")
            params.append(split)
        if verified_only:
            where.append("topic IN ('verified:definition','verified:arithmetic')")
            where.append("license='SELF-GENERATED-VERIFIED'")
        sql = "SELECT * FROM documents"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self.db.execute(sql + " ORDER BY id", params).fetchall()

    def count(self, table):
        if table not in {"documents", "frontier", "experiences", "model_versions", "runs", "skills"}:
            raise ValueError("Invalid table")
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def get(self, key, fallback=None):
        result = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(result[0]) if result else fallback

    def put(self, key, value):
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
        self.db.commit()

    def append_focus_activity(self, profile, run_id, at_utc, kind,
                              *, phase="", detail="", url=None):
        """Append one factual worker event without a total-history limit.

        Idempotent event keys permit migration of the previous bounded
        recent_activity list while the older worker is still running.
        This is an audit trail, not a neural-training dataset.
        """
        payload = [
            str(profile), str(run_id), str(at_utc), str(kind),
            str(detail or ""), str(url or ""),
        ]
        event_key = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO focus_activity_events"
            "(event_key,profile,run_id,at_utc,kind,phase,detail,url)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (event_key, str(profile), str(run_id), str(at_utc),
             str(kind), str(phase or ""), str(detail or "")[:500],
             str(url)[:1400] if url else None),
        )
        self.db.commit()
        return bool(cur.rowcount)

    def backfill_focus_activity(self, profile, run=None):
        """Recover the available legacy 16-event snapshot, without duplicates.

        Pre-existing truncated events cannot be reconstructed from the
        snapshot; new runs retain their entire activity history here.
        """
        run = run if run is not None else self.get("focus_run_" + profile, {})
        run = run or {}
        if not run.get("run_id"):
            return
        events = run.get("recent_activity", [])
        # Polling Studio every second must not repeatedly write or lock
        # SQLite for an unchanged legacy snapshot.
        digest = hashlib.sha256(json.dumps(
            events, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        marker = "focus_activity_backfill_digest_" + str(run["run_id"])
        if self.get(marker) == digest:
            return
        for event in events:
            if not isinstance(event, dict) or not event.get("at_utc"):
                continue
            self.append_focus_activity(
                profile, run["run_id"], event["at_utc"],
                event.get("kind", "activity"), phase=event.get("phase", ""),
                detail=event.get("detail", ""), url=event.get("url"),
            )
        self.put(marker, digest)

    def focus_activity(self, profile, *, after=None, before=None, limit=150):
        """Cursor-paginate all saved events without truncating history.

        Missing cursor = latest; after = new messages; before = older.
        A per-response page size protects the browser but is not a retention
        or total view limit.
        """
        if after is not None and before is not None:
            raise ValueError("Supply either after or before, not both")
        if not isinstance(limit, int) or not 1 <= limit <= 250:
            raise ValueError("Activity page size must be 1..250")
        params = [str(profile)]
        if before is not None:
            if not isinstance(before, int) or before < 1:
                raise ValueError("Invalid before cursor")
            sql = ("SELECT * FROM focus_activity_events "
                   "WHERE profile=? AND id<? ORDER BY id DESC LIMIT ?")
            params.extend([before, limit + 1])
            direction = "older"
        elif after is not None:
            if not isinstance(after, int) or after < 0:
                raise ValueError("Invalid after cursor")
            sql = ("SELECT * FROM focus_activity_events "
                   "WHERE profile=? AND id>? ORDER BY id ASC LIMIT ?")
            params.extend([after, limit + 1])
            direction = "newer"
        else:
            sql = ("SELECT * FROM focus_activity_events "
                   "WHERE profile=? ORDER BY id DESC LIMIT ?")
            params.append(limit + 1)
            direction = "latest"
        rows = self.db.execute(sql, params).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        if direction in ("older", "latest"):
            rows.reverse()
        total = self.db.execute(
            "SELECT COUNT(*) FROM focus_activity_events WHERE profile=?",
            (str(profile),),
        ).fetchone()[0]
        return {
            "status": "ok", "profile": profile, "direction": direction,
            "events": [
                {key: row[key] for key in (
                    "id", "profile", "run_id", "at_utc", "kind",
                    "phase", "detail", "url",
                )} for row in rows
            ],
            "has_more": has_more, "total_events": int(total),
            "history_retention_limit": None,
        }

    def log(self, action, status, details):
        self.db.execute("INSERT INTO runs(action,status,details,created_at) VALUES(?,?,?,?)",
                        (action, status, json.dumps(details, ensure_ascii=False), now()))
        self.db.commit()

    def experience(self, domain, prompt, prediction, expected, success, verifier, model_sha=None):
        self.db.execute(
            "INSERT INTO experiences(domain,prompt,prediction,expected,success,verified_by,model_sha256,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (domain, prompt, prediction, expected, int(success), verifier, model_sha, now()),
        )
        self.db.commit()

    def save_skill(self, key, instructions, verifier, evidence_count):
        if evidence_count < 3:
            raise ValueError("A reusable skill needs three or more independently checked examples")
        self.db.execute(
            "INSERT INTO skills(key,instructions,evidence_count,verified_by,created_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "instructions=excluded.instructions, evidence_count=excluded.evidence_count",
            (key, instructions, int(evidence_count), verifier, now()),
        )
        self.db.commit()

    def verified_math_count(self, operator):
        rows = self.db.execute(
            "SELECT prompt FROM experiences WHERE domain='arithmetic' AND success=1 AND "
            "verified_by='python_integer_arithmetic'"
        ).fetchall()
        return sum(1 for row in rows if f" {operator} " in row["prompt"])

    def relevant_memories(self, query, limit=3):
        """Simple keyword retrieval for the earliest pre-semantic-learning phase."""
        import re
        keywords = set(re.findall(r"[\w]+", query.casefold()))
        if not keywords:
            return []
        candidates = []
        for row in self.db.execute("SELECT text,url FROM documents ORDER BY id DESC LIMIT 2000"):
            words = set(re.findall(r"[\w]+", row["text"].casefold()))
            score = len(keywords & words)
            if score:
                candidates.append((score, f"{row['text']} [source: {row['url']}]"))
        for row in self.db.execute("SELECT instructions FROM skills"):
            words = set(re.findall(r"[\w]+", row["instructions"].casefold()))
            score = len(keywords & words)
            if score:
                candidates.append((score + 1, row["instructions"]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [value for _, value in candidates[:limit]]

    def latest_promoted(self):
        return self.db.execute("SELECT * FROM model_versions WHERE promoted=1 ORDER BY id DESC LIMIT 1").fetchone()

    def insert_model(self, path, sha256, count, loss, baseline, promoted):
        self.db.execute(
            "INSERT INTO model_versions(path,sha256,train_documents,val_loss,baseline_val_loss,promoted,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (str(path), sha256, count, loss, baseline, int(promoted), now()),
        )
        self.db.commit()

    def status(self):
        return {
            "documents": self.count("documents"),
            "collected_web_urls": self.db.execute(
                "SELECT COUNT(*) FROM web_corpus_sources").fetchone()[0],
            "unique_cleaned_web_texts": self.db.execute(
                "SELECT COUNT(*) FROM web_corpus_texts").fetchone()[0],
            "research_pages": self.db.execute("SELECT COUNT(*) FROM page_observations").fetchone()[0],
            "research_only_pages": self.db.execute(
                "SELECT COUNT(*) FROM page_observations WHERE training_status='research_only'"
            ).fetchone()[0],
            "training_documents": len(self.documents("train")),
            "heldout_documents": len(self.documents("val")),
            "frontier_pending": self.db.execute("SELECT COUNT(*) FROM frontier WHERE status='pending'").fetchone()[0],
            "frontier_done": self.db.execute("SELECT COUNT(*) FROM frontier WHERE status='done'").fetchone()[0],
            "frontier_errors": self.db.execute("SELECT COUNT(*) FROM frontier WHERE status='error'").fetchone()[0],
            "reusable_skills": self.count("skills"),
            "verified_experiences": self.db.execute("SELECT COUNT(*) FROM experiences WHERE success=1").fetchone()[0],
            "web_pending_urls": self.db.execute("SELECT COUNT(*) FROM web_frontier WHERE status='pending'").fetchone()[0],
            "web_completed_urls": self.db.execute("SELECT COUNT(*) FROM web_frontier WHERE status='done'").fetchone()[0],
            "web_error_urls": self.db.execute("SELECT COUNT(*) FROM web_frontier WHERE status='error'").fetchone()[0],
            "concepts_pending": self.pending_concepts(),
            "research_questions_open": self.open_questions_count(),
            "research_questions_verified": self.db.execute(
                "SELECT COUNT(*) FROM research_questions WHERE status='verified'").fetchone()[0],
            "verified_answers_training_eligible": self.db.execute(
                "SELECT COUNT(*) FROM verified_answers WHERE training_eligible=1").fetchone()[0],
            "verified_neural_documents": len(self.documents(verified_only=True)),
            "verified_arithmetic_documents": self.db.execute(
                "SELECT COUNT(*) FROM documents WHERE topic='verified:arithmetic'").fetchone()[0],
            "unsupported_open_questions": self.unsupported_question_count(),
            "last_arithmetic_evaluation": self.get("last_arithmetic_evaluation"),
            "numeric_lab_latest": self.get("numeric_lab_latest"),
            "rule_lab_latest_candidate": self.get("rule_lab_latest_candidate"),
            "magnitude_lab_latest": self.get("magnitude_lab_latest"),
            "magnitude_lab_latest_candidate": self.get("magnitude_lab_latest_candidate"),
            "linear_lab_latest": self.get("linear_lab_latest"),
            "linear_lab_latest_candidate": self.get("linear_lab_latest_candidate"),
            "multiop_lab_latest": self.get("multiop_lab_latest"),
            "multiop_lab_latest_candidate": self.get("multiop_lab_latest_candidate"),
            "interaction_lab_latest": self.get("interaction_lab_latest"),
            "interaction_lab_latest_candidate": self.get("interaction_lab_latest_candidate"),
            "basis_lab_latest": self.get("basis_lab_latest"),
            "basis_lab_latest_candidate": self.get("basis_lab_latest_candidate"),
            "calibration_lab_latest": self.get("calibration_lab_latest"),
            "calibration_lab_latest_candidate": self.get("calibration_lab_latest_candidate"),
            "gradient_lab_latest": self.get("gradient_lab_latest"),
            "gradient_lab_latest_candidate": self.get("gradient_lab_latest_candidate"),
            "concept_lab_latest": self.get("concept_lab_latest"),
            "concept_lab_latest_candidate": self.get("concept_lab_latest_candidate"),
            "char_lab_latest": self.get("char_lab_latest"),
            "char_lab_latest_candidate": self.get("char_lab_latest_candidate"),
            "web_lab_latest": self.get("web_lab_latest"),
            "web_lab_latest_candidate": self.get("web_lab_latest_candidate"),
            "concepts_discovered": self.db.execute("SELECT COUNT(*) FROM concept_tasks").fetchone()[0],
            "latest_promoted_model": dict(self.latest_promoted()) if self.latest_promoted() else None,
        }
