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

    def next_web_url(self):
        return self.db.execute("SELECT * FROM web_frontier WHERE status='pending' ORDER BY depth,discovered_at LIMIT 1").fetchone()

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

    def next_open_questions(self, limit=5, cooldown_seconds=3600, supported_only=False):
        """Never spend the evidence-fetch budget on question formats we cannot answer."""
        import time
        cutoff = time.time() - max(0, float(cooldown_seconds))
        support_filter = "AND question LIKE 'What is %?' " if supported_only else ""
        return [dict(row) for row in self.db.execute(
            "SELECT id,concept,question,source_url,source_title FROM research_questions "
            "WHERE status='open' AND (last_attempt_at IS NULL OR last_attempt_at<=?) "
            + support_filter + "ORDER BY attempts, id LIMIT ?",
            (cutoff, max(1, min(int(limit), 50))))]

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
            "concepts_discovered": self.db.execute("SELECT COUNT(*) FROM concept_tasks").fetchone()[0],
            "latest_promoted_model": dict(self.latest_promoted()) if self.latest_promoted() else None,
        }
