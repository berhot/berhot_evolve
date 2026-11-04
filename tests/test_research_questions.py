"""Source-grounded curiosity questions are durable but not verified facts."""
import tempfile
import unittest

from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import register_exploration, discover_web

URL = "https://learning.example.org/alphabet"
HTML = """<html><head><title>English Alphabet</title></head>
<body><h1>English Alphabet Letters</h1>
<p>The English alphabet has letters. An alphabet contains symbols used for
writing words. Vowels and consonants are kinds of alphabet letters.
Alphabet letters form words and words form sentences. Learners use letters
to read and write messages.</p></body></html>"""


class ResearchQuestionsTests(unittest.TestCase):
    def test_questions_persist_without_training_or_verified_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            settings = load_config()
            settings.update(storage_root=folder, crawl_delay_seconds=0, daily_request_budget=30)
            register_exploration(store, URL)
            report = discover_web(store, settings, limit=1,
                                  fetcher=lambda _: HTML, sleep_fn=lambda _: None)
            self.assertEqual(report["research_observed"], 1)
            self.assertGreater(report["questions_created"], 0)
            self.assertEqual(report["research_questions_open"], report["questions_created"])
            questions = store.list_research_questions(30)
            self.assertTrue(any(q["concept"] == "alphabet" for q in questions))
            self.assertTrue(all(q["status"] == "open" and q["source_url"] == URL
                                for q in questions))
            status = store.status()
            self.assertEqual(status["research_questions_verified"], 0)
            self.assertEqual(status["verified_experiences"], 0)
            self.assertEqual(status["training_documents"], 0)
            self.assertEqual(status["heldout_documents"], 0)
            self.assertEqual(store.pending_concepts(), 0)
            store.close()

            store = Store(folder)
            self.assertEqual(len(store.list_research_questions(100)), len(questions))
            repeat = discover_web(store, settings, limit=1,
                                  fetcher=lambda _: HTML, sleep_fn=lambda _: None)
            self.assertEqual(repeat["questions_created"], 0)
            self.assertEqual(store.open_questions_count(), len(questions))
            store.close()

    def test_v04_reviewed_topics_requeued_for_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.add_concept_task("alphabet", URL, priority=15, source_title="Alphabet")
            store.mark_concept_task("alphabet", "reviewed")
            store.close()
            reopened = Store(folder)
            self.assertEqual(reopened.pending_concepts(), 1)
            self.assertEqual(reopened.open_questions_count(), 0)
            reopened.close()

    def test_question_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertTrue(store.add_research_question("alphabet", "What is alphabet?", URL))
            self.assertFalse(store.add_research_question("alphabet", "What is alphabet?", URL))
            self.assertEqual(store.open_questions_count(), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
