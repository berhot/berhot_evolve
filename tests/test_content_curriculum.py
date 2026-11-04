"""Concept choices must come from observed pages, never a manually added focus."""
import tempfile
import unittest
from evolve.concept_mining import rank_page_concepts
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import discover_web, register_exploration

URL = 'https://lessons.example.org/alphabet'
HTML = '''<html><head><title>English Alphabet</title></head><body>
<h1>English alphabet and letters</h1>
<p>English alphabet letters are used to form words. Each letter is part of the
alphabet. Vowels and consonants are groups of letters.</p>
<a href="/other">Next lesson</a></body></html>'''


class ContentCurriculumTests(unittest.TestCase):
    def test_titles_and_headings_rank_observed_topics(self):
        topics = rank_page_concepts(
            'English alphabet letters are used. Alphabet letters form words.',
            'English Alphabet', ['English alphabet and letters'], limit=4)
        terms = [term for term, _ in topics]
        self.assertIn('alphabet', terms)
        self.assertNotIn('school', terms)
        self.assertEqual(topics[0][0], 'alphabet')

    def test_focus_cannot_invent_concepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            settings = load_config()
            settings.update(storage_root=tmp, crawl_delay_seconds=0, daily_request_budget=50)
            profile = register_exploration(store, URL, focus='school')
            self.assertNotIn('focus', profile)
            report = discover_web(store, settings, limit=1, fetcher=lambda _: HTML,
                                  sleep_fn=lambda _: None)
            self.assertEqual(report['research_observed'], 1)
            terms = [row['term'] for row in store.list_concept_tasks()]
            self.assertIn('alphabet', terms)
            self.assertNotIn('school', terms)
            self.assertTrue(all(row['source_url'] == URL for row in store.list_concept_tasks()))
            self.assertTrue(store.concept_sources('alphabet'))
            store.close()

    def test_v03_schema_migrates_without_deleting_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.add_concept_task('alphabet', URL)
            store.close()
            store = Store(tmp)
            self.assertEqual(store.list_concept_tasks()[0]['term'], 'alphabet')
            store.close()


if __name__ == '__main__':
    unittest.main()
