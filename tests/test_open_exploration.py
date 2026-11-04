"""The explore command may read publicly accessible HTML without training rights."""
import tempfile
import unittest
from evolve.settings import load_config
from evolve.store import Store
from evolve.web_discovery import register_exploration, discover_web, WebSourceError, register_seed

START = 'https://start.example.org/lesson'
SECOND = 'https://other.example.net/school'
THIRD = 'https://start.example.org/math'
PAGES = {
    START: '<html><head><title>ABC lesson</title></head><body><p>School and classroom learning includes alphabet letters, words, books and teachers. Students read every day.</p><a href="https://other.example.net/school">more</a><a href="/math">math</a></body></html>',
    SECOND: '<html><head><title>Outside site</title></head><body><p>Schools contain teachers and pupils learning language, grammar, and reading through many different lessons.</p></body></html>',
    THIRD: '<html><head><title>Mathematics</title><link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/"></head><body><p>Mathematics introduces addition, numbers, multiplication, subtraction, fractions and new ideas to school students.</p></body></html>',
}

class OpenExplorationTests(unittest.TestCase):
    def test_research_only_keeps_content_out_of_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            settings=load_config()
            settings.update(storage_root=tmp,crawl_delay_seconds=0,daily_request_budget=100)
            register_exploration(store,START,focus='school',cross_host=True)
            result=discover_web(store,settings,limit=10,fetcher=lambda url: PAGES[url],sleep_fn=lambda _:None)
            self.assertEqual(result['research_observed'],3)
            self.assertEqual(result['saved'],1)
            self.assertEqual(store.status()['research_only_pages'],2)
            self.assertEqual(store.status()['training_documents']+store.status()['heldout_documents'],1)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM documents WHERE url=?',(START,)).fetchone()[0],0)
            self.assertGreater(store.status()['concepts_discovered'],0)
            # No verbatim text from the unclear-rights site in neural training.
            self.assertTrue(all('Outside site' not in row['text'] for row in store.documents()))
            store.close()

    def test_same_host_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            settings=load_config()
            settings.update(storage_root=tmp,crawl_delay_seconds=0,daily_request_budget=100)
            register_exploration(store,START,focus='school')
            result=discover_web(store,settings,limit=10,fetcher=lambda url: PAGES[url],sleep_fn=lambda _:None)
            self.assertEqual(result['research_observed'],2)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM web_frontier').fetchone()[0],2)
            store.close()

    def test_false_same_page_rights_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            with self.assertRaisesRegex(WebSourceError,'not, by itself'):
                register_seed(store,START,'AUTHORIZED-TRAINING',START)
            store.close()

if __name__=='__main__':
    unittest.main()
