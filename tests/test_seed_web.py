import tempfile
import unittest
from evolve.web_discovery import canonical_url, extract, top_concepts, discover_web, register_seed, WebSourceError, detect_declared_rights
from evolve.store import Store
from evolve.settings import load_config

PAGES={
  'https://learn.example.org/lesson': '<html><head><title>Beginner letters</title></head><body><h1>School lessons</h1>'
  '<p>Letters form words. Words form sentences. At school students learn words and letters.</p>'
  '<a href="/school">Read about school</a><a href="https://others.example.net/">Other site</a></body></html>',
  'https://learn.example.org/school': '<html><body><h1>School</h1><p>A school is a place for learning. A teacher helps students learn '
  'how to read, write, and practice numbers. Students learn reading and writing in their school.</p>'
  '<a href="/lesson">Lessons</a></body></html>',
}

class SeedWebTests(unittest.TestCase):
    def test_url_safety(self):
        for bad in ('http://learn.example.org', 'https://127.0.0.1/a', 'https://localhost/a',
                    'https://learn.example.org:8443/', 'https://u:pw@learn.example.org/a'):
            with self.subTest(bad=bad), self.assertRaises(WebSourceError):canonical_url(bad)
        self.assertEqual(canonical_url('https://learn.example.org/lesson?q=secret#abc'), 'https://learn.example.org/lesson')

    def test_html_extraction(self):
        text,title,links,noai=extract(PAGES['https://learn.example.org/lesson'])
        self.assertIn('school students learn',text.lower())
        self.assertIn('/school',links)
        self.assertFalse(noai)
        self.assertIn('school',top_concepts(text,'school',3))

    def test_seed_cruise_same_host_and_task_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            profile=register_seed(store,'https://learn.example.org/lesson','CC0-1.0',
                                  'https://learn.example.org/license','school')
            config=load_config()
            config.update(storage_root=tmp,crawl_delay_seconds=0,daily_request_budget=30)
            result=discover_web(store,config,limit=10,fetcher=lambda url:PAGES[url],sleep_fn=lambda _:None)
            self.assertEqual(result['saved'],2)
            self.assertEqual(store.count('documents'),2)
            self.assertTrue(store.db.execute('SELECT 1 FROM concept_tasks WHERE term=?',('school',)).fetchone())
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM web_frontier').fetchone()[0],2)
            self.assertEqual(result['errors'],0)
            again=discover_web(store,config,limit=10,fetcher=lambda url:PAGES[url],sleep_fn=lambda _:None)
            self.assertEqual(again['saved'],0)
            store.close()

    def test_automatic_license_detection(self):
        html='<html><head><link rel="license" href="https://creativecommons.org/publicdomain/zero/1.0/"></head><body>Lesson on words.</body></html>'
        self.assertEqual(detect_declared_rights(html,'https://learn.example.org/lesson')[0],'CC0-1.0')
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            p=register_seed(store,'https://learn.example.org/lesson',seed_html=html,focus='school')
            self.assertEqual(p['rights_mode'],'page_declaration')
            store.close()

    def test_license_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp)
            with self.assertRaises(WebSourceError):
                register_seed(store,'https://learn.example.org','UNKNOWN','https://learn.example.org/rights')
            store.close()

    def test_noai_signal(self):
        page='<html><head><meta name="robots" content="noai"></head><body>Example sentences about school.</body></html>'
        self.assertTrue(extract(page)[3])

if __name__=='__main__':unittest.main()
