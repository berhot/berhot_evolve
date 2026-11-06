"""Idle/failed collection is evidence, never a prepared text dataset."""
import json
from pathlib import Path
import tempfile
import unittest

from evolve.store import Store
from evolve.settings import load_config
from evolve.focus_overnight import init_state, cycle
from evolve.plain_corpus import (begin_extraction_run, capture_extraction,
    finalize_extraction_run, finish_collection_cycle, organize_extraction_runs)


class EmptyCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.addCleanup(self.store.close)
        self.settings = load_config()
        self.settings.update(storage_root=self.tmp.name, crawl_delay_seconds=0)
        self.profile = "saudi-history-ar"

    def test_idle_cycle_does_not_create_empty_text_bundle_or_claim_saved_files(self):
        state = init_state(self.store, self.profile)
        state['_queue'] = []
        result = cycle(self.store, self.settings, state, pages=1, train_steps=1,
                       fetcher=lambda _: self.fail("No page should be fetched"))
        bundle = result['extraction_bundle']
        self.assertEqual(bundle['status'], 'no_extraction')
        self.assertIsNone(bundle['directory'])
        self.assertFalse((self.store.root / 'corpus_runs').exists())
        events = self.store.focus_activity(self.profile)['events']
        self.assertTrue(any(e['kind'] == 'corpus_batch_empty' for e in events))
        self.assertFalse(any(e['kind'] == 'corpus_batch_prepared' for e in events))
        self.assertTrue(Path(bundle['audit_log']).is_file())

    def test_failed_fetch_is_not_an_extraction_and_keeps_last_populated_pointer(self):
        state = init_state(self.store, self.profile)
        state['_queue'] = [{'url': 'https://source.example.org/saudi-history', 'depth': 0}]
        state['last_extraction_bundle'] = {'directory': 'earlier-populated-batch', 'extracted_pages': 1}
        def fail(_):
            raise OSError('unavailable')
        result = cycle(self.store, self.settings, state, pages=1, train_steps=1, fetcher=fail)
        self.assertEqual(result['extraction_bundle']['status'], 'no_extraction')
        self.assertEqual(result['extraction_bundle']['extracted_pages'], 0)
        self.assertEqual(state['last_extraction_bundle']['directory'], 'earlier-populated-batch')
        self.assertFalse((self.store.root / 'corpus_runs').exists())

    def test_finish_without_read_writes_audit_and_no_fake_training_text(self):
        bundle = finish_collection_cycle(self.store, 'no-read', self.profile,
                                         reason='search_unavailable', attempted=0)
        self.assertFalse((self.store.root / 'corpus_runs' / 'no-read').exists())
        lines = Path(bundle['audit_log']).read_text().splitlines()
        self.assertEqual(json.loads(lines[0])['reason'], 'search_unavailable')

    def test_archiving_is_reversible_preserves_manifest_and_skips_live_or_nonempty_data(self):
        empty = begin_extraction_run(self.store, 'old-empty', self.profile)
        finalize_extraction_run(self.store, 'old-empty')
        original = (empty/'manifest.json').read_bytes()
        live = begin_extraction_run(self.store, 'live-unfinished', self.profile)
        populated = begin_extraction_run(self.store, 'populated', self.profile)
        capture_extraction(self.store, 'populated', 1, 'https://source.example.org/article', 'title',
                           '<article>نص عربي محفوظ</article>', 'نص عربي محفوظ')
        finalized = finalize_extraction_run(self.store, 'populated')
        text = (populated/'raw.txt').read_bytes()
        result = organize_extraction_runs(self.store)
        moved = Path(result['empty_batch_archive'])/'old-empty'
        self.assertEqual(result['empty_batches_archived'], 1)
        self.assertEqual((moved/'manifest.json').read_bytes(), original)
        self.assertTrue(live.exists())
        self.assertEqual((populated/'raw.txt').read_bytes(), text)
        self.assertEqual(Path(result['latest_with_text']).resolve(), populated.resolve())
        self.assertEqual(organize_extraction_runs(self.store)['empty_batches_archived'], 0)

    def test_inconsistent_zero_page_manifest_with_real_text_is_never_archived(self):
        directory = begin_extraction_run(self.store, 'inconsistent', self.profile)
        finalize_extraction_run(self.store, 'inconsistent')
        (directory/'raw.txt').write_text('Actual text must survive')
        result = organize_extraction_runs(self.store)
        self.assertEqual(result['empty_batches_archived'], 0)
        self.assertTrue(directory.exists())


if __name__ == '__main__':
    unittest.main()
