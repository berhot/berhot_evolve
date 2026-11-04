import tempfile
import unittest
from pathlib import Path

from evolve.discovery import clean, discover, parse_entity
from evolve.store import Store
from evolve.settings import load_config


def wikidata_fixture(qid, next_qid=None):
    claim = []
    if next_qid:
        claim = [{"rank":"normal", "mainsnak":{"datavalue":{"value":{"entity-type":"item","id":next_qid}}}}]
    return {"entities":{qid:{
        "type":"item", "labels":{"en":{"value":f"Object {qid}"},"ar":{"value":"اختبار"}},
        "descriptions":{"en":{"value":f"An automatically discovered object number {qid}."}},
        "claims":{"P31":claim}}}}


class DiscoveryTests(unittest.TestCase):
    def test_entity_text_and_relationship(self):
        text, refs = parse_entity(wikidata_fixture("Q42", "Q55"), "Q42")
        self.assertIn("Object Q42", text)
        self.assertIn("Q55", refs)

    def test_cleaning(self):
        self.assertEqual(clean(" A   \n  B\x00 "), "A B")

    def test_automated_discovery_and_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config()
            config.update(storage_root=temp, seed_entities=["Q42"], discovery_per_cycle=2, daily_request_budget=10,
                          crawl_delay_seconds=0, max_linked_entities_per_source=2)
            store = Store(temp)
            def local_fetch(qid, ua, cap):
                return wikidata_fixture(qid, "Q55" if qid == "Q42" else None)
            first = discover(store, config, fetcher=local_fetch, sleep_fn=lambda t: None)
            self.assertEqual(first["saved"], 2)
            self.assertEqual(store.count("documents"), 2)
            again = discover(store, config, fetcher=local_fetch, sleep_fn=lambda t: None)
            self.assertEqual(again["saved"], 0)
            self.assertTrue((Path(temp)/"raw"/"wikidata"/"Q42.json").exists())
            store.close()
