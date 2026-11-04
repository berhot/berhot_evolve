"""Offline regression checks for the v1.19 curiosity pipeline."""
import copy
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v118_core import qa_readiness
from evolve.v119_curiosity import (_topic, best_passage, candidates,
                                   smart_lookup, research, sentence_lessons)


def fixture(store):
    for i, topic in enumerate(("volcano", "star", "coral")):
        article = " ".join(
            f"Scientific observation {n} of {topic} describes natural "
            f"phenomena measured with careful instruments and repeated "
            f"tests by independent observers over multiple years. "
            f"The same observations {n} help researchers compare "
            f"temperature and motion in documented experiments."
            for n in range(160))
        archive_page(
            store, f"https://science-{i}.example.org/research/{topic}",
            "<html><title>Science</title><body>"+article+"</body></html>",
            article, title=topic, declared_license="CC0-1.0",
            rights_url="fixture:self-authored")


class CuriosityTests(unittest.TestCase):
    def test_topic_and_local_caption_quality(self):
        self.assertEqual(_topic("moon"), "moon")
        self.assertEqual(_topic("What is the Moon?"), "moon")
        self.assertEqual(_topic("What is a volcano?"), "volcano")
        self.assertIsNone(_topic("how does the moon change through the year?"))
        self.assertIn("https://science.nasa.gov/moon/facts/", candidates("moon"))
        s = ("Media Redoubt volcano with minor ash eruption. "
             "A volcano is a geological opening through which melted rock "
             "and hot gases rise to the planetary surface during eruptions.")
        self.assertIn("volcano is a geological opening",
                      best_passage(s, "volcano"))

    def test_finds_missing_moon_and_caches_read_only_source(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            cfg["daily_request_budget"] = 3
            cfg["crawl_delay_seconds"] = 0
            called = []
            def fake(url):
                called.append(url)
                if url == "https://science.nasa.gov/moon/facts/":
                    return ("<html><title>Moon Facts</title><body>"
                            "<p>The Moon is Earth's only natural satellite, "
                            "which orbits our planet and influences ocean tides "
                            "through gravity.</p></body></html>")
                raise ValueError("No fixture for this URL")
            try:
                self.assertEqual(smart_lookup(store,"moon")["status"],
                                 "needs_research")
                result = research(store,cfg,"moon",fetcher=fake,
                                  sleep_fn=lambda _:None)
                self.assertEqual(result["status"],"researched",result)
                self.assertIn("natural satellite",result["answer"])
                row = store.db.execute(
                    "SELECT rights_status FROM web_corpus_sources "
                    "WHERE url='https://science.nasa.gov/moon/facts/'").fetchone()
                self.assertEqual(row[0],"UNKNOWN")
                self.assertEqual(smart_lookup(store,"moon")["mode"],
                                 "research_memory_not_neural")
                self.assertEqual(research(store,cfg,"moon",fetcher=fake)
                                 ["status"],"cached_researched")
                self.assertEqual(len(called),2)
                self.assertIsNone(store.get("v117_latest"))
            finally:
                store.close()

    def test_unapproved_content_not_neural_training(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                text=("A Moon is a natural satellite revolving around "
                      "another astronomical body. ")*100
                archive_page(store,"https://unknown.example.org/moon",
                             "<html>"+text+"</html>",text,title="Moon")
                self.assertEqual(sentence_lessons(store)["new_lessons"],0)
            finally:
                store.close()

    def test_self_study_generates_partitioned_reproducible_examples(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root)
            try:
                fixture(store)
                first=sentence_lessons(store,per_host=14)
                self.assertEqual(first["status"],"generated",first)
                self.assertGreaterEqual(first["new_lessons"],30,first)
                second=sentence_lessons(store,per_host=14)
                self.assertEqual(second["new_lessons"],0)
                self.assertEqual(qa_readiness(store)["status"],"ready")
                for item in store.get("v119_auto_lessons"):
                    self.assertEqual(item["source_provenance"],
                        "auto_exact_source_continuation_not_verified_qa")
            finally:
                store.close()


if __name__=="__main__":
    unittest.main()
