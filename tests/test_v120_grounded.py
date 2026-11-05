"""v1.20: honest question answering from archived sources and real neural ranking.

Self-written science sentences for deterministic tests; never use network.
"""
import copy
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v120_grounded import (
    extract_definition, grounded_answer, inventory, readiness,
    topic_from_question, train_ranker, title_question, title_evidence,
    diagnose_corpus,
)

try:
    import torch
except ImportError:
    torch = None


SCIENCE = [
    ("earth-science.example.org", "geology", [
        "A volcano is an opening in the surface of a planet through which molten rock and gas can escape during an eruption.",
        "Lava is a type of molten rock that reaches the surface of Earth through openings in volcanic regions.",
        "A mineral is a naturally occurring solid material with a specific chemical composition and an organized structure.",
        "A mountain is a large natural rise of land that stands above nearby terrain and can be formed by tectonic forces.",
        "A crater is a bowl shaped depression on a planetary surface that can form through impacts or volcanic activity.",
    ]),
    ("space-science.example.org", "astronomy", [
        "The Sun is a star at the center of our solar system that produces energy by fusion of hydrogen in its core.",
        "The Moon is Earth's natural satellite that travels in orbit around our planet and reflects sunlight from its surface.",
        "A planet is a large celestial body that moves around a star and has enough gravity to be nearly round.",
        "A galaxy is a massive collection of stars, dust, gas, and dark matter bound together by gravity.",
        "A comet is a small icy object that can develop a bright coma when it approaches the central star.",
    ]),
    ("ocean-science.example.org", "oceanography", [
        "A reef is a ridge of rock or coral found near the surface of the sea that can shelter marine life.",
        "A tide is the periodic rise and fall of sea level caused mainly by gravitational attraction from the Moon and Sun.",
        "A current is a continuous movement of seawater that carries heat and dissolved material between different locations.",
        "A wave is a disturbance that transports energy through water without transporting the same water mass across oceans.",
        "A lagoon is a shallow body of water separated from a larger sea by a barrier or coral formation.",
    ]),
]


def prepare_fixture(store):
    urls = []
    for host, subject, facts in SCIENCE:
        # Each document is long enough for the existing strict-host corpus
        # filter, but only the five explicit definitions are QA candidates.
        filler = " ".join(
            f"Researchers in field section {subject} measured object sample "
            f"{i} with multiple scientific instruments and careful repeated "
            f"observations over several seasons. The collected measurement "
            f"{i} supports cautious interpretation in this field report."
            for i in range(85))
        text = " ".join(facts) + " " + filler
        url = f"https://{host}/articles/{subject}"
        result = archive_page(
            store, url,
            "<html><title>Science study</title><body>" + text + "</body></html>",
            text, title="Science study", noai=False,
            declared_license="CC0-1.0", rights_url="fixture:self-authored")
        assert result["status"] == "archived"
        urls.append(url)
    return urls


class DefinitionUnitTests(unittest.TestCase):
    def test_exact_definitions_not_incidental_sun_sentence(self):
        self.assertEqual(topic_from_question("sun"), "sun")
        self.assertEqual(topic_from_question("What is the Sun?"), "sun")
        self.assertEqual(topic_from_question("Define a volcano"), "volcano")
        self.assertIsNone(topic_from_question("Why does the Sun shine?"))
        self.assertIsNone(topic_from_question("What happens when a star dies?"))
        self.assertEqual(extract_definition(SCIENCE[1][2][0]), "sun")
        self.assertEqual(extract_definition("The Sun is a star."), "sun")
        # A verbatim citation is not necessarily a useful scientific
        # definition; these frequent noise patterns must not self-teach.
        self.assertIsNone(extract_definition(
            "The result is a collection of scientific observations."))
        self.assertIsNone(extract_definition(
            "The word photosphere is a word used to describe a stellar layer."))
        self.assertIsNone(topic_from_question("What is result?"))
        self.assertIsNone(topic_from_question("Define word photosphere"))
        self.assertIsNone(extract_definition(
            "Our Sun is roughly midway through its main sequence stage."))
        self.assertIsNone(extract_definition(
            "Media Redoubt volcano with minor ash eruption."))

    def test_grounded_answer_is_exact_quote_and_sourced(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                prepare_fixture(store)
                response = grounded_answer(store, "What is the Sun?")
                self.assertEqual(response["status"], "answered", response)
                self.assertEqual(response["answer"], SCIENCE[1][2][0])
                self.assertEqual(response["sources"][0]["url"],
                                 "https://space-science.example.org/articles/astronomy")
                self.assertFalse(response["neural_weights_updated"])
                self.assertFalse(response["neural_ranker_used"])
                self.assertEqual(
                    grounded_answer(store, "Why does the sun shine?")["status"],
                    "unsupported")
                self.assertEqual(
                    grounded_answer(store, "What is electricity?")["status"],
                    "no_definition")
                summary = inventory(store)
                self.assertGreaterEqual(summary["source_derived_definitions"], 12)
                self.assertGreaterEqual(summary["unique_topics"], 12)
            finally:
                store.close()

    def test_real_noaa_navigation_is_never_accepted_as_a_definition(self):
        # This reproduces the user's actual v1.22.3 false answers, including
        # a valid source title and two true, independent source definitions.
        chrome = (
            "Toggle navigation Toggle search National Ocean Service "
            "National Oceanic and Atmospheric Administration "
        )
        navigation_current = chrome + "What is a current?"
        navigation_estuary = chrome + "What is an estuary?"
        current_definition = (
            'The term "current" describes the motion of the ocean.')
        estuary_definition = (
            "Estuaries and their surrounding wetlands are bodies of water "
            "usually found where rivers meet the sea.")
        self.assertIsNone(title_evidence(navigation_current, "current"))
        self.assertIsNone(title_evidence(navigation_estuary, "estuary"))
        self.assertEqual(
            title_evidence(navigation_current + " " + current_definition,
                           "current"), current_definition)
        self.assertEqual(
            title_evidence(navigation_estuary + " " + estuary_definition,
                           "estuary"), estuary_definition)
        self.assertIsNone(extract_definition(navigation_current))

        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                for topic, definition, navigation in (
                    ("current", current_definition, navigation_current),
                    ("estuary", estuary_definition, navigation_estuary),
                ):
                    filler = " ".join(
                        "Oceanographers collected seawater measurements "
                        "during independent field surveys at site " + str(i) +
                        " to document coastal science accurately."
                        for i in range(80))
                    text = navigation + " " + definition + " " + filler
                    result = archive_page(
                        store, "https://ocean.example.org/facts/" + topic,
                        "<html><body>" + text + "</body></html>",
                        text, title="What is a " + topic + "?",
                        noai=False, declared_license="CC0-1.0",
                        rights_url="fixture:self-authored")
                    self.assertEqual(result["status"], "archived")
                # A legacy ranker may have passed its previous contaminated
                # test; neither its old source-quote answers nor its
                # checkpoint's acceptance flag can override new evidence rules.
                old = {"accepted_by_validation": True,
                       "sha256": "historic-invalid-checkpoint",
                       "checkpoint": "/tmp/historic.pt"}
                store.put("v120_ranker_latest", old)
                for topic, expected in (
                    ("current", current_definition),
                    ("estuary", estuary_definition),
                ):
                    response = grounded_answer(store, "What is " + topic + "?")
                    self.assertEqual(response["status"], "answered", response)
                    self.assertEqual(response["answer"], expected)
                    self.assertFalse(response["neural_ranker_used"])
                    self.assertNotIn("Toggle", response["answer"])
            finally:
                store.close()

    def test_title_question_without_real_definition_is_not_training_data(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                navigation = (
                    "Toggle navigation Toggle search National Ocean Service "
                    "National Oceanic and Atmospheric Administration "
                    "What is a current?")
                filler = ("Oceanographers measured ocean water during "
                          "careful scientific surveys. " * 80)
                text = navigation + " " + filler
                archive_page(
                    store, "https://ocean.example.org/facts/current",
                    "<html><body>" + text + "</body></html>", text,
                    title="What is a current?", declared_license="CC0-1.0",
                    rights_url="fixture:self-authored")
                response = grounded_answer(store, "What is a current?")
                self.assertEqual(response["status"], "no_definition")
                self.assertEqual(inventory(store)["source_derived_definitions"], 0)
            finally:
                store.close()

    def test_question_title_grounds_exact_answer_without_invention(self):
        self.assertEqual(title_question("What is a current? - NOAA"), "current")
        self.assertEqual(title_question("What is an estuary?"), "estuary")
        self.assertIsNone(title_question("NASA Science: Our Solar System"))
        first = ("The term current describes the regular motion of ocean "
                 "water around the planet as observed in scientific studies.")
        second = ("Estuaries and their surrounding wetlands are bodies of "
                  "water where fresh rivers mix with salty ocean water.")
        self.assertEqual(title_evidence(first, "current"), first)
        self.assertEqual(title_evidence(second, "estuary"), second)
        self.assertIsNone(title_evidence(
            "Our Sun is roughly midway through its main sequence stage.",
            "sun"))
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                filler = " ".join(
                    "A scientific instrument collected measurements of "
                    "seawater during a careful field experiment." for _ in range(90))
                for topic, content in (("current", first), ("estuary", second)):
                    url = "https://marine-science.example.org/facts/" + topic
                    result = archive_page(
                        store, url,
                        "<html><title>What is " + topic + "?</title><body>" +
                        content + filler + "</body></html>",
                        content + " " + filler,
                        title="What is " + topic + "?",
                        declared_license="CC0-1.0",
                        rights_url="fixture:self-authored")
                    self.assertEqual(result["status"], "archived")
                ans = grounded_answer(store, "What is a current?")
                self.assertEqual(ans["answer"], first)
                self.assertEqual(ans["sources"][0]["url"],
                                 "https://marine-science.example.org/facts/current")
                self.assertTrue(ans["sources"][0]["source_sha256"])
                report = diagnose_corpus(store, question="current")
                self.assertEqual(report["status"], "diagnosed")
                self.assertEqual(report["topic"], "current")
                self.assertEqual(len(report["source_reports"]), 2)
                self.assertGreaterEqual(
                    sum(x["extracted_definitions"]
                        for x in report["source_reports"]), 2)
                self.assertFalse(ans["neural_weights_updated"])
            finally:
                store.close()

    def test_title_alone_does_not_create_answer_or_train(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                page = ("Our Sun is roughly midway through its main sequence "
                        "stage, and it changes gradually over time. ") * 20
                archive_page(
                    store, "https://science.example.org/sun",
                    "<html><title>What is the Sun?</title>" + page + "</html>",
                    page, title="What is the Sun?",
                    declared_license="CC0-1.0", rights_url="fixture:self-authored")
                self.assertEqual(grounded_answer(store, "sun")["status"],
                                 "no_definition")
            finally:
                store.close()

    def test_unknown_rights_never_used(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                article = SCIENCE[1][2][0] * 30
                archive_page(store, "https://unreviewed.example.org/articles/sun",
                             "<html>" + article + "</html>",
                             article, title="Sun", declared_license=None)
                self.assertEqual(grounded_answer(store, "sun")["status"],
                                 "no_definition")
                self.assertEqual(inventory(store)["source_derived_definitions"], 0)
            finally:
                store.close()


class HoldoutTests(unittest.TestCase):
    def test_strict_host_partitions(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                prepare_fixture(store)
                result = readiness(store)
                self.assertEqual(result["status"], "ready", result)
                hosts = [set(result["source_hosts"][split])
                         for split in ("train", "val", "test")]
                self.assertFalse(hosts[0] & hosts[1])
                self.assertFalse(hosts[0] & hosts[2])
                self.assertFalse(hosts[1] & hosts[2])
                self.assertTrue(all(v >= 4 for v in result["counts"].values()))
            finally:
                store.close()


@unittest.skipUnless(torch is not None, "PyTorch optional; neural CI runs these")
class NeuralRankerTests(unittest.TestCase):
    def test_real_pytorch_ranker_and_immutable_source_citations(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["train"]["max_seconds"] = 60
            try:
                prepare_fixture(store)
                earlier = store.get("v118_qa_latest")
                before = grounded_answer(store, "What is the Sun?")
                result = train_ranker(store, settings, steps=12, seed=13)
                self.assertEqual(result["status"], "trained", result)
                self.assertEqual(result["actual_training_steps"], 12)
                self.assertTrue(result["previous_checkpoints_unchanged"])
                self.assertGreaterEqual(result["source_derived_examples"]["train"], 4)
                self.assertGreaterEqual(result["source_derived_examples"]["test"], 2)
                self.assertTrue(result["checkpoint"].endswith(".pt"))
                self.assertIsNotNone(result["test_after"]["loss"])
                self.assertEqual(store.get("v118_qa_latest"), earlier)
                # Whether learned ranker was accepted or not, no output is
                # invented; only a literal quote from verified-source bytes.
                after = grounded_answer(store, "What is the Sun?")
                self.assertEqual(before["answer"], after["answer"])
                self.assertTrue(after["sources"])
                self.assertFalse(after["neural_weights_updated"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
