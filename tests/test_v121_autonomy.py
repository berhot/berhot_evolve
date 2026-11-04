"""v1.21 autonomous cycle: real source locks, generated questions, bounded research.

Fixtures are original local science sentences, never pulled from web.
No fixture claims verified factual understanding.
"""
import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v120_grounded import grounded_answer
from evolve.v121_autonomy import (
    source_questions, curiosity_queue, build_lesson_bank, audit_quiz,
    plan, cycle, history, learning_report,
)
from evolve.v122_evidence import evidence_report

try:
    import torch
except ImportError:
    torch = None


SOURCES = [
    ("volcano.example.org", [
        "A volcano is an opening in Earth's crust where lava and gases reach the surface during eruptions.",
        "Lava is a type of molten rock that travels across a landscape after volcanic eruptions.",
        "A mineral is a naturally occurring substance with an ordered crystalline structure and specific composition.",
        "A mountain is a large natural landform rising above the surrounding surface due to geological forces.",
    ]),
    ("nasa.example.org", [
        "The Sun is made of super-hot, electrically charged gas called plasma.",
        "The Sun is the only star in our solar system.",
        "A planet is a celestial body orbiting a star with a sufficiently rounded shape from its gravity.",
        "A galaxy is a vast system of stars, gas, dust and other material held together by gravity.",
        "A comet is a small icy celestial body that may develop a visible tail when approaching a star.",
    ]),
    ("ocean.example.org", [
        "A tide is the regular rise and fall of the sea surface caused primarily by gravitational interactions.",
        "A current is a continual movement of water in oceans produced by forces including winds and density gradients.",
        "A reef is a shallow structure formed from coral or rock near the surface of a body of water.",
        "A lagoon is a shallow body of coastal water that is separated from the surrounding ocean by a barrier.",
    ]),
]


def populate(store, hosts=3):
    for host, facts in SOURCES[:hosts]:
        # Distinct per-page and per-passage observations are required by
        # the real v1.17 near-copy filter. Repeating one identical filler
        # is correctly excluded and must not masquerade as training data.
        filler = " ".join(
            "Researchers working at " + host.split(".")[0] +
            " measured scientific sample number " + str(i) +
            " with instrument version " + str(i % 7) +
            " to compare environmental observations and test their "
            "measurement hypotheses." for i in range(95))
        text = " ".join(facts) + " " + filler
        url = "https://" + host + "/science"
        assert archive_page(
            store, url, "<html><title>Science facts</title><body>" +
            text + "</body></html>", text, title="Science facts",
            declared_license="CC0-1.0", noai=False,
            rights_url="https://creative-commons.example.org/license"
        )["status"] == "archived"


class AutonomyTests(unittest.TestCase):
    def test_question_discovery_and_immutable_self_teaching(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                populate(store, hosts=2)
                store.add_concept_task("photon", "https://nasa.example.org/science",
                                       priority=90)
                q = source_questions(store, limit=6)
                self.assertIn("photon", [x["topic"] for x in q])
                self.assertNotIn("sun", [x["topic"] for x in q])
                before = plan(store, load_config())
                self.assertEqual(before["network_requests_made"], 0)
                self.assertFalse(before["neural_weights_updated"])
                bank = build_lesson_bank(store, write=True)
                self.assertGreaterEqual(bank["questions_created"], 12)
                self.assertTrue(Path(bank["manifest_path"]).is_file())
                self.assertEqual(
                    bank["manifest_sha256"],
                    Path(bank["manifest_path"]).stem.removeprefix("quiz-"))
                self.assertEqual(
                    build_lesson_bank(store, write=True)["manifest_path"],
                    bank["manifest_path"])
                audit = audit_quiz(store)
                self.assertEqual(audit["pass"], audit["tested"], audit)
                result = grounded_answer(store, "What is the Sun?")
                self.assertEqual(result["answer"],
                                 "The Sun is the only star in our solar system.")
            finally:
                store.close()

    def test_research_progression_skips_cached_and_cooldown_topics(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                populate(store, hosts=2)
                store.add_concept_task(
                    "photon", "https://nasa.example.org/science", priority=50)
                store.add_concept_task(
                    "nebula", "https://nasa.example.org/science", priority=40)
                store.add_concept_task(
                    "entropy", "https://nasa.example.org/science", priority=30)
                from evolve.v119_curiosity import _key
                import time
                store.put(_key("photon"), {
                    "status": "researched", "topic": "photon",
                    "answer": "A saved source excerpt, not training",
                    "sources": []})
                store.put(_key("nebula"), {
                    "status": "unresolved", "topic": "nebula",
                    "retry_after_epoch": time.time() + 7200,
                    "reason": "no_supported_passage"})
                progress = curiosity_queue(store, limit=8)
                self.assertEqual(
                    [x["topic"] for x in progress["questions"]],
                    ["entropy"])
                self.assertEqual(
                    progress["deferred_by_reason"]["previous_research_available"], 1)
                self.assertEqual(
                    progress["deferred_by_reason"]["retry_cooldown_active"], 1)
                self.assertEqual(
                    progress["status"], "questions_available")
                # Planning does not modify network quotas or training rights.
                self.assertEqual(progress["network_requests_made"], 0)
            finally:
                store.close()

    def test_evidence_qualification_excludes_unknown_and_tracks_hosts(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                populate(store, hosts=2)
                archive_page(
                    store, "https://unreviewed.example.org/glacier",
                    "<html><title>What is a glacier?</title></html>",
                    "A glacier is a large slow-moving mass of ice "
                    "formed from accumulated snow on land. " * 15,
                    title="What is a glacier?", declared_license=None,
                    noai=False)
                result = evidence_report(store)
                self.assertEqual(result["status"], "review_only")
                self.assertGreaterEqual(result["sources_examined"], 3)
                proposal = next(item for item in result["sources"]
                                if item["host"] == "unreviewed.example.org")
                self.assertEqual(proposal["status"],
                                 "research_only_requires_article_rights_review")
                self.assertFalse(proposal["neural_training_admitted_by_this_report"])
                self.assertFalse(proposal["source_rights_eligible_for_qa_extraction"])
                self.assertIn("glacier", proposal["extracted_candidate_topics"])
                self.assertEqual(
                    store.db.execute(
                        "SELECT rights_status FROM web_corpus_sources "
                        "WHERE url='https://unreviewed.example.org/glacier'"
                    ).fetchone()[0], "UNKNOWN")
                self.assertFalse(
                    result["claims"]["rights_approved_automatically"])
                self.assertFalse(
                    result["claims"]["unverified_extraction_is_truth"])
            finally:
                store.close()

    def test_offline_cycle_does_not_network_or_invent_train_test_hosts(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                populate(store, hosts=2)
                store.add_concept_task("photon", "https://nasa.example.org/science",
                                       priority=5)
                def fail_network(*args, **kwargs):
                    raise AssertionError("Offline mode must never research")
                report = cycle(store, load_config(), network=False,
                               max_topics=2, steps=4,
                               research_fn=fail_network)
                self.assertEqual(report["new_requests_this_cycle"], 0)
                self.assertFalse(report["new_neural_weights_trained"])
                self.assertFalse(report["understanding_proven"])
                self.assertNotEqual(report["heldout_readiness"]["status"], "ready")
                self.assertEqual(history(store)["recent_runs"][-1]["status"],
                                 "completed_with_training_blocker")
                compact = learning_report(store)
                self.assertEqual(compact["status"], "completed_with_training_blocker")
                self.assertEqual(compact["network_requests_in_last_cycle"], 0)
                self.assertFalse(compact["ranker"]["candidate_weights_trained"])
                self.assertGreaterEqual(
                    compact["missing_to_train_at_last_cycle"]["test"]["examples"], 2)
                self.assertFalse(compact["understanding_proven"])
            finally:
                store.close()

    def test_explicit_network_budget_and_unknown_rights_stays_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            try:
                populate(store, hosts=2)
                store.add_concept_task("photon", "https://nasa.example.org/science",
                                       priority=10)
                calls = []
                settings = copy.deepcopy(load_config())
                settings["daily_request_budget"] = 12
                day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                def fake_research(st, cfg, question):
                    calls.append((question, cfg["daily_request_budget"]))
                    key = "web_requests_" + day
                    st.put(key, int(st.get(key, 0)) + 1)
                    archive_page(
                        st, "https://unknown.example.org/photon",
                        "<html>Photon facts</html>", "Photon facts " * 100,
                        title="Photon facts", declared_license=None,
                        noai=False)
                    return {"status": "researched"}
                report = cycle(store, settings, network=True,
                               max_topics=1, max_requests=1, steps=4,
                               research_fn=fake_research)
                self.assertEqual(report["new_requests_this_cycle"], 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    store.db.execute("SELECT rights_status FROM web_corpus_sources "
                                     "WHERE url='https://unknown.example.org/photon'"
                                     ).fetchone()[0], "UNKNOWN")
                self.assertFalse(report["new_public_research_is_auto_authorized_for_training"])
                second = cycle(store, settings, network=True,
                               max_topics=1, max_requests=1, steps=4,
                               research_fn=fake_research)
                self.assertEqual(len(calls), 1)
                # Progress to NEW words instead of repeating the same
                # researched question on the next autonomous cycle.
                self.assertEqual(second["curiosity_question_count"], 0)
                self.assertEqual(second["new_requests_this_cycle"], 0)
                self.assertGreaterEqual(
                    second["curiosity_triage"]["deferred_by_reason"][
                        "already_attempted_today"], 1)
            finally:
                store.close()


@unittest.skipUnless(torch is not None, "PyTorch optional")
class NeuralCycleTests(unittest.TestCase):
    def test_real_ranker_trains_once_on_three_independent_hosts(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            settings = copy.deepcopy(load_config())
            settings["storage_root"] = root
            settings["train"]["max_seconds"] = 90
            try:
                populate(store, hosts=3)
                first = cycle(store, settings, network=False,
                              steps=10, seed=31)
                self.assertEqual(first["neural_training"]["status"], "trained",
                                 first["neural_training"])
                self.assertGreaterEqual(
                    first["neural_training"]["test_after"]["n"], 2)
                benchmark = first["neural_training"]["benchmark"]
                self.assertEqual(
                    benchmark["lexical_test"]["n"],
                    first["neural_training"]["test_after"]["n"])
                self.assertAlmostEqual(
                    benchmark["test_accuracy_delta_vs_lexical"],
                    round(
                        benchmark["neural_test_pair_accuracy"] -
                        benchmark["lexical_test"]["pair_accuracy"], 4))
                self.assertFalse(
                    benchmark["selected_model_using_test_results"])
                self.assertFalse(
                    benchmark["external_semantic_generalization_proven"])
                self.assertTrue(
                    benchmark["different_topic_easy_negative_pairs"])
                self.assertTrue(first["new_neural_weights_trained"])
                self.assertFalse(first["understanding_proven"])
                sha = first["neural_training"]["sha256"]
                second = cycle(store, settings, network=False,
                               steps=10, seed=31)
                self.assertEqual(second["neural_training"]["status"], "unchanged")
                self.assertEqual(
                    (store.get("v120_ranker_candidate") or {})["sha256"], sha)
                self.assertEqual(len(history(store)["recent_runs"]), 2)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
