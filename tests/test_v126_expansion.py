"""Offline v1.26 source expansion qualification and v1.22 host-filter regression."""
import tempfile
import unittest

from evolve.open_corpus import archive_page
from evolve.store import Store
from evolve.v122_evidence import evidence_report
from evolve.v126_expansion import expansion_report


def seed(store, url, sentence, *, license_id=None, noai=False, title="Article"):
    filler = " ".join(
        "Researchers carefully recorded scientific observations using "
        "independent instruments at station " + str(i) + " during each survey."
        for i in range(45))
    text = sentence + " " + filler
    result = archive_page(
        store, url, "<html><body>" + text + "</body></html>",
        text, title=title, noai=noai, declared_license=license_id,
        rights_url="fixture:self-authored" if license_id else None)
    assert result["status"] == "archived", result


class V126SourceTriageTests(unittest.TestCase):
    def test_same_topic_prioritized_but_never_auto_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                approved = "https://science-a.example.org/current"
                same = "https://science-b.example.org/current"
                new = "https://science-c.example.org/planet"
                suspect = "https://science-d.example.org/tutorial"
                noai = "https://science-e.example.org/current"
                seed(store, approved,
                     "A current is a movement of water through an ocean basin.",
                     license_id="CC0-1.0")
                seed(store, same,
                     "A current is a continuous movement of ocean water "
                     "across several coastal areas.")
                seed(store, new,
                     "A planet is a celestial body orbiting a star in space.")
                seed(store, suspect,
                     "Currents Tutorial is a section of the website showing "
                     "different diagrams of water.")
                seed(store, noai,
                     "A current is water moving through a coastal estuary.",
                     noai=True)
                before = store.db.total_changes
                report = expansion_report(store)
                self.assertEqual(store.db.total_changes, before)
                self.assertEqual(report["status"],
                                 "research_review_only_no_training")
                self.assertEqual(report["unknown_rights_sources_examined"], 3)
                self.assertEqual(report["noai_sources_excluded"], 1)
                self.assertEqual(report["already_approved_sources_examined"], 1)
                self.assertEqual(report[
                    "unapproved_sources_with_same_topic_candidates"], 1)
                review = report["queue_to_review"]
                self.assertEqual(review[0]["url"], same)
                self.assertEqual(review[0]["topics_already_present"],
                                 ["current"])
                self.assertEqual(review[0][
                    "topics_from_other_independent_hosts"], ["current"])
                self.assertFalse(review[0]["rights_approved_for_training"])
                self.assertEqual(review[1]["url"], new)
                self.assertEqual(review[1]["topics_not_yet_present"],
                                 ["planet"])
                self.assertEqual(review[2]["url"], suspect)
                self.assertEqual(review[2]["suspected_navigation_topics"],
                                 ["currents tutorial"])
                quote = review[0]["exact_source_preview"][0]
                self.assertTrue(quote["quote_contained_in_archived_text"])
                self.assertEqual(len(quote["quote_sha256"]), 64)
                self.assertEqual(report["new_training_pairs_created"], 0)
                self.assertFalse(report["reviewed_training_rights_automatically"])
                self.assertFalse(report["model_weights_updated"])
                self.assertNotIn(noai, [x["url"] for x in review])
            finally:
                store.close()

    def test_filter_and_no_auto_fabricated_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                seed(store, "https://x.example.org/info",
                     "Oceanographers collected observations during "
                     "a regular marine field survey.")
                seed(store, "https://y.example.org/planet",
                     "A planet is an object moving around a nearby star.")
                self.assertEqual(expansion_report(store)["filtered_host"], None)
                one = expansion_report(store, host="y.example.org")
                self.assertEqual(one["sources_examined"], 1)
                self.assertEqual(one["filtered_host"], "y.example.org")
                self.assertEqual(one["queue_to_review"][0]["url"],
                                 "https://y.example.org/planet")
                self.assertEqual(expansion_report(
                    store, host="x.example.org")["queue_to_review"][0][
                        "candidate_quotes"], 0)
                with self.assertRaises(ValueError):
                    expansion_report(store, host="https://x.example.org")
                with self.assertRaises(ValueError):
                    expansion_report(store, limit=0)
                with self.assertRaises(ValueError):
                    expansion_report(store, quote_limit=0)
            finally:
                store.close()

    def test_v122_filtered_host_is_not_overwritten_by_last_source(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                seed(store, "https://a.example.org/planet",
                     "A planet is a celestial body orbiting a star.")
                seed(store, "https://z.example.org/current",
                     "A current is a movement of ocean water.")
                full = evidence_report(store)
                self.assertIsNone(full["filtered_host"])
                only = evidence_report(store, host="a.example.org")
                self.assertEqual(only["filtered_host"], "a.example.org")
                self.assertEqual(only["sources_examined"], 1)
                self.assertEqual(only["sources"][0]["host"],
                                 "a.example.org")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
