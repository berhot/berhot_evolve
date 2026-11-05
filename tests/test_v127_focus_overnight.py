"""Offline regression tests for bounded Arabic overnight focus-agent safety."""
import copy
import tempfile
import unittest

from evolve.focus_data import (
    get_profile, arabic_ratio, topic_relevance,
    arabic_curiosity, review_leads,
)
from evolve.focus_overnight import plan, init_state, cycle, report
from evolve.focus_training import readiness, train_if_eligible
from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v117_rights import review_rights

try:
    import torch
except ImportError:
    torch = None


def arabic_text(subject="الأولى", size=65):
    main = (
        "تأسست الدولة السعودية " + subject +
        " على يد أحد الحكام في عام 1727م في مدينة الدرعية. "
    )
    filler = (
        "تاريخ المملكة العربية السعودية والدرعية يعرض أحداث "
        "الدولة السعودية ومراحل التأسيس في سياق دراسات التاريخ. "
    )
    return main + filler * size


def fake_page(text):
    return "<html><head><title>تاريخ المملكة العربية السعودية</title>" \
           "</head><body><article>" + text + "</article></body></html>"


class FocusDataTests(unittest.TestCase):
    def test_profile_and_real_arabic_content_derived_questions(self):
        p = get_profile("saudi-history-ar")
        self.assertEqual(len(p["allowed_hosts"]), 3)
        self.assertEqual(len(p["seeds"]), 8)
        self.assertTrue(all(u.startswith("https://") for u in p["seeds"]))
        text = arabic_text()
        self.assertGreater(arabic_ratio(text), .9)
        quality = topic_relevance(text, "الدرعية", p)
        self.assertTrue(quality["topical"])
        self.assertTrue(quality["language_ok"])
        findings = arabic_curiosity(text, "https://www.spa.gov.sa/123",
                                    "a"*64, p)
        self.assertGreater(len(findings), 0)
        self.assertEqual(findings[0]["status"],
                         "unverified_research_question")
        self.assertFalse(findings[0]["candidate_is_a_true_answer"])
        self.assertEqual(findings[0]["year_mentions"], ["1727"])
        self.assertEqual(review_leads([]), [])
        with self.assertRaises(ValueError):
            get_profile("unapproved-topic")
        with self.assertRaises(ValueError):
            plan(extra_urls=["https://example.com/anything"])

    def test_offline_fetcher_archive_dedup_report_and_no_unlicensed_training(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                cfg = copy.deepcopy(load_config())
                cfg["storage_root"] = folder
                cfg["daily_request_budget"] = 10
                cfg["crawl_delay_seconds"] = 0
                state = init_state(store, "saudi-history-ar", max_pages=3)
                self.assertEqual(len(state["_queue"]), 8)
                fetched = []
                html = fake_page(arabic_text())
                def fetch(url):
                    fetched.append(url)
                    return html
                first = cycle(store, cfg, state, pages=3, train_steps=2,
                              fetcher=fetch, sleep_fn=lambda sec: None)
                self.assertEqual(first["status"], "cycle_completed")
                self.assertEqual(state["pages_attempted"], 3)
                self.assertEqual(state["pages_archived"], 3)
                self.assertEqual(state["new_unique_pages"], 1)
                self.assertEqual(state["exact_duplicate_pages"], 2)
                self.assertEqual(state["training_steps_this_run"], 0)
                self.assertEqual(state["training_status"],
                                 "skipped_no_eligible_training_corpus")
                self.assertGreater(state[
                    "candidate_questions_created_this_run"], 0)
                summary = report(store)
                self.assertEqual(summary["last_run"]["pages_attempted"], 3)
                self.assertEqual(summary[
                    "rights_eligible_arabic_training"]["rights_eligible_arabic_pages"], 0)
                self.assertGreater(summary[
                    "cumulative_source_derived_unverified_questions"], 0)
                self.assertEqual(len(fetched), 3)
                # The shared daily request count is honored, not bypassed.
                self.assertEqual(store.get("web_requests_" +
                    __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc).strftime("%Y-%m-%d")), 3)
                self.assertEqual(
                    store.db.execute("SELECT COUNT(*) FROM web_corpus_sources "
                                     "WHERE rights_status='UNKNOWN'").fetchone()[0], 3)
            finally:
                store.close()

    def test_no_network_or_training_in_plan_and_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                profile = plan()
                self.assertTrue(profile[
                    "research_only_until_per_article_rights_review"])
                self.assertEqual(profile["network_requests_without_explicit_run"], 0)
                summary = report(store)
                self.assertEqual(summary["status"], "not_started")
                self.assertIsNone(summary["latest_accepted_focus_model"])
                self.assertFalse(readiness(store, "saudi-history-ar")["status"] == "ready")
            finally:
                store.close()

    def test_rights_are_not_automatically_admitted_by_page_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                host = "saudipedia.com"
                url = "https://" + host + "/test"
                text = arabic_text(size=70)
                archive_page(store, url, fake_page(text), text,
                             title="تاريخ المملكة", declared_license="PUBLIC-DOMAIN",
                             rights_url="https://rights.example.org/")
                self.assertEqual(readiness(
                    store, "saudi-history-ar")["rights_eligible_arabic_pages"], 0)
            finally:
                store.close()


@unittest.skipUnless(torch is not None, "Optional PyTorch integration")
class FocusTrainTests(unittest.TestCase):
    def test_actual_training_only_after_three_independent_human_reviews(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            try:
                cfg = copy.deepcopy(load_config())
                cfg["storage_root"] = folder
                urls = [
                    "https://saudipedia.com/saudi-history-test",
                    "https://www.spa.gov.sa/saudi-history-test",
                    "https://www.mofa.gov.sa/ar/ksa/Pages/history-test.aspx",
                ]
                for i, url in enumerate(urls):
                    # Each publisher must contribute DISTINCT text. Identical
                    # hashes across hosts are deduplicated before holdout.
                    text = arabic_text(size=80) + (
                        " السجل التاريخي المصدر المستقل والتوثيق. " * (i + 1))
                    archive_page(
                        store, url, fake_page(text), text,
                        title="تاريخ المملكة العربية السعودية",
                        declared_license=None)
                self.assertEqual(readiness(
                    store, "saudi-history-ar")["rights_eligible_arabic_pages"], 0)
                self.assertEqual(train_if_eligible(
                    store, "saudi-history-ar", steps=2)["steps"], 0)
                for url in urls:
                    result = review_rights(
                        store, url, "PUBLIC-DOMAIN",
                        "https://rights.example.org/policy",
                        confirmed=True)
                    self.assertEqual(result["status"], "review_recorded")
                pre = readiness(store, "saudi-history-ar")
                self.assertEqual(pre["status"], "ready", pre)
                result = train_if_eligible(store, "saudi-history-ar",
                                           steps=3, seed=17)
                self.assertEqual(result["status"], "trained", result)
                self.assertEqual(result["steps"], 3)
                self.assertGreater(result["trainable_parameters"], 1000)
                self.assertTrue(result["sha256"])
                self.assertFalse(result["selected_on_test"])
                self.assertFalse(result["understands_saudi_history"])
                repeat = train_if_eligible(store, "saudi-history-ar",
                                           steps=3, seed=17)
                self.assertEqual(repeat["status"], "skipped_unchanged_approved_data")
                self.assertEqual(repeat["steps"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
