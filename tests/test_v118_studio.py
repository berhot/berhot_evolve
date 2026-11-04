"""BERHOT v1.18 integration: source chat, verified lessons, actual QA gradients,
controlled one-at-a-time jobs, local dashboard access policy.
All test data is self-authored and does not require internet access.
"""
import copy
import http.client
import json
import tempfile
import threading
import time
import unittest
from hashlib import sha256
from pathlib import Path

from evolve.open_corpus import archive_page
from evolve.settings import load_config
from evolve.store import Store
from evolve.v117 import prepare
from evolve.v118_core import (ask, add_qa_example, qa_readiness,
                              train_qa, qa_neural_response, curriculum)
from evolve.v118_studio import (LearningController, _handler, ThreadingHTTPServer)

try:
    import torch
except ImportError:
    torch = None


TOPICS = [
    ("volcano", "pressure volcanic magma hot lava ash mantle rock mineral",
     "Volcanic pressure affects rock and magma in geological investigations."),
    ("stars", "radiation stellar hydrogen gravity nuclear orbit energy",
     "Stellar gravity affects hydrogen and radiation in space investigations."),
    ("corals", "reef marine ocean polyp algae ecology tropical",
     "Marine coral growth affects reefs and tropical ocean habitats."),
]


def fixture(folder):
    store = Store(folder)
    corpus = {}
    for i, (topic, words, sentence) in enumerate(TOPICS):
        url = f"https://fictional-{i}.example.org/articles/{topic}"
        snippets = []
        for n in range(175):
            snippets.append(
                f"Observation {n} of {topic} investigates {words} "
                f"with independent measurement {n*n+4}. {sentence} "
                f"Further evaluation number {n} collects evidence from "
                "repeatable scientific study.")
        article = " ".join(snippets)
        archive_page(store, url, "<html><title>" + topic + "</title><body>" +
                     article + "</body></html>",
                     article, title=topic, declared_license="CC0-1.0",
                     rights_url="fixture:self-authored")
        corpus[f"fictional-{i}.example.org"] = (url, article, topic, sentence)
    return store, corpus


class SourceAndLessonTests(unittest.TestCase):
    def test_grounded_chat_with_citation_not_marked_neural(self):
        with tempfile.TemporaryDirectory() as root:
            store, corpus = fixture(root)
            try:
                output = ask(store, "Tell me about volcanic magma", "sources")
                self.assertEqual(output["mode"], "source_lookup_not_neural")
                self.assertFalse(output["learning"])
                self.assertTrue(output["sources"])
                self.assertIn("fictional-0.example.org", output["sources"][0]["url"])
                none = ask(store, "purple bananas starlight dragons", "sources")
                self.assertEqual(none["sources"], [])
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    ask(store, "hello", "external_model")
                with self.assertRaisesRegex(ValueError, "1..280"):
                    ask(store, " ", "sources")
                p = curriculum(store)
                self.assertTrue(p["ready_for_independent_training"], p)
                self.assertEqual(p["eligible_articles"], 3)
            finally:
                store.close()

    def test_lessons_require_a_rights_checked_exact_quote(self):
        with tempfile.TemporaryDirectory() as root:
            store, corpus = fixture(root)
            try:
                url, article, topic, sentence = next(iter(corpus.values()))
                self.assertEqual(add_qa_example(
                    store, "What causes magma?", sentence, url)["status"], "added")
                self.assertEqual(add_qa_example(
                    store, "What causes magma?", sentence, url)["status"], "duplicate")
                with self.assertRaisesRegex(ValueError, "exact quotation"):
                    add_qa_example(
                        store, "What causes magma?", "This answer is invented by me.", url)
                with self.assertRaisesRegex(ValueError, "rights-reviewed"):
                    add_qa_example(store, "What causes magma?", sentence,
                                   "https://missing.example.org/article")
            finally:
                store.close()


@unittest.skipIf(torch is None, "Actual CPU PyTorch required")
class SupervisedQATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_train_separate_qa_and_verify_immutable_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            store, corpus = fixture(root)
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            cfg["train"]["max_seconds"] = 75
            try:
                frozen = prepare(store)
                self.assertEqual(frozen["status"], "prepared", frozen)
                host_to_split = {host: split
                                 for split, hosts in frozen["source_hosts"].items()
                                 for host in hosts}
                for host, (url, text, topic, sentence) in corpus.items():
                    split = host_to_split[host]
                    needed = 9 if split == "train" else 3
                    for i in range(needed):
                        # Human-reviewed synthetic question whose answer is
                        # a literal sentence from the same synthetic article.
                        answer = (
                            f"Observation {i} of {topic} investigates " +
                            TOPICS[int(host.split("-")[1][0])][1] +
                            f" with independent measurement {i*i+4}.")
                        self.assertIn(answer, text)
                        q = f"What was measured in {topic} observation {i}?"
                        result = add_qa_example(store, q, answer, url)
                        self.assertEqual(result["status"], "added")
                ready = qa_readiness(store)
                self.assertEqual(ready["status"], "ready", ready)
                progress = []
                out = train_qa(store, cfg, steps=4,
                               progress=lambda phase, n, t, loss:
                               progress.append((phase, n, t)))
                self.assertEqual(out["status"], "trained", out)
                self.assertEqual(out["actual_training_steps"], 4)
                self.assertTrue(progress)
                self.assertTrue(Path(out["qa_manifest_path"]).is_file())
                self.assertEqual(sha256(
                    Path(out["qa_manifest_path"]).read_bytes()).hexdigest(),
                    out["qa_manifest_sha256"])
                self.assertIn("test_answer_loss_after", out)
                self.assertIsNone(store.get("v117_latest"))
                self.assertEqual(len(store.get("v118_qa_history")), 1)
                if out["accepted_by_validation"]:
                    response = qa_neural_response(store, "What was measured?")
                    self.assertEqual(response["mode"], "qa_neural_raw")
                else:
                    no_model = qa_neural_response(store, "What was measured?")
                    self.assertEqual(no_model["status"], "no_accepted_qa_model")
                # Tampering with an explicitly chosen candidate is rejected.
                store.put("v118_qa_latest", out)
                with Path(out["qa_manifest_path"]).open("ab") as handle:
                    handle.write(b"unexpected")
                with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                    qa_neural_response(store, "What was measured?")
            finally:
                store.close()


class StudioControllerTests(unittest.TestCase):
    def test_local_status_and_bounded_job_without_eligible_data(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            controller = LearningController(cfg, interval=10)
            try:
                status = controller.status()
                self.assertEqual(status["archived_pages"], 0)
                self.assertEqual(status["rights_eligible_articles"], 0)
                self.assertFalse(status["auto"]["enabled"])
                started = controller.submit("language", steps=2)
                self.assertEqual(started["status"], "started")
                controller.active.join(timeout=15)
                self.assertFalse(controller.active.is_alive())
                s = controller.status()
                self.assertEqual(s["job"]["status"], "skipped")
                self.assertEqual(s["job"]["result_summary"]["status"],
                                 "skipped")
                self.assertEqual(len(s["job_history"]), 1)
                controller.set_auto(True)
                self.assertTrue(controller.status()["auto"]["enabled"])
                controller.set_auto(False)
                self.assertFalse(controller.status()["auto"]["enabled"])
                with self.assertRaises(ValueError):
                    controller.submit("arbitrary_python", steps=10)
                with self.assertRaises(ValueError):
                    controller.submit("language", steps=10000)
            finally:
                controller.shutdown()

    def test_http_same_origin_key_local_only(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = copy.deepcopy(load_config())
            cfg["storage_root"] = root
            control = LearningController(cfg, interval=30)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), _handler(control, "verified-test-secret",
                                         "http://127.0.0.1:0"))
            # Bind the handler to its actual ephemeral port for the Host guard.
            # Used only in this synthetic HTTP policy test.
            server.RequestHandlerClass = _handler(
                control, "verified-test-secret",
                f"http://127.0.0.1:{server.server_port}")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/api/status")
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                data = json.loads(response.read())
                self.assertEqual(data["status"], "online")
                conn.close()

                data = json.dumps({"question": "Are you learning?",
                                   "mode": "sources"})
                origin = f"http://127.0.0.1:{server.server_port}"
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=10)
                conn.request("POST", "/api/chat", body=data,
                             headers={"Origin": origin,
                                      "Content-Type": "application/json"})
                forbidden = conn.getresponse()
                self.assertEqual(forbidden.status, 403)
                forbidden.read()
                conn.close()
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=10)
                conn.request("POST", "/api/chat", body=data,
                             headers={"Origin": origin,
                                      "X-Evolve-Key": "verified-test-secret",
                                      "Content-Type": "application/json"})
                good = conn.getresponse()
                self.assertEqual(good.status, 200)
                self.assertEqual(json.loads(good.read())["mode"],
                                 "source_lookup_not_neural")
                conn.close()
                conn = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=10)
                conn.request("GET", "/api/status",
                             headers={"Host": "evil.example.org"})
                self.assertEqual(conn.getresponse().status, 403)
                conn.close()
            finally:
                server.shutdown()
                server.server_close()
                control.shutdown()
                worker.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
