"""Command-line interface: python3 -m evolve ..."""
import argparse
import json

from .settings import load_config
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description="BERHOT EVOLVE MVP — autonomous research learner")
    parser.add_argument("--config", help="Custom settings JSON path")
    sub = parser.add_subparsers(dest="command", required=True)
    discover_parser = sub.add_parser("discover", help="Continue registered seed website discovery")
    discover_parser.add_argument("--limit", type=int, default=None)
    discover_parser.add_argument("--legacy-wikidata", action="store_true", help="Old API connector requiring EVOLVE_CONTACT")
    web_start = sub.add_parser("web-start", help="Register a permitted public HTML seed URL; no provider API")
    web_start.add_argument("--seed-url", required=True)
    web_start.add_argument("--license", choices=["CC0-1.0","PUBLIC-DOMAIN","AUTHORIZED-TRAINING"])
    web_start.add_argument("--rights-evidence-url")
    web_start.add_argument("--focus", default="", help=argparse.SUPPRESS)  # legacy flag ignored
    web_start.add_argument("--limit", type=int, default=12)
    explore = sub.add_parser('explore', help='Read public HTML for concepts; no training-rights assertion needed')
    explore.add_argument('--seed-url', required=True)
    explore.add_argument('--focus', default='', help=argparse.SUPPRESS)  # legacy flag ignored
    explore.add_argument('--limit', type=int, default=8)
    explore.add_argument('--cross-host', action='store_true', help='Follow public links across domains within crawl budgets')
    focus_plan = sub.add_parser("focus-plan", help="Preview isolated Arabic Saudi-history overnight research scope without networking")
    focus_plan.add_argument("--profile", choices=["saudi-history-ar"], default="saudi-history-ar")
    focus_report = sub.add_parser("focus-report", help="Read-only morning progress and true training-weight statistics")
    focus_report.add_argument("--profile", choices=["saudi-history-ar"], default="saudi-history-ar")
    focus_run = sub.add_parser("focus-overnight", help="Opt-in bounded local overnight Arabic research agent; needs Mac awake")
    focus_run.add_argument("--profile", choices=["saudi-history-ar"], default="saudi-history-ar")
    focus_run.add_argument("--network", action="store_true", help="Required explicit consent for robots-compliant site access")
    focus_run.add_argument("--hours", type=float, default=8.0)
    focus_run.add_argument("--until-stopped", action="store_true",
        help="Keep agent process alive until Ctrl+C, SIGTERM or dashboard Stop; research remains capped per UTC day")
    focus_stop = sub.add_parser("focus-stop", help="Request graceful stop of the running Saudi-history worker")
    focus_stop.add_argument("--profile", choices=["saudi-history-ar"], default="saudi-history-ar")
    focus_run.add_argument("--interval-minutes", type=int, default=15)
    focus_run.add_argument("--max-pages", type=int, default=24)
    focus_run.add_argument("--pages-per-cycle", type=int, default=3)
    focus_run.add_argument("--train-steps", type=int, default=40)
    sub.add_parser("status", help="Show persistent learner status")
    sub.add_parser("model-audit", help="Read-only actual local knowledge, saved models, autonomy and dependency inventory")
    sub.add_parser("corpus-status", help="Show collected web pages, exact dedup, rights metadata and training separation")
    corpus_seed = sub.add_parser("corpus-seed", help="Queue 1..20 public HTTPS sites without requiring a CC0 license")
    corpus_seed.add_argument("--urls", nargs="+", required=True)
    corpus_seed.add_argument("--same-host", action="store_true",
                             help="Do not follow links across different hosts")
    corpus_crawl = sub.add_parser("corpus-crawl", help="Bounded existing-queue HTML extraction and independent raw/clean corpus archive")
    corpus_crawl.add_argument("--limit", type=int, default=5)
    corpus_crawl.add_argument("--host", default=None,
                              help="Only crawl already-queued URLs on this exact host, e.g. oceanservice.noaa.gov")
    retry_parser = sub.add_parser("retry-url", help="Retry one failed URL after correcting its connection problem")
    retry_parser.add_argument("--url", required=True)

    errors_parser = sub.add_parser("errors", help="Show exact failed website URLs and stored error reasons")
    errors_parser.add_argument("--limit", type=int, default=10)
    concept_parser = sub.add_parser("concepts", help="Show discovered concepts ranked by page evidence")
    concept_parser.add_argument("--limit", type=int, default=20)
    question_parser = sub.add_parser("questions", help="Show stored unanswered source-grounded research questions")
    question_parser.add_argument("--limit", type=int, default=20)
    answer_parser = sub.add_parser("answers", help="Show corroborated answers with independent source proofs")
    answer_parser.add_argument("--limit", type=int, default=20)
    sub.add_parser("research-health", help="Read-only UTC quota, supported question cooldown and eligible evidence diagnostics")
    research_parser = sub.add_parser("research", help="Seek independent evidence for open questions")
    research_parser.add_argument("--limit", type=int, default=5)
    research_parser.add_argument("--concept", default=None,
                                 help="Spend research budget only on a specific eligible concept")
    research_plan = sub.add_parser("research-plan", help="Offline preview of the exact concept's source candidates and publisher groups")
    research_plan.add_argument("--concept", required=True)
    sub.add_parser("v117-sources", help="List archived page URLs, metadata and manual rights-review candidates")
    rights_review = sub.add_parser("v117-review-rights", help="Attest rights evidence for ONE archived public article; never automatic")
    rights_review.add_argument("--url", required=True)
    rights_review.add_argument("--license", choices=["CC0-1.0", "PUBLIC-DOMAIN"], required=True)
    rights_review.add_argument("--evidence-url", required=True)
    rights_review.add_argument("--confirm-manual-review", action="store_true",
        help="I personally reviewed this exact article, publisher ownership, no-AI signs and linked rights policy")
    curiosity = sub.add_parser("v119-research", help="Research a missing topic using bounded robots-compliant public HTML sources")
    curiosity.add_argument("--question", required=True)
    sub.add_parser("v119-inventory", help="Inspect curated source-grounded research and automatic lessons")
    sub.add_parser("v119-self-study", help="Mine reading exercises from rights-reviewed local sources (no web calls)")
    study_train = sub.add_parser("v119-study-train", help="Create exact-source reading lessons and train a separate neural QA model when ready")
    study_train.add_argument("--steps", type=int, default=60)
    grounded = sub.add_parser("v120-ask", help="Answer a simple definition from an exact approved source quote (not neural generation)")
    grounded.add_argument("--question", required=True)
    sub.add_parser("v120-inventory", help="Count source-extracted definition candidates and neural ranker")
    gap = sub.add_parser("v120-diagnose", help="Show which archived sources lack definition examples, including exact topic evidence nearby")
    gap.add_argument("--question", default=None, help="Optional topic to inspect exact source passages, such as 'sun'")
    sub.add_parser("v120-readiness", help="Check host-held-out definition question ranking data")
    ranker = sub.add_parser("v120-train", help="Train independent PyTorch evidence ranker on source-derived definition pairs")
    ranker.add_argument("--steps", type=int, default=120)
    ranker.add_argument("--seed", type=int, default=31415)
    study_plan = sub.add_parser("v121-plan", help="Preview autonomous questions, source-backed lessons and neural test readiness (offline)")
    study_plan.add_argument("--max-topics", type=int, default=6)
    study_cycle = sub.add_parser("v121-cycle", help="Run bounded autonomous curiosity -> research -> local lessons -> PyTorch train -> evaluation")
    study_cycle.add_argument("--network", action="store_true", help="Explicitly allow robots-compliant HTTPS research with daily quota; without it all steps stay offline")
    study_cycle.add_argument("--max-topics", type=int, default=6)
    study_cycle.add_argument("--max-requests", type=int, default=6)
    study_cycle.add_argument("--steps", type=int, default=120)
    study_cycle.add_argument("--seed", type=int, default=31415)
    sub.add_parser("v121-history", help="Show prior autonomous cycle results and saved question-bank manifest")
    sub.add_parser("v121-report", help="Compact last-cycle scientific learning report and next source-host requirements")
    qualifier = sub.add_parser("v122-evidence", help="Read-only rights and QA-evidence host qualification report; never admits training data")
    qualifier.add_argument("--limit", type=int, default=80)
    qualifier.add_argument("--host", default=None, help="Inspect one publisher host, e.g. oceanservice.noaa.gov")
    evaluator = sub.add_parser("v124-eval", help="Read-only current citation/abstention audit and real neural-vs-lexical inference competition coverage")
    evaluator.add_argument("--limit", type=int, default=30)
    pairs = sub.add_parser("v125-pairs", help="Read-only real same-topic approved quote pairs needing human relevance review")
    pairs.add_argument("--limit", type=int, default=40)
    label = sub.add_parser("v125-label", help="Operator-attest immutable same-topic better/worse source quote preference; DEVELOPMENT only")
    label.add_argument("--question", required=True)
    label.add_argument("--better-url", required=True)
    label.add_argument("--better-sha", required=True)
    label.add_argument("--worse-url", required=True)
    label.add_argument("--worse-sha", required=True)
    label.add_argument("--confirm-human-review", action="store_true")
    reviewed = sub.add_parser("v125-score", help="Read-only lexical vs neural results on explicitly human-reviewed development pairs (not blind holdout)")
    reviewed.add_argument("--limit", type=int, default=100)
    expansion = sub.add_parser("v126-expansion", help="Read-only rights-aware exact-quote evidence expansion review queue")
    expansion.add_argument("--host", default=None, help="Optional exact source host")
    expansion.add_argument("--limit", type=int, default=80)
    expansion.add_argument("--quote-limit", type=int, default=3)
    studio = sub.add_parser("studio", help="Launch local live web dashboard with chat and controlled learning")
    studio.add_argument("--port", type=int, default=8787)
    studio.add_argument("--interval", type=int, default=3600,
                        help="Seconds between optional automatic dataset checks")
    sub.add_parser("v118-plan", help="Inspect rights-cleared curriculum growth and learning targets")
    sub.add_parser("v118-qa-readiness", help="Inspect human-approved source-backed QA training examples")
    qa_add = sub.add_parser("v118-qa-add", help="Add human-reviewed source-quoted QA training example")
    qa_add.add_argument("--question", required=True)
    qa_add.add_argument("--answer", required=True)
    qa_add.add_argument("--source-url", required=True)
    qa_train = sub.add_parser("v118-qa-train", help="Train separate answer-only model on verified reviewed QA examples")
    qa_train.add_argument("--steps", type=int, default=60)
    qa_train.add_argument("--seed", type=int, default=31415)
    chat = sub.add_parser("v118-chat", help="Talk to actual local neural weights or search approved local sources")
    chat.add_argument("--question", required=True)
    chat.add_argument("--mode", choices=["sources", "char", "byte_bpe", "qa"],
                      default="sources")
    sub.add_parser("v117-plan", help="Audit available rights-declared training sources and missing hosts")
    sub.add_parser("v117-prepare", help="Freeze and verify strict three-host train/validation/test partitions")
    v117_train = sub.add_parser("v117-train", help="Train separate from-scratch char and original byte-BPE models")
    v117_train.add_argument("--steps", type=int, default=120)
    v117_train.add_argument("--seed", type=int, default=31415)
    v117_train.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    v117_train.add_argument("--bpe-vocab", type=int, default=320)
    sub.add_parser("v117-evaluate", help="Read-only checkpoint-verified independent-host and generation evaluation")
    v117_smoke = sub.add_parser("v117-smoke", help="One-command 3-host SYNTHETIC smoke test; no user checkpoint writes")
    v117_smoke.add_argument("--steps", type=int, default=8)
    web_inventory = sub.add_parser("web-inventory", help="Curate archived articles and report exclusion reasons without training")
    web_inventory.add_argument("--include-unknown-rights", action="store_true")
    web_freeze = sub.add_parser("web-freeze", help="Build SHA256-frozen article train/val/test byte shards; no neural training")
    web_freeze.add_argument("--include-unknown-rights", action="store_true")
    web_freeze.add_argument("--diagnostic-within-document", action="store_true")
    char_lab = sub.add_parser("char-lab", help="Train BERHOT Unicode character tokenizer + separate 128-context Transformer, optionally resume accepted weights")
    char_lab.add_argument("--steps", type=int, default=400)
    char_lab.add_argument("--seed", type=int, default=31415)
    char_lab.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    char_lab.add_argument("--include-unknown-rights", action="store_true")
    char_lab.add_argument("--diagnostic-within-document", action="store_true")
    char_lab.add_argument("--resume", action="store_true", help="Continue accepted Unicode checkpoint if same frozen data and model config")
    char_lab.add_argument("--context", type=int, default=128)
    char_lab.add_argument("--embedding-dim", type=int, default=64)
    char_lab.add_argument("--layers", type=int, default=2)
    char_lab.add_argument("--heads", type=int, default=4)
    char_lab.add_argument("--vocab-limit", type=int, default=512)
    char_eval = sub.add_parser("char-evaluate", help="Verify saved Unicode weights/tokenizer/frozen dataset then reevaluate without gradients")
    char_eval.add_argument("--accepted", action="store_true")
    char_predict = sub.add_parser("char-predict", help="Sample Unicode characters from trained model weights, no source lookup")
    char_predict.add_argument("--prompt", required=True)
    char_predict.add_argument("--new-chars", type=int, default=100)
    char_predict.add_argument("--temperature", type=float, default=0.85)
    char_predict.add_argument("--seed", type=int, default=42)
    char_predict.add_argument("--top-k", type=int, default=30)
    char_predict.add_argument("--greedy", action="store_true")
    char_predict.add_argument("--accepted", action="store_true")
    char_compare = sub.add_parser("char-compare", help="Compare two SHA-verified Unicode checkpoints using identical frozen windows and decoding seeds")
    char_compare.add_argument("--reference-sha256", required=True)
    char_compare.add_argument("--prompt", default="Energy is")
    char_compare.add_argument("--new-chars", type=int, default=100)
    char_compare.add_argument("--temperature", type=float, default=0.85)
    char_compare.add_argument("--top-k", type=int, default=30)
    char_compare.add_argument("--seed", type=int, default=42)
    char_compare.add_argument("--accepted", action="store_true")
    web_lab = sub.add_parser("web-lab", help="One-shot frozen web corpus to trained neural checkpoint + heldout evaluation")
    web_lab.add_argument("--steps", type=int, default=60)
    web_lab.add_argument("--seed", type=int, default=31415)
    web_lab.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    web_lab.add_argument("--include-unknown-rights", action="store_true")
    web_lab.add_argument("--diagnostic-within-document", action="store_true")
    web_evaluate = sub.add_parser("web-evaluate", help="Check frozen shard and neural checkpoint hashes then re-evaluate without training")
    web_evaluate.add_argument("--accepted", action="store_true")
    web_predict = sub.add_parser("web-predict", help="Generate from saved web-trained neural weights, not database retrieval")
    web_predict.add_argument("--prompt", required=True)
    web_predict.add_argument("--new-bytes", type=int, default=120)
    web_predict.add_argument("--accepted", action="store_true")
    web_predict.add_argument("--decoding", choices=["greedy", "sample"],
                             default="greedy")
    web_predict.add_argument("--temperature", type=float, default=0.9)
    web_predict.add_argument("--seed", type=int, default=42)
    web_diagnose = sub.add_parser("web-diagnose", help="Compare SHA-verified web model to train-only byte unigram and detect generation collapse")
    web_diagnose.add_argument("--accepted", action="store_true")
    web_diagnose.add_argument("--prompt", default="Energy is")
    web_diagnose.add_argument("--new-bytes", type=int, default=80)
    web_diagnose.add_argument("--temperature", type=float, default=0.9)
    web_diagnose.add_argument("--seed", type=int, default=42)
    learn_parser = sub.add_parser("learn", help="Research and train only on verified eligible answers")
    learn_parser.add_argument("--limit", type=int, default=10)
    learn_parser.add_argument("--force", action="store_true", help="Ignore new verified example threshold")
    learn_parser.add_argument("--practice-arithmetic", action="store_true",
                              help="Add separately labeled Python-verified math exercises before training")
    learn_parser.add_argument("--practice-limit", type=int, default=32)
    practice_parser = sub.add_parser("practice", help="Generate independently verified arithmetic QA exercises")
    practice_parser.add_argument("--limit", type=int, default=32)
    practice_parser.add_argument("--skill", choices=["mixed", "single-digit-add"], default="mixed")
    math_parser = sub.add_parser("math-finetune", help="Train a separate neural arithmetic answerer with answer-only loss")
    math_parser.add_argument("--skill", choices=["mixed", "single-digit-add"], default="mixed")
    math_parser.add_argument("--steps", type=int, default=400)
    math_parser.add_argument("--force", action="store_true", help="Train longer without new arithmetic examples")
    numeric_parser = sub.add_parser("numeric-lab", help="Train/test a separate structured digit-addition neural baseline")
    numeric_parser.add_argument("--steps", type=int, default=400)
    numeric_parser.add_argument("--seed", type=int, default=42)
    numeric_predict_parser = sub.add_parser("numeric-predict", help="Predict two digits using learned numeric-lab parameters")
    numeric_predict_parser.add_argument("--a", type=int, required=True)
    numeric_predict_parser.add_argument("--b", type=int, required=True)
    rule_parser = sub.add_parser("rule-lab", help="Train digit-pair MLP without built-in numeric summation")
    rule_parser.add_argument("--steps", type=int, default=800)
    rule_parser.add_argument("--seed", type=int, default=31415)
    rule_parser.add_argument("--hidden", type=int, default=64)
    rule_predict_parser = sub.add_parser("rule-predict", help="Predict with saved generic neural arithmetic classifier")
    rule_predict_parser.add_argument("--a", type=int, required=True)
    rule_predict_parser.add_argument("--b", type=int, required=True)
    magnitude = sub.add_parser("magnitude-lab", help="Train numerical-input MLP without a hard-coded sum")
    magnitude.add_argument("--seed", type=int, default=31415)
    magnitude.add_argument("--steps", type=int, default=800)
    magnitude.add_argument("--hidden", type=int, default=32)
    magnitude_prediction = sub.add_parser("magnitude-predict", help="Predict using saved numerical-input neural model")
    magnitude_prediction.add_argument("--a", type=int, required=True)
    magnitude_prediction.add_argument("--b", type=int, required=True)
    magnitude_prediction.add_argument("--candidate", action="store_true", help="Inspect rejected model instead of accepted")
    magnitude_study = sub.add_parser("magnitude-study", help="Run predeclared seeds and report all test results")
    magnitude_study.add_argument("--seeds", nargs="+", type=int, default=[31415, 42, 1001])
    magnitude_study.add_argument("--steps", type=int, default=800)
    magnitude_study.add_argument("--hidden", type=int, default=32)
    ood = sub.add_parser("magnitude-extrapolate", help="Test frozen v1.1 weights on unseen 10..29 operands")
    ood.add_argument("--model-seed", type=int, default=None,
                     help="Choose a stored model by training seed; default latest")
    ood.add_argument("--count", type=int, default=20, help="Examples in each OOD group (1..50)")
    ood.add_argument("--probe-seed", type=int, default=20261104)
    ood_study = sub.add_parser("magnitude-extrapolation-study",
                               help="Audit multiple existing frozen magnitude checkpoints")
    ood_study.add_argument("--model-seeds", nargs="+", type=int,
                           default=[31415, 42, 1001])
    ood_study.add_argument("--count", type=int, default=20)
    ood_study.add_argument("--probe-seed", type=int, default=20261104)
    linear_parser = sub.add_parser("linear-lab", help="Learn non-saturating numeric coefficients and audit OOD")
    linear_parser.add_argument("--seed", type=int, default=31415)
    linear_parser.add_argument("--steps", type=int, default=500)
    linear_parser.add_argument("--count", type=int, default=20)
    linear_parser.add_argument("--probe-seed", type=int, default=20261104)
    linear_study = sub.add_parser("linear-study", help="Compare frozen OOD accuracy across predeclared seeds")
    linear_study.add_argument("--seeds", nargs="+", type=int, default=[31415,42,1001])
    linear_study.add_argument("--steps", type=int, default=500)
    linear_study.add_argument("--count", type=int, default=20)
    linear_study.add_argument("--probe-seed", type=int, default=20261104)
    linear_predict = sub.add_parser("linear-predict", help="Predict numeric result from saved learned coefficients")
    linear_predict.add_argument("--a", type=int, required=True)
    linear_predict.add_argument("--b", type=int, required=True)
    linear_predict.add_argument("--seed", type=int, default=None)
    linear_predict.add_argument("--candidate", action="store_true")
    multiop = sub.add_parser("multiop-lab", help="Train one shared neural model on addition, subtraction and multiplication")
    multiop.add_argument("--seed", type=int, default=31415)
    multiop.add_argument("--steps", type=int, default=800)
    multiop.add_argument("--hidden", type=int, default=64)
    multiop.add_argument("--count", type=int, default=20)
    multiop.add_argument("--probe-seed", type=int, default=20261104)
    multiop_study = sub.add_parser("multiop-study", help="Run predeclared seeds across all three operations")
    multiop_study.add_argument("--seeds", nargs="+", type=int, default=[31415,42,1001])
    multiop_study.add_argument("--steps", type=int, default=800)
    multiop_study.add_argument("--hidden", type=int, default=64)
    multiop_study.add_argument("--count", type=int, default=20)
    multiop_study.add_argument("--probe-seed", type=int, default=20261104)
    multiop_predict = sub.add_parser("multiop-predict", help="Predict with trained shared arithmetic network")
    multiop_predict.add_argument("--a", type=int, required=True)
    multiop_predict.add_argument("--b", type=int, required=True)
    multiop_predict.add_argument("--operator", choices=["+", "-", "*"], required=True)
    multiop_predict.add_argument("--seed", type=int, default=None)
    multiop_predict.add_argument("--candidate", action="store_true")
    interaction = sub.add_parser("interaction-lab", help="Train a shared 3-operation bilinear neural experiment")
    interaction.add_argument("--seed", type=int, default=31415)
    interaction.add_argument("--steps", type=int, default=800)
    interaction.add_argument("--rank", type=int, default=8)
    interaction.add_argument("--count", type=int, default=20)
    interaction.add_argument("--probe-seed", type=int, default=20261104)
    interaction_study = sub.add_parser("interaction-study", help="Report all seeds on fresh and previous OOD probes")
    interaction_study.add_argument("--seeds", nargs="+", type=int, default=[31415,42,1001])
    interaction_study.add_argument("--steps", type=int, default=800)
    interaction_study.add_argument("--rank", type=int, default=8)
    interaction_study.add_argument("--count", type=int, default=20)
    interaction_study.add_argument("--probe-seed", type=int, default=20261104)
    interaction_predict = sub.add_parser("interaction-predict", help="Predict with saved learned bilinear model")
    interaction_predict.add_argument("--a", type=int, required=True)
    interaction_predict.add_argument("--b", type=int, required=True)
    interaction_predict.add_argument("--operator", choices=["+", "-", "*"], required=True)
    interaction_predict.add_argument("--seed", type=int, default=None)
    interaction_predict.add_argument("--candidate", action="store_true")
    audit = sub.add_parser("basis-audit", help="Read-only effective polynomial coefficient audit for v1.5 checkpoint")
    audit.add_argument("--seed", type=int, default=31415)
    audit.add_argument("--a", type=int, default=27)
    audit.add_argument("--b", type=int, default=28)
    basis = sub.add_parser("basis-lab", help="Fit a supplied polynomial basis on 0..9 examples with float64 least squares")
    basis.add_argument("--seed", type=int, default=31415)
    basis.add_argument("--count", type=int, default=20)
    basis.add_argument("--probe-seed", type=int, default=20261104)
    basis_study = sub.add_parser("basis-study", help="Report fitted polynomial reference and frozen v1.5 for all seeds")
    basis_study.add_argument("--seeds", nargs="+", type=int, default=[31415, 42, 1001])
    basis_study.add_argument("--count", type=int, default=20)
    basis_study.add_argument("--probe-seed", type=int, default=20261104)
    basis_predict = sub.add_parser("basis-predict", help="Predict from saved fitted polynomial coefficients")
    basis_predict.add_argument("--a", type=int, required=True)
    basis_predict.add_argument("--b", type=int, required=True)
    basis_predict.add_argument("--operator", choices=["+", "-", "*"], required=True)
    basis_predict.add_argument("--seed", type=int, default=None)
    calibration = sub.add_parser("calibration-lab", help="Refit only v1.5's output head from original training pairs")
    calibration.add_argument("--seed", type=int, default=31415)
    calibration.add_argument("--count", type=int, default=20)
    calibration.add_argument("--probe-seed", type=int, default=20261104)
    calibration_study = sub.add_parser("calibration-study", help="Report every seed's frozen-v1.5 vs head-calibrated accuracy")
    calibration_study.add_argument("--seeds", nargs="+", type=int, default=[31415,42,1001])
    calibration_study.add_argument("--count", type=int, default=20)
    calibration_study.add_argument("--probe-seed", type=int, default=20261104)
    calibration_predict = sub.add_parser("calibration-predict", help="Predict using a saved calibrated v1.5 model")
    calibration_predict.add_argument("--a", type=int, required=True)
    calibration_predict.add_argument("--b", type=int, required=True)
    calibration_predict.add_argument("--operator", choices=["+", "-", "*"], required=True)
    calibration_predict.add_argument("--seed", type=int, default=31415)
    gradient = sub.add_parser("gradient-lab", help="Train ONLY saved v1.5 output head by Adam or SGD")
    gradient.add_argument("--seed", type=int, default=31415)
    gradient.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    gradient.add_argument("--steps", type=int, default=2000)
    gradient.add_argument("--learning-rate", type=float, default=None)
    gradient.add_argument("--count", type=int, default=20)
    gradient.add_argument("--probe-seed", type=int, default=20261104)
    gradient_study = sub.add_parser("gradient-study", help="Predeclared Adam and SGD comparisons across saved v1.5 seeds")
    gradient_study.add_argument("--seeds", nargs="+", type=int, default=[31415, 42, 1001])
    gradient_study.add_argument("--optimizers", nargs="+", choices=["adam", "sgd"], default=["adam", "sgd"])
    gradient_study.add_argument("--steps", type=int, default=2000)
    gradient_study.add_argument("--count", type=int, default=20)
    gradient_study.add_argument("--probe-seed", type=int, default=20261104)
    gradient_predict = sub.add_parser("gradient-predict", help="Predict from saved Adam or SGD calibrated head")
    gradient_predict.add_argument("--a", type=int, required=True)
    gradient_predict.add_argument("--b", type=int, required=True)
    gradient_predict.add_argument("--operator", choices=["+", "-", "*"], required=True)
    gradient_predict.add_argument("--seed", type=int, default=31415)
    gradient_predict.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    sub.add_parser("concept-inventory", help="Count two-source reusable verified concepts eligible for neural text learning")
    concept = sub.add_parser("concept-lab", help="Train byte Transformer on verified definitions and test unseen paraphrases")
    concept.add_argument("--steps", type=int, default=300)
    concept.add_argument("--seed", type=int, default=31415)
    concept.add_argument("--max-concepts", type=int, default=8)
    concept.add_argument("--max-new-bytes", type=int, default=128)
    concept.add_argument("--force", action="store_true")
    concept_evaluate = sub.add_parser("concept-evaluate", help="Re-evaluate saved neural concept model on withheld question templates")
    concept_evaluate.add_argument("--candidate", action="store_true")
    concept_evaluate.add_argument("--max-new-bytes", type=int, default=128)
    concept_predict = sub.add_parser("concept-predict", help="Generate text from learned checkpoint without any definition lookup")
    concept_predict.add_argument("--question", required=True)
    concept_predict.add_argument("--candidate", action="store_true")
    concept_predict.add_argument("--max-new-bytes", type=int, default=128)
    rule_predict_parser.add_argument("--candidate", action="store_true",
                                     help="Inspect latest candidate even if validation rejected it")
    assess_parser = sub.add_parser("assess", help="Measure raw numeric formatting and exact answers on unseen arithmetic")
    assess_parser.add_argument("--questions", type=int, default=10)
    assess_parser.add_argument("--seed", type=int, default=90210)
    assess_parser.add_argument("--skill", choices=["mixed", "single-digit-add"], default="mixed")
    assess_parser.add_argument("--distribution", choices=["in-range", "extrapolation", "both"], default="extrapolation")
    assess_parser.add_argument("--model-source", choices=["auto", "general", "math"], default="auto")
    train_parser = sub.add_parser("train", help="Train model candidate with PyTorch")
    train_parser.add_argument("--force", action="store_true", help="Ignore new-data threshold")
    train_parser.add_argument("--verified-only", action="store_true",
                              help="Train exclusively from source-corroborated, eligible QA examples")
    sub.add_parser("lab", help="Try verifiable arithmetic experiments")
    run_parser = sub.add_parser("once", help="Run one bounded autonomous cycle")
    run_parser.add_argument("--offline", action="store_true", help="Skip internet requests")
    run_parser.add_argument("--no-train", action="store_true")
    run_parser.add_argument("--no-lab", action="store_true")
    run_parser.add_argument("--practice-arithmetic", action="store_true",
                            help="Enable Python-verified math practice for this learning cycle")
    sub.add_parser("daemon", help="Run scheduled bounded learning cycles")
    infer_parser = sub.add_parser("infer", help="Generate sample text from promoted tiny model")
    infer_parser.add_argument("--prompt", default="Earth")
    infer_parser.add_argument("--new-bytes", type=int, default=100)
    infer_parser.add_argument("--use-memory", action="store_true", help="Prepend short retrieved context where block_size permits")
    args = parser.parse_args(argv)
    settings = load_config(args.config)

    if args.command == "studio":
        from .v118_studio import serve
        serve(settings, port=args.port, interval=args.interval)
        return
    if args.command == "daemon":
        from .runtime import daemon
        daemon(settings)
        return
    if args.command == "focus-overnight":
        if not args.network:
            parser.error("focus-overnight requires --network: no implicit web access")
        from .focus_overnight import run
        focus_store = Store(settings["storage_root"])
        try:
            run(
                focus_store, settings,
                profile_name=args.profile,
                hours=args.hours, interval_minutes=args.interval_minutes,
                max_pages=args.max_pages,
                pages_per_cycle=args.pages_per_cycle,
                train_steps=args.train_steps,
                until_stopped=args.until_stopped)
        finally:
            focus_store.close()
        return
    if args.command == "once":
        from .runtime import once
        result = once(settings, network=not args.offline, train=not args.no_train,
                      lab=not args.no_lab, practice=args.practice_arithmetic)
    else:
        store = Store(settings["storage_root"])
        try:
            if args.command == "status":
                result = store.status()
            elif args.command == "model-audit":
                from .model_audit import audit_model_and_ownership
                result = audit_model_and_ownership(store, settings=settings)
            elif args.command == "corpus-status":
                from .open_corpus import corpus_status
                result = corpus_status(store)
            elif args.command == "focus-plan":
                from .focus_overnight import plan
                result = plan(profile_name=args.profile)
            elif args.command == "focus-report":
                from .focus_overnight import report
                result = report(store, profile_name=args.profile)
            elif args.command == "focus-stop":
                from .focus_control import request_stop
                result = request_stop(store, profile=args.profile)
            elif args.command == "corpus-seed":
                from .open_corpus import corpus_seed
                result = corpus_seed(store, args.urls, cross_host=not args.same_host)
            elif args.command == "corpus-crawl":
                from .web_discovery import discover_web
                result = discover_web(store, settings, limit=args.limit,
                                      review_questions=False, host=args.host)
            elif args.command == "errors":
                result = {"errors": store.recent_web_errors(args.limit)}
            elif args.command == "retry-url":
                from .web_discovery import canonical_url
                result = store.retry_failed_web_url(canonical_url(args.url))
            elif args.command == "concepts":
                result = {"concepts": store.list_concept_tasks(args.limit)}
            elif args.command == "questions":
                result = {"questions": store.list_research_questions(args.limit)}
            elif args.command == "answers":
                result = {"answers": store.list_verified_answers(args.limit)}
            elif args.command == "research-health":
                from .research_diagnostics import research_health
                result = research_health(store, settings)
            elif args.command == "research":
                from .research import research_questions
                result = research_questions(store, settings, limit=args.limit, concept=args.concept)
            elif args.command == "research-plan":
                from .research_diagnostics import research_plan
                result = research_plan(store, settings, concept=args.concept)
            elif args.command == "v117-sources":
                from .v117_rights import sources
                result = sources(store)
            elif args.command == "v117-review-rights":
                from .v117_rights import review_rights
                result = review_rights(store, args.url, args.license,
                                       args.evidence_url,
                                       confirmed=args.confirm_manual_review)
            elif args.command == "v121-plan":
                from .v121_autonomy import plan
                result = plan(store, settings, max_topics=args.max_topics)
            elif args.command == "v121-cycle":
                from .v121_autonomy import cycle
                result = cycle(store, settings, network=args.network,
                               max_topics=args.max_topics,
                               max_requests=args.max_requests,
                               steps=args.steps, seed=args.seed)
            elif args.command == "v121-history":
                from .v121_autonomy import history
                result = history(store)
            elif args.command == "v121-report":
                from .v121_autonomy import learning_report
                result = learning_report(store)
            elif args.command == "v122-evidence":
                from .v122_evidence import evidence_report
                result = evidence_report(store, limit=args.limit,
                                         host=args.host)
            elif args.command == "v124-eval":
                from .v124_inference_audit import evaluate_inference
                result = evaluate_inference(store, limit=args.limit)
            elif args.command == "v125-pairs":
                from .v125_reviewed_benchmark import candidate_pairs
                result = candidate_pairs(store, limit=args.limit)
            elif args.command == "v125-label":
                from .v125_reviewed_benchmark import review_pair
                result = review_pair(
                    store, args.question, args.better_url, args.better_sha,
                    args.worse_url, args.worse_sha,
                    confirm=args.confirm_human_review)
            elif args.command == "v125-score":
                from .v125_reviewed_benchmark import score_reviewed_pairs
                result = score_reviewed_pairs(store, limit=args.limit)
            elif args.command == "v126-expansion":
                from .v126_expansion import expansion_report
                result = expansion_report(
                    store, host=args.host, limit=args.limit,
                    quote_limit=args.quote_limit)
            elif args.command == "v120-ask":
                from .v120_grounded import grounded_answer
                result = grounded_answer(store, args.question)
            elif args.command == "v120-inventory":
                from .v120_grounded import inventory
                result = inventory(store)
            elif args.command == "v120-readiness":
                from .v120_grounded import readiness
                result = readiness(store)
            elif args.command == "v120-diagnose":
                from .v120_grounded import diagnose_corpus
                result = diagnose_corpus(store, args.question)
            elif args.command == "v120-train":
                from .v120_grounded import train_ranker
                result = train_ranker(store, settings, steps=args.steps, seed=args.seed)
            elif args.command == "v119-research":
                from .v119_curiosity import research
                result = research(store, settings, args.question)
            elif args.command == "v119-inventory":
                from .v119_curiosity import inventory
                result = inventory(store)
            elif args.command == "v119-self-study":
                from .v119_curiosity import sentence_lessons
                result = sentence_lessons(store)
            elif args.command == "v119-study-train":
                from .v119_curiosity import sentence_lessons
                from .v118_core import train_qa
                extracted = sentence_lessons(store)
                trained = train_qa(store, settings, steps=args.steps)
                result = {"source_exercises": extracted, "training": trained}
            elif args.command == "v118-plan":
                from .v118_core import curriculum
                result = curriculum(store)
            elif args.command == "v118-qa-readiness":
                from .v118_core import qa_readiness
                result = qa_readiness(store)
            elif args.command == "v118-qa-add":
                from .v118_core import add_qa_example
                result = add_qa_example(store, args.question, args.answer,
                                        args.source_url)
            elif args.command == "v118-qa-train":
                from .v118_core import train_qa
                result = train_qa(store, settings, steps=args.steps, seed=args.seed)
            elif args.command == "v118-chat":
                from .v118_core import ask
                result = ask(store, args.question, args.mode)
            elif args.command == "v117-plan":
                from .v117 import plan
                result = plan(store)
            elif args.command == "v117-prepare":
                from .v117 import prepare
                result = prepare(store)
            elif args.command == "v117-train":
                from .v117 import train
                result = train(store, settings, steps=args.steps, seed=args.seed,
                               device=args.device, bpe_vocab=args.bpe_vocab)
            elif args.command == "v117-evaluate":
                from .v117 import evaluate
                result = evaluate(store)
            elif args.command == "v117-smoke":
                from .v117 import smoke
                result = smoke(settings, steps=args.steps)
            elif args.command == "web-inventory":
                from .web_lab import web_inventory
                result = web_inventory(store, include_unknown=args.include_unknown_rights)
            elif args.command == "web-freeze":
                from .web_lab import freeze_dataset
                result = freeze_dataset(store, include_unknown=args.include_unknown_rights,
                                        diagnostic=args.diagnostic_within_document)
            elif args.command == "char-lab":
                from .char_lab import train_char_lab
                result = train_char_lab(
                    store, settings, steps=args.steps, seed=args.seed,
                    device=args.device,
                    include_unknown=args.include_unknown_rights,
                    diagnostic=args.diagnostic_within_document,
                    resume=args.resume, context=args.context,
                    embedding_dim=args.embedding_dim,
                    layers=args.layers, heads=args.heads,
                    vocab_limit=args.vocab_limit)
            elif args.command == "char-compare":
                from .char_compare import compare_char_checkpoints
                result = compare_char_checkpoints(
                    store, args.reference_sha256, prompt=args.prompt,
                    new_chars=args.new_chars, temperature=args.temperature,
                    top_k=args.top_k, seed=args.seed,
                    candidate=not args.accepted)
            elif args.command == "char-evaluate":
                from .char_lab import evaluate_char_lab
                result = evaluate_char_lab(store, candidate=not args.accepted)
            elif args.command == "char-predict":
                from .char_lab import predict_char_lab
                result = predict_char_lab(
                    store, args.prompt, candidate=not args.accepted,
                    new_chars=args.new_chars,
                    temperature=args.temperature, seed=args.seed,
                    top_k=args.top_k, greedy=args.greedy)
            elif args.command == "web-lab":
                from .web_lab import train_web_lab
                result = train_web_lab(store, settings, steps=args.steps,
                                       seed=args.seed, device=args.device,
                                       include_unknown=args.include_unknown_rights,
                                       diagnostic=args.diagnostic_within_document)
            elif args.command == "web-evaluate":
                from .web_lab import evaluate_web_lab
                result = evaluate_web_lab(store, candidate=not args.accepted)
            elif args.command == "web-predict":
                from .web_lab import predict_web_lab
                result = predict_web_lab(
                    store, args.prompt, candidate=not args.accepted,
                    max_new_bytes=args.new_bytes, decoding=args.decoding,
                    temperature=args.temperature, seed=args.seed)
            elif args.command == "web-diagnose":
                from .web_diagnostics import diagnose_web_lab
                result = diagnose_web_lab(
                    store, candidate=not args.accepted,
                    prompt=args.prompt, max_new_bytes=args.new_bytes,
                    temperature=args.temperature, seed=args.seed)
            elif args.command == "learn":
                from .research import research_questions
                from .training import train_candidate
                research_report = research_questions(store, settings, limit=args.limit)
                result = {"research": research_report}
                if args.practice_arithmetic:
                    from .practice import practice_arithmetic
                    result["practice"] = practice_arithmetic(store, limit=args.practice_limit)
                result["training"] = train_candidate(store, settings, force=args.force, verified_only=True)
            elif args.command == "practice":
                from .practice import practice_arithmetic
                result = practice_arithmetic(store, limit=args.limit, skill=args.skill)
            elif args.command == "math-finetune":
                from .answer_training import train_math_answerer
                result = train_math_answerer(store, settings, skill=args.skill,
                                             steps=args.steps, force=args.force)
            elif args.command == "numeric-lab":
                from .numeric_lab import run_numeric_lab
                result = run_numeric_lab(store, settings, steps=args.steps, seed=args.seed)
            elif args.command == "numeric-predict":
                from .numeric_lab import predict_numeric_lab
                result = predict_numeric_lab(store, args.a, args.b)
            elif args.command == "rule-lab":
                from .rule_lab import run_rule_lab
                result = run_rule_lab(store, settings, steps=args.steps,
                                      seed=args.seed, hidden=args.hidden)
            elif args.command == "rule-predict":
                from .rule_lab import predict_rule_lab
                result = predict_rule_lab(store, args.a, args.b, use_candidate=args.candidate)
            elif args.command == "magnitude-lab":
                from .magnitude_lab import run_magnitude_lab
                result = run_magnitude_lab(store, settings, steps=args.steps,
                                           seed=args.seed, hidden=args.hidden)
            elif args.command == "magnitude-study":
                from .magnitude_lab import run_magnitude_study
                result = run_magnitude_study(store, settings, seeds=args.seeds,
                                             steps=args.steps, hidden=args.hidden)
            elif args.command == "magnitude-predict":
                from .magnitude_lab import predict_magnitude_lab
                result = predict_magnitude_lab(store, args.a, args.b,
                                               candidate=args.candidate)
            elif args.command == "magnitude-extrapolate":
                from .extrapolation import magnitude_extrapolate
                result = magnitude_extrapolate(store, model_seed=args.model_seed,
                                               count=args.count, probe_seed=args.probe_seed)
            elif args.command == "magnitude-extrapolation-study":
                from .extrapolation import magnitude_extrapolation_study
                result = magnitude_extrapolation_study(store,
                    model_seeds=args.model_seeds, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "linear-lab":
                from .linear_lab import run_linear_lab
                result = run_linear_lab(store, settings, seed=args.seed,
                    steps=args.steps, count=args.count, probe_seed=args.probe_seed)
            elif args.command == "linear-study":
                from .linear_lab import run_linear_study
                result = run_linear_study(store, settings, seeds=args.seeds,
                    steps=args.steps, count=args.count, probe_seed=args.probe_seed)
            elif args.command == "linear-predict":
                from .linear_lab import predict_linear_lab
                result = predict_linear_lab(store, args.a, args.b, seed=args.seed,
                                            candidate=args.candidate)
            elif args.command == "multiop-lab":
                from .multiop_lab import run_multiop_lab
                result = run_multiop_lab(store, settings, seed=args.seed,
                    steps=args.steps, hidden=args.hidden, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "multiop-study":
                from .multiop_lab import run_multiop_study
                result = run_multiop_study(store, settings, seeds=args.seeds,
                    steps=args.steps, hidden=args.hidden, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "multiop-predict":
                from .multiop_lab import predict_multiop_lab
                result = predict_multiop_lab(store, args.a, args.b,
                    operator=args.operator, seed=args.seed, candidate=args.candidate)
            elif args.command == "interaction-lab":
                from .interaction_lab import run_interaction_lab
                result = run_interaction_lab(store, settings, seed=args.seed,
                    steps=args.steps, rank=args.rank, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "interaction-study":
                from .interaction_lab import run_interaction_study
                result = run_interaction_study(store, settings, seeds=args.seeds,
                    steps=args.steps, rank=args.rank, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "interaction-predict":
                from .interaction_lab import predict_interaction_lab
                result = predict_interaction_lab(store, args.a, args.b,
                    operator=args.operator, seed=args.seed, candidate=args.candidate)
            elif args.command == "basis-audit":
                from .basis_lab import audit_interaction_coefficients
                result = audit_interaction_coefficients(store, seed=args.seed,
                                                        a=args.a, b=args.b)
            elif args.command == "basis-lab":
                from .basis_lab import run_basis_lab
                result = run_basis_lab(store, seed=args.seed, count=args.count,
                                       probe_seed=args.probe_seed)
            elif args.command == "basis-study":
                from .basis_lab import run_basis_study
                result = run_basis_study(store, seeds=args.seeds, count=args.count,
                                         probe_seed=args.probe_seed)
            elif args.command == "basis-predict":
                from .basis_lab import predict_basis_lab
                result = predict_basis_lab(store, a=args.a, b=args.b,
                                           operator=args.operator, seed=args.seed)
            elif args.command == "calibration-lab":
                from .calibration_lab import run_calibration_lab
                result = run_calibration_lab(store, seed=args.seed,
                    count=args.count, probe_seed=args.probe_seed)
            elif args.command == "calibration-study":
                from .calibration_lab import run_calibration_study
                result = run_calibration_study(store, seeds=args.seeds,
                    count=args.count, probe_seed=args.probe_seed)
            elif args.command == "calibration-predict":
                from .calibration_lab import predict_calibration_lab
                result = predict_calibration_lab(store, a=args.a, b=args.b,
                    operator=args.operator, seed=args.seed)
            elif args.command == "gradient-lab":
                from .gradient_lab import run_gradient_lab
                result = run_gradient_lab(store, seed=args.seed,
                    optimizer=args.optimizer, steps=args.steps,
                    learning_rate=args.learning_rate, count=args.count,
                    probe_seed=args.probe_seed)
            elif args.command == "gradient-study":
                from .gradient_lab import run_gradient_study
                result = run_gradient_study(store, seeds=args.seeds,
                    optimizers=args.optimizers, steps=args.steps,
                    count=args.count, probe_seed=args.probe_seed)
            elif args.command == "gradient-predict":
                from .gradient_lab import predict_gradient_lab
                result = predict_gradient_lab(store, args.a, args.b,
                    operator=args.operator, seed=args.seed, optimizer=args.optimizer)
            elif args.command == "concept-inventory":
                from .concept_lab import concept_inventory
                result = concept_inventory(store, max_concepts=32)
            elif args.command == "concept-lab":
                from .concept_lab import train_concept_lab
                result = train_concept_lab(store, settings, steps=args.steps,
                    seed=args.seed, max_concepts=args.max_concepts,
                    force=args.force, max_new_bytes=args.max_new_bytes)
            elif args.command == "concept-evaluate":
                from .concept_lab import evaluate_concept_lab
                result = evaluate_concept_lab(store, candidate=args.candidate,
                    max_new_bytes=args.max_new_bytes)
            elif args.command == "concept-predict":
                from .concept_lab import predict_concept_lab
                result = predict_concept_lab(store, question=args.question,
                    candidate=args.candidate, max_new_bytes=args.max_new_bytes)
            elif args.command == "assess":
                from .evaluation import evaluate_arithmetic
                result = evaluate_arithmetic(store, settings, questions=args.questions,
                    seed=args.seed, skill=args.skill,
                    distribution=args.distribution, model_source=args.model_source)
            elif args.command == 'explore':
                from .web_discovery import register_exploration, discover_web
                profile = register_exploration(store,args.seed_url,cross_host=args.cross_host)
                result = {'profile':profile, 'first_discovery':discover_web(store,settings,limit=args.limit)}
            elif args.command == "web-start":
                from .web_discovery import register_seed,discover_web,fetch_page
                if bool(args.license) != bool(args.rights_evidence_url):
                    raise ValueError('Provide both --license and --rights-evidence-url, or neither to use on-page CC0 detection')
                seed_html=(fetch_page(args.seed_url,settings) if not args.license else None)
                profile=register_seed(store,args.seed_url,args.license,args.rights_evidence_url,seed_html=seed_html)
                result={"profile":profile,"first_discovery":discover_web(store,settings,limit=args.limit)}
            elif args.command == "discover":
                if store.get('web_source_profile'):
                    from .web_discovery import discover_web
                    result = discover_web(store, settings, limit=args.limit)
                elif args.legacy_wikidata:
                    from .discovery import discover
                    result = discover(store, settings, limit=args.limit)
                else:
                    result = {'status':'not_configured','message':'Use explore --seed-url ... to initialize public web discovery'}
            elif args.command == "train":
                from .training import train_candidate
                result = train_candidate(store, settings, force=args.force,
                                         verified_only=args.verified_only)
            elif args.command == "lab":
                from .lab import run_math_experiments
                result = run_math_experiments(store, settings)
            elif args.command == "infer":
                from .brain import load_checkpoint, get_device, sample_text
                active = store.latest_promoted()
                if not active:
                    raise RuntimeError("No accepted model. Run discovery and training first.")
                model, meta = load_checkpoint(active["path"])
                prompt = args.prompt
                used_memory = []
                if args.use_memory:
                    block = meta["model_config"]["block_size"]
                    # The tiny model has a tiny context window; don't push the prompt out of it.
                    available = block - len(prompt.encode("utf-8")) - 8
                    if available > 20:
                        for candidate in store.relevant_memories(prompt, limit=1):
                            excerpt = candidate.encode("utf-8")[:available].decode("utf-8", errors="ignore")
                            if excerpt:
                                used_memory = [excerpt]
                                prompt = excerpt + "\n" + prompt
                            break
                result = {"generation":sample_text(model, prompt, meta["model_config"]["block_size"],
                    max_new_bytes=min(max(1,args.new_bytes), 2000), device=get_device(settings["train"]["device"])),
                    "retrieved_memory":used_memory}
            else:
                raise RuntimeError("Invalid command")
        finally:
            store.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
