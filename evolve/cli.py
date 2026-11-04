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
    sub.add_parser("status", help="Show persistent learner status")
    sub.add_parser("model-audit", help="Read-only actual local knowledge, saved models, autonomy and dependency inventory")
    sub.add_parser("corpus-status", help="Show collected web pages, exact dedup, rights metadata and training separation")
    corpus_seed = sub.add_parser("corpus-seed", help="Queue 1..20 public HTTPS sites without requiring a CC0 license")
    corpus_seed.add_argument("--urls", nargs="+", required=True)
    corpus_seed.add_argument("--same-host", action="store_true",
                             help="Do not follow links across different hosts")
    corpus_crawl = sub.add_parser("corpus-crawl", help="Bounded existing-queue HTML extraction and independent raw/clean corpus archive")
    corpus_crawl.add_argument("--limit", type=int, default=5)
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

    if args.command == "daemon":
        from .runtime import daemon
        daemon(settings)
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
            elif args.command == "corpus-seed":
                from .open_corpus import corpus_seed
                result = corpus_seed(store, args.urls, cross_host=not args.same_host)
            elif args.command == "corpus-crawl":
                from .web_discovery import discover_web
                result = discover_web(store, settings, limit=args.limit,
                                      review_questions=False)
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
