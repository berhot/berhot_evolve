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
    research_parser = sub.add_parser("research", help="Seek independent evidence for open questions")
    research_parser.add_argument("--limit", type=int, default=5)
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
            elif args.command == "research":
                from .research import research_questions
                result = research_questions(store, settings, limit=args.limit)
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
