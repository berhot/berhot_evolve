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
                result = practice_arithmetic(store, limit=args.limit)
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
