import json
import argparse

from config import AgentConfig
from llmbackends import OllamaBackend
from orchestrator import run_agent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI coding agent")
    parser.add_argument("prompt", help="User programming request")
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Model name")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Test run timeout in seconds")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print final report as JSON")
    return parser


def _print_progress(report) -> None:
    for record in report.records:
        execution = record.execution
        if execution is None:
            continue
        print(
            f"Attempt {record.attempt}: "
            f"success={execution.success} "
            f"passed={execution.passed} "
            f"failed={execution.failed} "
            f"errors={execution.errors} "
            f"category={execution.category}"
        )
    print(f'Final result: success={report.success}')
    if not report.success:
        print(f"Failure reason: {report.failure_summary}")

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = AgentConfig.from_env()
    if args.model:
        config.model = args.model
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.timeout:
        config.test_timeout_sec = args.timeout
    if args.keep_artifacts:
        config.keep_artifacts = True

    backend = OllamaBackend(model_name=config.model)
    report = run_agent(args.prompt, backend, config)
    
    _print_progress(report)
    
    if args.json:
        payload = {
            "success": report.success,
            "attempts_used": report.attempts_used,
            "max_iterations": report.max_iterations,
            "failure_category": report.failure_category,
            "failure_summary": report.failure_summary,
        }
        print(json.dumps(payload, indent=2))
    
    else:
        print("-" * 60)
        print(f"Success: {report.success}")
        print(f"Attempts used: {report.attempts_used}/{report.max_iterations}")
        print(f"Failure category: {report.failure_category}")
        print(f"Summary: {report.failure_summary}")

    if report.success:
        print("Final implementation:\n################################################")
        print(report.records[-1].artifacts.implementation_code)
        print("################################################")
        return 0

    return 1
    

if __name__ == "__main__":
    raise SystemExit(main())