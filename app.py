from llmbackends import OllamaBackend
from config import AgentConfig
from orchestrator import run_agent


user_request = "Build a Python function that returns factorial of a number."


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

def main():

    config = AgentConfig()
    backend = OllamaBackend(model_name=config.model)
    report = run_agent(user_request, backend, config)
    _print_progress(report)
    if report.success:
        print("Final implementation:\n################################################")
        print(report.records[-1].artifacts.implementation_code)
        print("################################################")

if __name__ == "__main__":
    main()