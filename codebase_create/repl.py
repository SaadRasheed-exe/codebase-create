"""Interactive session: persistent workspace, fresh conversation per request.

The workspace survives across requests so you can iterate on real
files ("now add negative-number handling"); the conversation history
does NOT persist — memory lives in the files, not in chat context.
That boundary keeps requests independent and token budgets small.
"""

from dotenv import load_dotenv

load_dotenv()  # standalone ReplDriver usage still picks up .env

from codebase_create.agent_loop import run_agent
from codebase_create.config import AgentConfig
from codebase_create.executor import TempWorkspace
from codebase_create.providers import build_provider
from codebase_create.providers.base import ProviderError
from codebase_create.ui import Renderer

BANNER = """\
Codebase Create - interactive agent mode.
Describe a programming task; the agent writes code and tests, runs
them, and iterates until green. Workspace persists across requests.

Commands: /files  /reset  /help  /exit"""

HELP_TEXT = """\
/files   list files in the current workspace
/reset   start a fresh workspace (old one is discarded)
/exit    leave interactive mode (also: /quit, Ctrl+D)
/help    show this help"""

EXIT_COMMANDS = {"/exit", "/quit", "/q"}


class ReplDriver:
    def __init__(
        self,
        config: AgentConfig,
        renderer: Renderer,
        input_fn=input,
        output_fn=print,
    ) -> None:
        self._config = config
        self._renderer = renderer
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._workspace: TempWorkspace | None = None

    def run(self) -> int:
        self._workspace = TempWorkspace(keep_artifacts=self._config.keep_artifacts)
        self._output_fn(BANNER)
        try:
            while True:
                try:
                    line = self._input_fn("agent> ")
                except EOFError:
                    self._output_fn("")
                    break
                except KeyboardInterrupt:
                    self._output_fn("^C")
                    continue

                command = line.strip()
                if not command:
                    continue
                if command in EXIT_COMMANDS:
                    break
                if command == "/help":
                    self._output_fn(HELP_TEXT)
                elif command == "/files":
                    self._show_files()
                elif command == "/reset":
                    self._reset_workspace()
                elif command.startswith("/"):
                    self._output_fn(f"Unknown command '{command}'. Try /help.")
                else:
                    self._run_request(command)
        finally:
            if not self._config.keep_artifacts and self._workspace is not None:
                self._workspace.cleanup()
        return 0

    def _run_request(self, request: str) -> None:
        assert self._workspace is not None
        # Fresh provider per request: mock scenarios replay cleanly and
        # real backends carry no state worth reusing across requests.
        try:
            provider = build_provider(self._config)
        except ProviderError as ex:
            self._output_fn(f"error: {ex}")
            return
        try:
            report = run_agent(
                request,
                provider,
                self._config,
                on_event=self._renderer.handle_event,
                workspace=self._workspace,
            )
        except KeyboardInterrupt:
            self._output_fn("(request cancelled - workspace kept)")
            return
        self._renderer.render_report(report)

    def _show_files(self) -> None:
        assert self._workspace is not None
        infos = self._workspace.list_files()
        if not infos:
            self._output_fn("(workspace is empty)")
            return
        for info in infos:
            self._output_fn(f"{info.path} ({info.bytes} bytes)")

    def _reset_workspace(self) -> None:
        assert self._workspace is not None
        if not self._config.keep_artifacts:
            self._workspace.cleanup()
        self._workspace = TempWorkspace(keep_artifacts=self._config.keep_artifacts)
        self._output_fn("(new workspace started)")


def run_repl(config: AgentConfig, renderer: Renderer) -> int:
    return ReplDriver(config, renderer).run()
