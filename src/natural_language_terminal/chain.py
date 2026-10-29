import argparse
import subprocess
import sys

from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from natural_language_terminal.ShellCommand import ShellCommand

SYSTEM_PROMPT = """You are a macOS and zsh terminal expert.
Translate the user's request into a single shell command that runs on macOS.

Rules:
- Use macOS / BSD tools and flags only (e.g. `ipconfig getifaddr en0`, `stat -f`, `sed -i ''`, `lsof`, `netstat -anv`, `vm_stat`).
- Never use Linux-only GNU tools or flags (e.g. `ip addr`, `free`, `ss`, `stat -c`, `sed -i` without '', `ls --color`).
- `explanation` must be exactly one sentence.
- `is_safe` is true ONLY if the command is read-only; false if it creates, modifies, moves, or deletes anything, or needs sudo."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{request}"),
    ]
)


def build_chain(model: str = "gemma4:26b"):
    llm = ChatOllama(model=model, temperature=0)
    return prompt | llm.with_structured_output(ShellCommand)


def execute_shell_command(command: str) -> subprocess.CompletedProcess[str]:
    """Run the command in zsh and capture stdout/stderr."""
    return subprocess.run(
        command,
        shell=True,
        executable="/bin/zsh",
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural language -> macOS shell command")
    parser.add_argument("request", nargs="?", default="Show me my current IP address on en0")
    parser.add_argument("-m", "--model", default="gemma4:26b", help="Ollama model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log full chain inputs/outputs")
    parser.add_argument("--dry-run", action="store_true", help="Generate command without executing it")
    args = parser.parse_args()

    # LangChain's global debug mode logs every step's input, output, and raw LLM response.
    set_debug(args.verbose)

    result: ShellCommand = build_chain(args.model).invoke({"request": args.request})

    print(f"command:     {result.command}")
    print(f"explanation: {result.explanation}")
    print(f"is_safe:     {result.is_safe}")

    if args.dry_run:
        return

    if not result.is_safe:
        answer = input("\n⚠️  Unsafe command (modifies state or requires sudo). Execute anyway? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    print("\n--- Output ---")
    completed = execute_shell_command(result.command)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        print(f"\n[Process exited with code {completed.returncode}]", file=sys.stderr)
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
