# Copilot Workspace Instructions

## Python command usage in this repository

- Always use `python3` for terminal commands.
- Do not use `python` in this code-server container.
- There is no `python -> python3` shim installed, so `python` returns `command not found`.
- If a command example currently uses `python`, convert it to `python3` before running.

## Validation reminders

- For quick interpreter checks, use `python3 --version`.
- For snippet execution, use `python3 -c "..."`.
