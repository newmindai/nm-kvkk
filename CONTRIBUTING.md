# Contributing

## Development setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

After `pre-commit install`, every `git commit` runs the hooks on the staged files:

- `ruff check --fix` lints the Python code and fixes what it can.
- `ruff format` formats the Python code.
- Standard checks: JSON, TOML and YAML syntax, merge-conflict markers, private keys, files larger than 1 MB,
  trailing whitespace and missing final newlines.

If a hook changes a file, the commit stops. Stage the changes and commit again.

## Useful commands

```bash
pre-commit run --all-files   # run every hook on the whole repository
ruff check .                 # lint only
ruff format .                # format only
python -m pytest tests       # tests
```

The rules are in `ruff.toml`. `GLiNER2/` is a vendored copy of the upstream library, so the hooks do not change it.
