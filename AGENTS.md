# Repository Guidelines

## Project Structure & Module Organization
- `src/` is the main package: feature extraction in `src/features/`, models in `src/models/` (including `deep_learning/`), API handlers in `src/api/`, annotation helpers in `src/annotation/`, and experiments in `src/experiments/`.
- Tests live in `src/tests/` and are organized as `test_*.py` files with a shared runner.
- Configuration is centralized in `src/config/config.py`.
- Data and tooling live in `data/` (notably `data/mm-tba/tools/`), while outputs land in `checkpoints/`, `logs/`, and `result/`.
- Project documentation is in `docs/` and `doc/`.

## Build, Test, and Development Commands
- Install dependencies: `pip install -r src/requirements.txt`.
- Analyze a single video: `python -m src.main analyze --video path/to/video.mp4 --teacher T001 --discipline 数学 --grade 初中 --mode deep_learning --device cuda`.
- Run batch analysis: `python -m src.main batch --dir path/to/videos --teacher T001 --discipline 数学 --grade 初中 --device cuda`.
- Start the API server: `python -m src.main server --host 0.0.0.0 --port 8000`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8–style naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep modules focused and mirror existing package layout (`src/features/`, `src/models/`, etc.).
- No formatter is enforced in repo; keep diffs small and consistent with adjacent code.

## Testing Guidelines
- Tests use `unittest` via the runner: `python -m src.tests.run_tests`.
- Run a single file: `python -m src.tests.run_tests -f test_feature_extractor.py`.
- Optional report: `python -m src.tests.run_tests -r test_report.md`.

## Commit & Pull Request Guidelines
- Commit messages in history are short, descriptive, and often in Chinese; follow that pattern and keep messages concise.
- PRs should explain the change, list commands run, and call out any new data, model weights, or long-running steps.
- Avoid adding large checkpoints unless required; if you must, explain size and provenance.

## Security & Configuration Tips
- API tokens (e.g., VLM annotation) should be set via environment variables and never committed.
- Review `src/config/config.py` for path defaults and service settings before running locally.
