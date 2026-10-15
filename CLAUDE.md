# CLAUDE.md

> **CRITICAL / MANDATORY INSTRUCTION:**
> You **MUST** read and strictly follow [AGENTS.md](AGENTS.md) and [docs/commit-convention.md](docs/commit-convention.md) before writing code, proposing commit messages, or performing any task in this repository.

---

## 1. Prime Directive: Follow AGENTS.md

All AI agents (including Claude) working in this repository are strictly bound by the rules specified in [AGENTS.md](AGENTS.md). Whenever there is any ambiguity, [AGENTS.md](AGENTS.md) takes precedence.

---

## 2. Commit Message Convention (Strict Enforcement)

All commit messages **MUST** follow:

```txt
TYPE(scope): summary
```

- **`TYPE` MUST be UPPERCASE:** `FEAT`, `FIX`, `REFACT`, `DATA`, `PERF`, `TEST`, `DOCS`, `STYLE`, `CHORE`, `SEC`, `OPS`, `REMOVE`.
  - **PROHIBITED:** Lowercase or mixed-case types (e.g., `feat`, `fix`, `Fix`, `docs`, `chore`). If the user specifies a lowercase type, you **MUST** normalize it to uppercase.
- **`scope` MUST be lowercase:** e.g., `model`, `preprocess`, `features`, `cli`, `eval`, `data`, `artifacts`, `tests`, `deps`, `docs`, `tooling`, `ci`, `infra`.
- **`summary`:** Must be written in **English**, imperative mood (e.g., `add`, `fix`, `update`), concise, and **NEVER** end with a period (`.`).
- **No vague summaries:** Do not use `update code`, `fix bug`, `changes`, etc.
- **Breaking changes:** Use `!` after scope (e.g., `FEAT(cli)!: change required arguments for train command`) or add a `BREAKING CHANGE:` footer.
- **Approval required:** **NEVER** run `git commit` or `git push` automatically without explicit user confirmation.

---

## 3. Data Safety & Privacy Rules

- **NEVER** commit real, private, or sensitive emails into the repository.
- Only synthetic, anonymized, or public sample data is allowed in [data/raw/emails.csv](data/raw/emails.csv).
- Adhere strictly to [.gitignore](.gitignore): do not force-add large raw datasets under `data/raw/*` or generated model binaries under `artifacts/*`.
- Verify dataset contracts: input datasets must contain a `label` column (`spam`/`ham`, `1`/`0`) and at least one text column (`text`, `subject`, `body`, `sender`, `recipient`).

---

## 4. Machine Learning & Code Standards

- **Determinism:** Always provide `random_state` in data splits and stochastic models.
- **Imbalanced class metrics:** Spam email datasets are typically class-imbalanced; always report `spam_f1`, `spam_precision`, and `spam_recall` alongside `accuracy`.
- **Safe persistence:** Ensure parent directories exist before saving model artifacts (`target.parent.mkdir(parents=True, exist_ok=True)`).

---

## 5. Essential Development Commands

Always use the virtual environment located at `.venv/`:

```bash
# Run test suite
.venv/bin/pytest

# Code formatting and linting
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# Train model
.venv/bin/python -m spam_classifier.cli train --data data/raw/emails.csv --model logistic_regression

# Evaluate model
.venv/bin/python -m spam_classifier.cli evaluate --data data/raw/emails.csv --model-path artifacts/model.joblib

# Predict single email
.venv/bin/python -m spam_classifier.cli predict --model-path artifacts/model.joblib --text "Free cash prize now!"
```
