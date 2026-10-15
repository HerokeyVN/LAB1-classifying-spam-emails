# Agent Guidelines & Workflow Rules

This document defines mandatory operational instructions for all AI agents working in this repository (`CLASSIFYING-SPAM-EMAILS`).

---

## 1. Mandatory Commit Message Convention

All commit messages suggested, drafted, or executed by AI agents **MUST** strictly follow the project convention defined in [docs/commit-convention.md](docs/commit-convention.md).

### Required Format

```txt
TYPE(scope): summary
```

### Strict Enforcement Rules

1. **`TYPE` MUST be Uppercase:**
   - **Allowed uppercase types:** `FEAT`, `FIX`, `REFACT`, `DATA`, `PERF`, `TEST`, `DOCS`, `STYLE`, `CHORE`, `SEC`, `OPS`, `REMOVE`.
   - **STRICTLY PROHIBITED:** Lowercase or mixed-case types (e.g., `feat`, `fix`, `Fix`, `docs`, `Refactor`, `chore`).
   - If a user request specifies or uses a lowercase type, the agent **MUST** normalize it to uppercase before proposing the commit.

2. **`scope` MUST be Lowercase:**
   - The scope must identify the exact domain or component being changed.
   - **Allowed/Recommended scopes:**
     - `model` - Classifiers (`logistic_regression`, `linear_svm`, `naive_bayes`), pipelines, estimators, hyperparameters, model persistence (`joblib`).
     - `preprocess` - Text cleaning (`clean_text`), HTML stripping, whitespace normalization, email field aggregation (`combine_email_fields`), label normalization (`normalize_labels`).
     - `features` - Feature extraction, TF-IDF vectorization, n-grams, stop-words, vocabulary.
     - `cli` - CLI commands (`train`, `evaluate`, `predict`), argument parsing, console output.
     - `eval` - Evaluation metrics (`accuracy`, `spam_precision`, `spam_recall`, `spam_f1`), classification reports, confusion matrix.
     - `data` - Raw datasets, CSV files (`data/raw/emails.csv`), sample generation, data schema/contracts.
     - `artifacts` - Model serialization (`save_model`, `load_model`), saved `.joblib` files, `metrics.json` outputs.
     - `tests` - Test suites (`tests/`), fixtures, synthetic test samples.
     - `deps` - Dependencies, `pyproject.toml`, packaging.
     - `docs` - Project documentation, README, coursework guides.
     - `tooling` - Linter (`ruff`), test runner (`pytest`), project scripts.
     - `ci` - CI workflows, automated checks, GitHub Actions.
     - `infra` - Git configuration, virtualenv setup, local environment.

3. **`summary` Rules:**
   - Must be written in **English**.
   - Must use the **imperative mood** (e.g., `add`, `fix`, `update`, `optimize`, not `added`, `fixes`, `updating`).
   - Must be concise, specific, and action-oriented.
   - **NEVER** use vague summaries (e.g., `update code`, `fix bug`, `changes`, `refactor`).
   - **Do NOT** end the summary with a period (`.`).

4. **Breaking Changes:**
   - Add `!` immediately after the scope (e.g., `FEAT(cli)!: change required arguments for train command`), OR
   - Include a `BREAKING CHANGE:` footer at the bottom of the commit body.

5. **Commit Message Validation Checklist for Agents:**
   Before suggesting or committing any change, the agent must check:
   - [ ] Is `TYPE` in uppercase? (e.g., `FEAT`, `FIX`, `DATA`, `TEST`, `DOCS`)
   - [ ] Is `scope` lowercase and matches one of the project's recommended scopes?
   - [ ] Is `summary` in English, imperative mood, without a trailing period?
   - [ ] Is the summary descriptive and free of vague words like `update` or `fix`?
   - [ ] Are breaking changes explicitly tagged if dataset contracts or CLI APIs change?

---

## 2. Git & Execution Safety Rules

1. **No Unauthorized Commits or Pushes:**
   - Do **NOT** automatically run `git commit` or `git push` without explicit user confirmation and approval.
   - Always present the proposed commit message (following the convention above) to the user first.
2. **Quality Verification Before Proposing Commits:**
   - Verify that all unit tests pass by running `pytest`.
   - Verify that code formatting and linting pass by running `ruff check` and `ruff format --check`.

---

## 3. Data Safety & Privacy Guidelines

1. **No Sensitive Data in Git:**
   - **NEVER** commit real, private, or sensitive emails into the repository.
   - Only sample, anonymized, or public synthetic data should ever be added to [data/raw/emails.csv](data/raw/emails.csv).
2. **Adhere to `.gitignore`:**
   - Large raw datasets under `data/raw/*` and binary model weights under `artifacts/*` must not be force-added to git.
3. **Dataset Contract Enforcement:**
   - Input CSV datasets must contain a `label` column (`spam`/`ham`, `1`/`0`) and at least one text column (`text`, `subject`, `body`, `sender`, `recipient`).

---

## 4. Machine Learning & Code Guidelines

1. **Determinism:**
   - Always set `random_state` in data splitting and stochastic estimators to guarantee reproducible coursework results.
2. **Imbalanced Class Metrics:**
   - Spam email datasets are typically class-imbalanced. Do not rely solely on accuracy. Always evaluate and report `spam_f1`, `spam_precision`, and `spam_recall`.
3. **Safe Model Persistence:**
   - Always verify and create target directories before saving artifacts (`target.parent.mkdir(parents=True, exist_ok=True)`).
