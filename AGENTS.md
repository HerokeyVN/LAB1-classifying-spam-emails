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
     - `data` - Datasets, sample additions (`data/samples/emails.csv`), data schema/contracts.
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

6. **Automated Enforcement & Local Validation:**
   - All commits in Pull Requests and PR titles are validated in CI via [.github/workflows/commit-validation.yml](.github/workflows/commit-validation.yml).
   - Agents can validate proposed commit messages locally before proposing them:
     ```bash
     .venv/bin/python scripts/validate_commits.py --message "FEAT(model): add LinearSVC classifier"
     ```

---

## 2. Git Workflow & Main Branch Protection Rules

### Safe Git Synchronization

- Before creating a new feature branch, fetch the latest origin/main.
- Before modifying the repository, inspect the current branch and working tree using `git status`.
- Never discard, overwrite, or reset uncommitted changes.
- Never automatically switch branches when doing so could affect existing work.
- Do not run `git reset --hard`, `git clean -fd`, or destructive commands without explicit approval.
- Synchronize feature branches with origin/main before opening a Pull Request.
- Prefer merging origin/main into shared feature branches.
- Never rebase a shared branch without coordinating with its contributors.

### Practical Multi-Contributor Synchronization Strategy

To prevent unnecessary conflict churn across concurrent developers/agents:
- **Branch Creation:** Always base new feature branches on the freshest `origin/main` (`git fetch origin main`).
- **During Development:** Commit changes normally, frequently, and atomically on the feature branch. Do **NOT** force a rebase before every single local commit.
- **Before Opening PR / Merging:** Synchronize with `origin/main` prior to opening a Pull Request or proposing a merge. Synchronization before PR/merge is the essential gate—not rebasing before every individual commit.
- **Shared Branch Safety:** When collaborating on a shared feature branch, **always prefer merging `origin/main` (`git merge origin/main`)** rather than rebasing to avoid rewriting shared commit history.

### Main Branch Protection & Conflict Prevention

- **No Direct Commits/Pushes to `main`:** The `main` branch is protected and must always remain stable, tested, and deployable. AI agents and developers must **NOT** commit or push directly to `main` for regular feature, fix, or refactoring work.
- **Feature Branch Workflow:** All development work must occur in dedicated feature/topic branches branched from the latest `origin/main`.
- **Branch Naming Standard:**
  - `feat/<short-name>` (e.g., `feat/linear-svm-tuning`)
  - `fix/<short-name>` (e.g., `fix/handle-empty-emails`)
  - `docs/<short-name>` (e.g., `docs/update-usage-guide`)
  - `refact/<short-name>` (e.g., `refact/extract-vectorizer-config`)
  - `chore/<short-name>` (e.g., `chore/bump-dependencies`)
  - `data/<short-name>` (e.g., `data/add-sample-spam`)
- **Strictly No Force Pushes:** Never run `git push --force` or `git push -f` on `main`. Force-pushing to protected branches is strictly prohibited.
- **Short-Lived Branches:** Keep feature branches small, atomic, and short-lived to minimize divergence and prevent merge conflicts across concurrent contributors.

### Execution & Commit Authorization

- Do **NOT** automatically run `git commit` or `git push` without explicit user confirmation and approval.
- Always present the proposed commit message (following the convention above) and target branch to the user first.

### Quality Verification Policy

- Run relevant tests and lint checks after code changes.
- Run the full test suite before proposing a PR when feasible.
- Distinguish pre-existing failures from failures introduced by the current task.
- Never claim that tests passed unless they were actually executed.
- If verification cannot run, explain the blocker and report which checks were skipped.
- Do not modify unrelated code just to make tests pass.
- Never install dependencies or recreate the virtual environment without user approval.

---

## 3. Scope Control & Minimal Changes

- Only modify files directly related to the requested task.
- Never refactor unrelated modules without explicit approval.
- Preserve existing public APIs unless a breaking change is requested.
- Do not introduce new dependencies unless necessary.
- Do not modify project configuration, CI workflows, or Git hooks without approval.
- Always summarize modified files and explain significant changes.
- Ask for clarification when requirements are ambiguous and a wrong assumption could cause substantial rework.

---

## 4. Data Safety & Privacy Guidelines

1. **Clear Directory Separation:**
   ```text
   data/
   ├── raw/               # Real/raw datasets (strictly gitignored)
   ├── samples/           # Safe sample datasets for testing & team use (tracked)
   │   └── emails.csv
   └── processed/         # Generated or cleaned data (strictly gitignored)
   ```
2. **No Sensitive Data in Git:**
   - **NEVER** commit real, private, or sensitive emails into the repository.
   - Any raw, un-anonymized email dumps must strictly stay in `data/raw/` and remain gitignored.
   - Only small, anonymized, or public synthetic sample data belongs in [data/samples/emails.csv](data/samples/emails.csv) so all contributors can run and test code immediately without privacy risks.
3. **Adhere to `.gitignore`:**
   - All files under `data/raw/*` and `data/processed/*` (except `.gitkeep`) and binary weights under `artifacts/*` (except `.gitkeep`) must never be added to git.
4. **Dataset Contract Enforcement:**
   - Input CSV datasets must contain a `label` column (`spam`/`ham`, `1`/`0`) and at least one text column (`text`, `subject`, `body`, `sender`, `recipient`).

---

## 5. Machine Learning & Code Guidelines

1. **Determinism:**
   - Always set `random_state` in data splitting and stochastic estimators to guarantee reproducible coursework results.
2. **Imbalanced Class Metrics:**
   - Spam email datasets are typically class-imbalanced. Do not rely solely on accuracy. Always evaluate and report `spam_f1`, `spam_precision`, and `spam_recall`.
3. **Safe Model Persistence:**
   - Always verify and create target directories before saving artifacts (`target.parent.mkdir(parents=True, exist_ok=True)`).

---

## 6. Essential Development Commands

Always use the virtual environment located at `.venv/`:

```bash
# Run test suite
.venv/bin/pytest

# Code formatting and linting
.venv/bin/ruff check .
.venv/bin/ruff format --check .

# Validate commit message convention locally
.venv/bin/python scripts/validate_commits.py --message "FEAT(model): add LinearSVC classifier"

# Train model
.venv/bin/python -m spam_classifier.cli train --data data/samples/emails.csv --model logistic_regression

# Evaluate model
.venv/bin/python -m spam_classifier.cli evaluate --data data/samples/emails.csv --model-path artifacts/model.joblib

# Predict single email
.venv/bin/python -m spam_classifier.cli predict --model-path artifacts/model.joblib --text "Free cash prize now!"
```

