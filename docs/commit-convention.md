# Commit Message Convention

This project follows a structured commit message convention to keep the history readable, searchable, and easy for both humans and AI agents to follow.

## Format

All commit messages must use the following format:

```txt
TYPE(scope): summary
```

`TYPE` is mandatory and MUST be uppercase.

### Valid / Invalid Examples

**Valid:**
- `FEAT(model): add LinearSVC classifier option`
- `FIX(preprocess): handle empty or whitespace-only email text`
- `DOCS(readme): add evaluation instructions and metrics guide`

**Invalid:**
- `feat(model): add LinearSVC classifier option`
- `Fix(preprocess): handle empty or whitespace-only email text`
- `docs(readme): add evaluation instructions and metrics guide`

### General Rules

1. **TYPE** is mandatory and MUST be uppercase. Lowercase or mixed-case types (e.g., `feat`, `Fix`) are strictly prohibited.
2. **scope** must be lowercase.
3. **summary** should be in English.
4. **summary** should be concise and use the imperative mood (e.g., "add", "fix", "change", not "added", "fixes", "changing").
5. Do not end the summary with a period.
6. Prefer one logical change per commit.

---

## Allowed Types

Allowed types are:

- **FEAT**: A new user-facing, CLI, model, or pipeline feature
  - *Example:* `FEAT(model): add LinearSVC classifier option`
- **FIX**: A bug fix in preprocessing, model training, CLI, or metrics
  - *Example:* `FIX(preprocess): handle empty or whitespace-only email text`
- **REFACT**: A code refactoring without changing observable behavior or model contracts
  - *Example:* `REFACT(modeling): extract pipeline builder mapping`
- **DATA**: Changes to dataset samples, data contracts, or data validation
  - *Example:* `DATA(contract): support sender and recipient in field combiner`
- **PERF**: A performance or memory optimization (vectorizer speed, sparse matrix efficiency)
  - *Example:* `PERF(features): prune low-frequency n-grams with min_df threshold`
- **TEST**: Adding, updating, or fixing tests and test fixtures
  - *Example:* `TEST(preprocess): add test case for html entity decoding`
- **DOCS**: Documentation-only changes (README, docstrings, project guides)
  - *Example:* `DOCS(readme): document training workflow and metrics report`
- **STYLE**: Code formatting, whitespace, or ruff formatting adjustments
  - *Example:* `STYLE(cli): format argument parser definitions`
- **CHORE**: Tooling, dependencies, packaging, .gitignore, and environment maintenance
  - *Example:* `CHORE(deps): add pytest and ruff to optional dev dependencies`
- **SEC**: Security improvements (sanitizing email input, preventing unsafe model loading)
  - *Example:* `SEC(data): prevent committing sensitive sample emails`
- **OPS**: CI/CD workflows, automation scripts, environment setup
  - *Example:* `OPS(ci): add automated test and lint workflow`
- **REMOVE**: Removing deprecated models, unused utilities, obsolete datasets, or dead code
  - *Example:* `REMOVE(model): drop deprecated custom tokenizer`

*Note: Lowercase or mixed-case types are not allowed.*

---

## Recommended Scopes

Use the most specific scope that describes the main changed area. If a change spans multiple areas, use the primary one.

- `model` - Model architectures (`logistic_regression`, `linear_svm`, `naive_bayes`), estimators, hyperparameter settings
- `preprocess` - Text cleaning (`clean_text`), HTML stripping, whitespace normalization, email field aggregation (`combine_email_fields`), label normalization (`normalize_labels`)
- `features` - Feature extraction, TF-IDF vectorization, n-grams, stop-words filtering, vocabulary configuration
- `cli` - CLI entry points, argument parsing (`train`, `evaluate`, `predict`), user prompts and console output
- `eval` - Metrics computation (`accuracy`, `spam_precision`, `spam_recall`, `spam_f1`), classification reports, confusion matrix
- `data` - Datasets, sample additions (`data/samples/emails.csv`), raw data contracts, label mappings
- `artifacts` - Model serialization (`save_model`, `load_model`), saved `.joblib` files, `metrics.json` outputs
- `tests` - Unit tests, integration tests, mock datasets, test fixtures
- `deps` - Dependency management, `pyproject.toml`, packaging configuration
- `docs` - Project documentation, README, coursework instructions
- `infra` - Git configuration, virtualenv setup, local execution environment
- `ci` - Continuous integration, automated test runners, GitHub Actions
- `tooling` - Linter (`ruff`), test framework (`pytest`), project scripts

### Scope Examples

- `FEAT(model): add multinomial naive bayes classifier`
- `FIX(preprocess): strip trailing html tags before vectorization`
- `FEAT(cli): support custom test split ratio in train command`
- `TEST(eval): verify f1 score computation with zero division handling`
- `DOCS(readme): update quick start cli instructions`

---

## Summary Style

Summaries must be descriptive and action-oriented. Avoid vague descriptions.

- **Good:**
  - `FIX(preprocess): handle missing text fields gracefully`
  - `FEAT(model): add LinearSVC with balanced class weights`
  - `DOCS(readme): add local development setup instructions`
  - `FEAT(cli): support custom threshold in predict command`
- **Bad:**
  - `FIX(model): fix bug`
  - `CHORE: update`
  - `FEAT: changes`
  - `REFACT(preprocess): improve stuff`

---

## Commit Body

For larger or more complex commits, use an optional body to explain the details.

```txt
FEAT(model): add multinomial naive bayes baseline

- integrate MultinomialNB into pipeline factory
- add unit test checking prediction output on sample emails
- update CLI choices to include naive_bayes
- evaluate baseline f1-score against hold-out split
```

- Separate the summary from the body with a blank line.
- Use bullet points for multi-part changes.
- Explain *why* the change was made if the reason is not obvious.
- Keep the body concise.

---

## Breaking Changes

To indicate a breaking change, add a `!` after the scope, or add a `BREAKING CHANGE:` footer at the bottom of the commit body.

### Syntax 1: Using `!` after scope

```txt
FEAT(cli)!: change required arguments for train command
```

### Syntax 2: Using a footer

```txt
FEAT(data): enforce strict email column schema

BREAKING CHANGE: Datasets without an explicit 'label' column are now rejected with a validation error.
```

### When to use Breaking Changes

- Dataset contract changes (e.g., removing support for a column format or changing label conventions)
- Incompatible model artifact serialization (older `.joblib` models cannot be loaded)
- CLI argument renames or removals that break existing scripts
- Incompatible changes to pipeline output format (`metrics.json` structure)

---

## Multi-Area Changes

To keep the git history clean, prefer splitting changes across separate commits:

1. `DATA(contract): add support for sender domain in field combiner`
2. `FEAT(preprocess): extract sender domain during text aggregation`
3. `FEAT(features): include sublinear tf scaling in tfidf vectorizer`
4. `TEST(preprocess): add test case for sender domain extraction`

*Note: If the change is small and tightly coupled, one single commit is acceptable using the primary scope.*

---

## Project-Specific Reference Examples

### Preprocessing & Data
- `FEAT(preprocess): strip html tags and email headers`
- `FIX(preprocess): handle empty or whitespace-only email text`
- `DATA(data): add edge-case spam samples with obfuscated words`
- `FIX(data): standardize label mapping for 'not spam' to ham`
- `TEST(preprocess): add test case for html entity decoding`

### Modeling & Features
- `FEAT(model): add MultinomialNB classifier baseline`
- `FEAT(features): support char n-grams in TF-IDF pipeline`
- `PERF(features): prune low-frequency n-grams with min_df threshold`
- `FIX(model): set zero_division=0 in precision_recall_fscore`
- `TEST(model): verify pipeline training on imbalanced classes`

### CLI & Artifacts
- `FEAT(cli): add --confusion-matrix flag to evaluate command`
- `FIX(cli): ensure output directory exists before saving model`
- `FEAT(eval): output spam-specific F1-score in metrics report`
- `FEAT(artifacts): write training parameters to metrics.json metadata`
- `FIX(artifacts): handle missing model file gracefully during predict`

### Tooling & Documentation
- `DOCS(readme): document train and evaluate workflow with sample data`
- `DOCS(readme): add explanation of precision vs recall trade-off`
- `CHORE(deps): add pytest and ruff to optional dev dependencies`
- `CHORE(ruff): configure lint rules and line length in pyproject.toml`
- `OPS(ci): run pytest and ruff check on pull requests`

---

## For AI Agents

AI agents cooperating in this workspace must follow these rules:

1. **Adherence:** Always suggest commit messages that follow this convention.
2. **Precision:** Never suggest vague summaries (e.g., `update`, `bug fix`, `refactor`).
3. **ML Scopes:** Use specific scopes: `model`, `preprocess`, `features`, `eval`, `cli`, `data`, `artifacts`, `tests`, `docs`, `deps`, `tooling`.
4. **Breaking Changes:** Explicitly mark any breaking changes with the `!` scope modifier or the `BREAKING CHANGE:` footer (especially when modifying dataset contracts, CLI interfaces, or serialized model formats).
5. **Casing Rules:**
   - Always write `TYPE` in uppercase.
   - Never suggest lowercase or mixed-case commit types.
   - If the user provides a lowercase type, normalize it to uppercase.

---

## Automated CI Validation & Enforcement

An automated GitHub Actions workflow (`.github/workflows/commit-validation.yml`) enforces this convention on all Pull Requests targeting `main`.

### Workflow Triggers & Behavior
- **Trigger Events:** Runs on PR `opened`, `reopened`, `synchronize` (new commits pushed), and `edited` (PR title changed).
- **Target Branch:** Pull Requests targeting `main`.
- **Validation Scope:**
  - Validates **every individual commit** in the PR by retrieving the commit history via GitHub REST API with pagination (supporting PRs with >100 commits).
  - Validates the **PR title** as well. This ensures that GitHub's "Squash and Merge" feature—which defaults to using the PR title as the squashed commit summary on `main`—does not produce a non-compliant commit.
- **Permissions:** Minimal read-only permissions (`contents: read`, `pull-requests: read`).

### Failure Reporting
When validation fails, GitHub Actions provides clear diagnostics in three places:
1. **Workflow Console:** Shows the short commit SHA, the commit message, the specific rule(s) violated, and an expected valid suggestion.
2. **Step Summary (`$GITHUB_STEP_SUMMARY`):** Renders a structured Markdown table of all invalid commits and PR title violations with recommended fixes.
3. **Workflow Annotations:** Emits GitHub Actions inline error annotations targeting the invalid commits.

### How to Fix Invalid Commits Locally

If the workflow fails, developers can fix the commit history before merging:

1. **Fixing the most recent commit:**
   ```bash
   git commit --amend -m "TYPE(scope): summary"
   git push --force-with-lease
   ```

2. **Fixing older commits in a branch:**
   ```bash
   git rebase -i origin/main
   # In the editor, change 'pick' to 'reword' (or 'r') for each invalid commit.
   # Save and edit each commit message when prompted.
   git push --force-with-lease
   ```

3. **Fixing an invalid PR title:**
   - In GitHub's web interface, click **Edit** next to the PR title, change it to follow `TYPE(scope): summary`, and click **Save**.
   - Or using GitHub CLI:
     ```bash
     gh pr edit <PR_NUMBER> --title "TYPE(scope): summary"
     ```

4. **Validating locally before pushing:**
   ```bash
   .venv/bin/python scripts/validate_commits.py --message "FEAT(model): add LinearSVC classifier"
   ```

### GitHub Branch Protection Configuration

To require that all PRs pass commit message and title validation before merging into `main`:

1. Navigate to your GitHub repository: **Settings** -> **Branches**.
2. Under **Branch protection rules**, click **Add branch ruleset** or **Add rule** (Branch name pattern: `main`).
3. Enable **Require status checks to pass before merging**.
4. In the search box under "Status checks that're required", search for and select:
   - `Validate PR Commits & Title` (matches the job name in `.github/workflows/commit-validation.yml`).
5. (Recommended) Enable **Require branches to be up to date before merging**.
6. (Recommended) Enable **Require linear history** and **Do not allow bypassing the above settings**.
7. Click **Save changes** / **Create**.

### Linguistic Limitations of Automated Checks

The automated validator enforces deterministic syntactic rules:
- Format structure `TYPE(scope): summary` and `TYPE(scope)!: summary`.
- Uppercase allowed type check and lowercase scope syntax.
- Non-empty summary, presence of colon and space, and absence of trailing periods.
- Rejection of common non-imperative verb forms (past tense `-ed`, gerund `-ing`, 3rd-person singular `-s`) using an extensive verb lookup table.
- Rejection of explicitly prohibited vague summaries (`fix bug`, `update code`, `wip`, etc.).

**Limitations:**
Automated regex and dictionary checks cannot evaluate semantic appropriateness, English grammar outside dictionary verbs, or subjective descriptive accuracy. Code reviewers and team members must verify that the summary accurately reflects the technical change during PR review.
