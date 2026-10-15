"""Unit tests for commit message and PR title validation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.validate_commits import (
    ALLOWED_TYPES,
    ValidationResult,
    emit_github_annotations,
    fetch_pr_commits,
    format_cli_report,
    main,
    suggest_valid_commit,
    validate_commit_message,
    write_github_step_summary,
)


# 1. Valid commit messages for all allowed types
@pytest.mark.parametrize("commit_type", sorted(ALLOWED_TYPES))
def test_valid_commit_types(commit_type: str):
    msg = f"{commit_type}(model): implement new functionality"
    errors = validate_commit_message(msg)
    assert errors == []


def test_valid_commit_with_subscope():
    msg = "FEAT(model/svm): tune linear svc hyperparameters"
    assert validate_commit_message(msg) == []


def test_valid_commit_with_hyphenated_scope():
    msg = "FIX(pre-process): handle empty whitespace lines"
    assert validate_commit_message(msg) == []


# 2. Lowercase and mixed-case types
@pytest.mark.parametrize(
    "invalid_type, expected_upper",
    [
        ("feat", "FEAT"),
        ("fix", "FIX"),
        ("Feat", "FEAT"),
        ("Fix", "FIX"),
        ("docs", "DOCS"),
        ("Chore", "CHORE"),
        ("Refactor", "REFACT"),
    ],
)
def test_lowercase_and_mixed_case_types(invalid_type: str, expected_upper: str):
    msg = f"{invalid_type}(model): add logistic regression"
    errors = validate_commit_message(msg)
    assert any("TYPE must be uppercase" in err for err in errors)


# 3. Unsupported types
@pytest.mark.parametrize(
    "unsupported_type",
    ["FOO", "BUG", "UPDATE", "TASK", "FEATURE", "HOTFIX", "CLEANUP"],
)
def test_unsupported_types(unsupported_type: str):
    msg = f"{unsupported_type}(model): add new model"
    errors = validate_commit_message(msg)
    assert any("Unsupported TYPE" in err for err in errors)


# 4. Scope validation: missing, empty, uppercase, invalid characters
def test_missing_scope():
    msg = "FEAT: add logistic regression"
    errors = validate_commit_message(msg)
    assert any("Scope is required" in err for err in errors)


def test_empty_scope():
    msg = "FEAT(): add logistic regression"
    errors = validate_commit_message(msg)
    assert any("Scope cannot be empty" in err for err in errors)


def test_uppercase_scope():
    msg = "FEAT(Model): add logistic regression"
    errors = validate_commit_message(msg)
    assert any("Scope must be lowercase" in err for err in errors)


def test_mixed_case_scope():
    msg = "FIX(PreProcess): clean text inputs"
    errors = validate_commit_message(msg)
    assert any("Scope must be lowercase" in err for err in errors)


def test_scope_with_spaces():
    msg = "FEAT(my scope): add logistic regression"
    errors = validate_commit_message(msg)
    assert any("Scope contains invalid characters" in err for err in errors)


# 5. Missing or invalid colon / space
def test_missing_colon():
    msg = "FEAT(model) add logistic regression"
    errors = validate_commit_message(msg)
    assert any("Missing colon separator" in err for err in errors)


def test_missing_space_after_colon():
    msg = "FEAT(model):add logistic regression"
    errors = validate_commit_message(msg)
    assert any("A space is required after ':'" in err for err in errors)


# 6. Summary validation: empty, trailing period, vague descriptions
def test_empty_summary():
    msg = "FEAT(model):"
    errors = validate_commit_message(msg)
    assert any("Summary is required and cannot be empty" in err for err in errors)


def test_whitespace_only_summary():
    msg = "FEAT(model):    "
    errors = validate_commit_message(msg)
    assert any("Summary is required and cannot be empty" in err for err in errors)


def test_trailing_period():
    msg = "FEAT(model): add linear svc classifier."
    errors = validate_commit_message(msg)
    assert any("Summary must not end with a period" in err for err in errors)


@pytest.mark.parametrize(
    "vague_summary",
    [
        "fix bug",
        "update",
        "update code",
        "changes",
        "refactor",
        "improve stuff",
        "wip",
        "cleanup",
        "minor changes",
    ],
)
def test_prohibited_vague_summaries(vague_summary: str):
    msg = f"FEAT(model): {vague_summary}"
    errors = validate_commit_message(msg)
    assert any("too vague" in err for err in errors)


# 7. Imperative mood detection
@pytest.mark.parametrize(
    "past_tense_verb",
    ["added", "fixed", "updated", "removed", "implemented", "refactored", "cleaned"],
)
def test_imperative_mood_past_tense(past_tense_verb: str):
    msg = f"FEAT(model): {past_tense_verb} classifier baseline"
    errors = validate_commit_message(msg)
    assert any("Summary should use imperative mood" in err for err in errors)


@pytest.mark.parametrize(
    "gerund_verb",
    ["adding", "fixing", "updating", "removing", "implementing", "cleaning"],
)
def test_imperative_mood_gerund(gerund_verb: str):
    msg = f"FEAT(model): {gerund_verb} classifier baseline"
    errors = validate_commit_message(msg)
    assert any("Summary should use imperative mood" in err for err in errors)


@pytest.mark.parametrize(
    "third_person_verb",
    ["adds", "fixes", "updates", "removes", "implements", "cleans"],
)
def test_imperative_mood_third_person(third_person_verb: str):
    msg = f"FEAT(model): {third_person_verb} classifier baseline"
    errors = validate_commit_message(msg)
    assert any("Summary should use imperative mood" in err for err in errors)


# 8. Breaking change syntax
def test_breaking_change_with_exclamation():
    msg = "FEAT(cli)!: change required arguments for train command"
    assert validate_commit_message(msg) == []


def test_breaking_change_with_footer():
    msg = (
        "FEAT(data): enforce strict email column schema\n\n"
        "BREAKING CHANGE: Datasets without an explicit 'label' column are now rejected."
    )
    assert validate_commit_message(msg) == []


def test_breaking_change_with_hyphen_footer():
    msg = (
        "FEAT(cli): drop deprecated options\n\n"
        "BREAKING-CHANGE: The --legacy-model flag has been removed."
    )
    assert validate_commit_message(msg) == []


# 9. Multiline commit messages
def test_valid_multiline_commit():
    msg = (
        "FEAT(model): add multinomial naive bayes baseline\n\n"
        "- integrate MultinomialNB into pipeline factory\n"
        "- add unit test checking prediction output on sample emails\n"
        "- update CLI choices to include naive_bayes"
    )
    assert validate_commit_message(msg) == []


def test_multiline_commit_missing_blank_line():
    msg = (
        "FEAT(model): add multinomial naive bayes baseline\n"
        "- integrate MultinomialNB into pipeline factory"
    )
    errors = validate_commit_message(msg)
    assert any(
        "Separate the subject from the commit body with a blank line" in err for err in errors
    )


# 10. PR title validation (with and without squash PR suffix)
def test_valid_pr_title():
    title = "FEAT(model): add LinearSVC classifier"
    assert validate_commit_message(title, is_pr_title=True) == []


def test_valid_pr_title_with_github_pr_number_suffix():
    title = "FEAT(model): add LinearSVC classifier (#42)"
    assert validate_commit_message(title, is_pr_title=True) == []


def test_invalid_pr_title():
    title = "update model code"
    errors = validate_commit_message(title, is_pr_title=True)
    assert len(errors) > 0


# 11. Suggestion generation
def test_suggestion_for_lowercase_and_past_tense():
    msg = "feat(model): added classifier baseline."
    suggestion = suggest_valid_commit(msg)
    assert suggestion == "FEAT(model): add classifier baseline"


def test_suggestion_for_missing_scope():
    msg = "FEAT: added classifier baseline"
    suggestion = suggest_valid_commit(msg)
    assert suggestion.startswith("FEAT(")
    assert "add classifier baseline" in suggestion


def test_suggestion_for_vague_summary():
    msg = "FIX(model): fix bug"
    suggestion = suggest_valid_commit(msg)
    assert suggestion.startswith("FIX(model): add specific descriptive action")


# 12. Pagination handling in GitHub API fetching
def test_fetch_pr_commits_pagination():
    page1_data = [
        {"sha": f"sha{i}", "commit": {"message": f"FEAT(model): commit {i}"}} for i in range(100)
    ]
    page2_data = [
        {"sha": f"sha{i}", "commit": {"message": f"FEAT(model): commit {i}"}}
        for i in range(100, 115)
    ]

    call_count = 0

    def mock_urlopen(request):
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()
        if "&page=1" in request.full_url:
            mock_response.read.return_value = json.dumps(page1_data).encode("utf-8")
        elif "&page=2" in request.full_url:
            mock_response.read.return_value = json.dumps(page2_data).encode("utf-8")
        else:
            mock_response.read.return_value = json.dumps([]).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        return mock_response

    with patch("scripts.validate_commits.urllib.request.urlopen", side_effect=mock_urlopen):
        commits = fetch_pr_commits("HerokeyVN/CLASSIFYING-SPAM-EMAILS", 1, token="fake-token")

    assert len(commits) == 115
    assert call_count == 2
    assert commits[0]["sha"] == "sha0"
    assert commits[114]["sha"] == "sha114"


# 13. Multiple invalid commits reporting and format_cli_report
def test_format_cli_report_multiple_failures():
    failures = [
        ValidationResult(
            sha="a1b2c3d4e5",
            message="feat(model): added classifier",
            errors=[
                "TYPE must be uppercase (found 'feat', expected 'FEAT').",
                "Summary should use imperative mood (found past tense 'added', consider 'add').",
            ],
            suggested="FEAT(model): add classifier",
        ),
        ValidationResult(
            sha="f6g7h8i9j0",
            message="FIX(data): fix bug",
            errors=[
                "Summary 'fix bug' is too vague. Provide a specific, action-oriented description."
            ],
            suggested="FIX(data): add specific descriptive action",
        ),
    ]
    report = format_cli_report(failures, pr_title_failure=None)
    assert "❌ Commit Message Validation Failed!" in report
    assert "Commit: a1b2c3d" in report
    assert "Message: feat(model): added classifier" in report
    assert "Expected:\n  FEAT(model): add classifier" in report
    assert "Commit: f6g7h8i" in report
    assert "Message: FIX(data): fix bug" in report
    assert "Expected:\n  FIX(data): add specific descriptive action" in report


def test_format_cli_report_pr_title_failure():
    pr_failure = ValidationResult(
        sha="PR_TITLE",
        message="wip: new changes",
        errors=["Invalid format. Expected 'TYPE(scope): summary' or 'TYPE(scope)!: summary'."],
        suggested="FEAT(model): add descriptive summary",
    )
    report = format_cli_report([], pr_title_failure=pr_failure)
    assert "❌ PR Title Validation Failed!" in report
    assert "PR Title: wip: new changes" in report
    assert "Expected:\n  FEAT(model): add descriptive summary" in report


# 14. GitHub Actions annotations and step summary
def test_emit_github_annotations(capsys):
    failures = [
        ValidationResult(
            sha="1234567890",
            message="feat: test",
            errors=["Scope is required: TYPE(scope): summary."],
            suggested="FEAT(model): test",
        )
    ]
    pr_failure = ValidationResult(
        sha="PR_TITLE",
        message="bad title",
        errors=["Invalid format."],
        suggested="FEAT(model): add feature",
    )
    emit_github_annotations(failures, pr_failure)
    captured = capsys.readouterr().out
    assert "::error title=Invalid PR Title::" in captured
    assert "::error title=Invalid Commit (1234567)::" in captured


def test_write_github_step_summary(tmp_path, monkeypatch):
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    failures = [
        ValidationResult(
            sha="1234567890",
            message="feat: test",
            errors=["Scope is required: TYPE(scope): summary."],
            suggested="FEAT(model): test",
        )
    ]
    write_github_step_summary(failures, pr_title_failure=None, total_commits=1)

    content = summary_file.read_text(encoding="utf-8")
    assert "Commit Convention Validation Failed" in content
    assert "`1234567`" in content
    assert "git commit --amend" in content


# 15. CLI main invocation tests
def test_main_cli_single_valid_message():
    assert main(["--message", "FEAT(model): add baseline classifier"]) == 0


def test_main_cli_single_invalid_message():
    assert main(["--message", "feat(model): added baseline classifier."]) == 1


def test_main_cli_commits_json(tmp_path):
    json_file = tmp_path / "commits.json"
    json_file.write_text(
        json.dumps(
            [
                {"sha": "abc1234", "message": "FEAT(model): add svm classifier"},
                {"sha": "def5678", "message": "DOCS(readme): update setup instructions"},
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--commits-json", str(json_file)]) == 0

    # Add an invalid commit
    json_file.write_text(
        json.dumps(
            [
                {"sha": "abc1234", "message": "FEAT(model): add svm classifier"},
                {"sha": "bad9999", "message": "fix bug"},
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--commits-json", str(json_file)]) == 1


def test_main_cli_missing_repo_or_pr(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert main([]) == 2
