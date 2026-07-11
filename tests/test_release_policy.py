from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml


def _workflow() -> dict[str, Any]:
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
    parsed = yaml.safe_load(workflow_path.read_text())
    assert isinstance(parsed, dict)
    return cast("dict[str, Any]", parsed)


def _run_commands(job: Mapping[str, Any]) -> list[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    commands: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            command = step.get("run")
            if isinstance(command, str):
                commands.append(command)
    return commands


def _strings(value: object) -> Iterator[tuple[str, bool]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                yield key, True
            yield from _strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value, False


def test_release_triggers_are_human_gated() -> None:
    workflow = _workflow()
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert set(triggers) == {"release", "workflow_dispatch"}, "push and pull_request triggers are forbidden"
    assert triggers["release"] == {"types": ["published"]}

    dispatch = triggers["workflow_dispatch"]
    assert isinstance(dispatch, dict)
    assert dispatch["inputs"]["target"]["options"] == ["testpypi"]


def test_release_jobs_enforce_build_and_routing_policy() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"build", "publish-testpypi", "publish-pypi"}
    assert "uv build" in _run_commands(jobs["build"])

    test_job = jobs["publish-testpypi"]
    pypi_job = jobs["publish-pypi"]
    for job in (test_job, pypi_job):
        assert job["needs"] == "build"
        assert job["permissions"]["id-token"] == "write"

    test_condition = test_job["if"]
    for predicate in (
        "github.event_name == 'workflow_dispatch'",
        "inputs.target == 'testpypi'",
        "github.event_name == 'release'",
        "github.event.release.prerelease == true",
    ):
        assert predicate in test_condition

    pypi_condition = pypi_job["if"]
    assert "github.event_name == 'release'" in pypi_condition
    assert "github.event.release.prerelease == false" in pypi_condition
    assert "workflow_dispatch" not in pypi_condition


def test_publish_commands_use_trusted_publishing() -> None:
    jobs = _workflow()["jobs"]
    test_command = "uv publish --trusted-publishing always --publish-url https://test.pypi.org/legacy/ dist/*"
    pypi_command = "uv publish --trusted-publishing always dist/*"
    assert _run_commands(jobs["publish-testpypi"]) == [test_command]
    assert _run_commands(jobs["publish-pypi"]) == [pypi_command]

    for command in (test_command, pypi_command):
        assert command.startswith("uv publish")
        assert "--trusted-publishing always" in command
    assert "--publish-url" in test_command
    assert "--publish-url" not in pypi_command


def test_workflow_has_no_credential_wiring() -> None:
    for value, is_key in _strings(_workflow()):
        normalized = value.casefold()
        if is_key and normalized == "id-token":
            continue
        for forbidden in ("password", "token", "secret"):
            assert forbidden not in normalized, f"forbidden credential term {forbidden!r} in {value!r}"
