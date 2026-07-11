import re
from pathlib import Path

FORBIDDEN_TERMS = (
    "palaestra",
    "hydria",
    "NormalizedForecast",
    "TensorBatch",
    "TensorScalarBatch",
    "group_ids",
    "input_end",
    "target_was_filled",
    "coords",
    "pad_mask",
)

FORBIDDEN_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, FORBIDDEN_TERMS)) + r")\b")


def test_shipped_contract_has_no_private_vocabulary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = sorted((repo_root / "src" / "hcx").rglob("*.py"))
    paths.extend((repo_root / "README.md", repo_root / "docs" / "spec.md"))
    leaks: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in FORBIDDEN_PATTERN.finditer(line):
                relative_path = path.relative_to(repo_root)
                leaks.append(f"{relative_path}:{line_number}: {match.group(0)}")
    assert leaks == []
