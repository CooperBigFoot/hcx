import configparser
import importlib.metadata
import subprocess
import zipfile
from pathlib import Path


def test_built_wheel_contains_public_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    expected_version = importlib.metadata.version("hcx")
    wheels = list(tmp_path.glob(f"hcx-{expected_version}-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        expected_sources = {
            "hcx/py.typed",
            "hcx/__init__.py",
            "hcx/batch.py",
            "hcx/conformance.py",
            "hcx/output.py",
            "hcx/protocol.py",
            "hcx/specifications.py",
            "hcx/synthetic.py",
            "hcx/models/__init__.py",
            "hcx/models/lstm.py",
        }
        assert expected_sources <= names

        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        metadata = wheel.read(metadata_paths[0]).decode("utf-8")
        assert "Name: hcx" in metadata
        assert f"Version: {expected_version}" in metadata
        assert "Requires-Python: >=3.11" in metadata

        entry_point_paths = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(entry_point_paths) == 1
        parser = configparser.ConfigParser()
        parser.read_string(wheel.read(entry_point_paths[0]).decode("utf-8"))
        assert parser.has_section("hcx.models")
        assert dict(parser["hcx.models"]) == {"scalar_lstm": "hcx.models.lstm:factory"}
