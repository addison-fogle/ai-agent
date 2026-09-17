import subprocess
import sys
from pathlib import Path


def test_inspection_commands_without_api(tmp_path):
    script = Path(__file__).resolve().parents[1] / "app.py"
    result = subprocess.run(
        [sys.executable, str(script), "--storage", str(tmp_path), "--debug"],
        input="/identity\n/state\n/memories\n/debug\n/unknown\n/quit\n",
        capture_output=True, text=True, timeout=20, check=True,
    )
    for expected in ("Elias: Ready.", '"self_summary": ""', '"name": "Elias"', "Debug off.", "Unknown command"):
        assert expected in result.stdout
    assert result.stderr == ""

