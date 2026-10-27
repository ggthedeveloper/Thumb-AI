import json
import pytest
from thumb_ai.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "thumb-ai" in captured.out


def test_cli_no_args(capsys):
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "thumb-ai" in captured.out


def test_cli_simulate(capsys):
    rc = main(["simulate", "--scenario", "converging"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "THUMB-AI" in captured.out
    assert "DECISION" in captured.out


def test_cli_simulate_json(capsys):
    rc = main(["simulate", "--scenario", "converging", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "action" in data
    assert "tcv" in data
    assert "confidence" in data
