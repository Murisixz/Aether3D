"""Implementation switches: values, defaults and errors."""

from __future__ import annotations

import pytest

from aether3d import _features


@pytest.fixture
def switch(monkeypatch):
    monkeypatch.setitem(
        _features.SWITCHES,
        "AETHER3D_TEST_SWITCH",
        _features.Switch("AETHER3D_TEST_SWITCH", True, "a test switch"),
    )
    monkeypatch.delenv("AETHER3D_TEST_SWITCH", raising=False)
    return "AETHER3D_TEST_SWITCH"


def test_unset_or_empty_means_the_default(switch, monkeypatch):
    assert _features.enabled(switch) is True
    monkeypatch.setenv(switch, "")
    assert _features.enabled(switch) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("on", True),
        ("TRUE", True),
        ("yes", True),
        ("0", False),
        ("Off", False),
        ("false", False),
        (" no ", False),
    ],
)
def test_values(switch, monkeypatch, value, expected):
    monkeypatch.setenv(switch, value)
    assert _features.enabled(switch) is expected


def test_an_invalid_value_names_the_variable(switch, monkeypatch):
    monkeypatch.setenv(switch, "maybe")
    with pytest.raises(ValueError, match="AETHER3D_TEST_SWITCH='maybe'"):
        _features.enabled(switch)


def test_the_report_lists_every_switch(switch):
    assert _features.report()[switch] is True
    assert set(_features.report()) == set(_features.SWITCHES)
