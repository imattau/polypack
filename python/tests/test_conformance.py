"""Run the shared conformance fixtures against the Python implementation."""

import pytest

from polypack.conformance import load_fixtures, run_fixture

FIXTURES = load_fixtures()


def test_fixtures_load_without_ts_implementation_details():
    assert len(FIXTURES) >= 10
    for fixture in FIXTURES:
        assert fixture["schemaVersion"] == 1
        assert fixture["name"]
        assert fixture["group"]


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_fixture(fixture):
    if fixture.get("group") == "hot-cache-eviction":
        pytest.skip("hot-cache-eviction is out of Python v1 scope")
    run_fixture(fixture)
