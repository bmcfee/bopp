import pytest

from bopp._version import get_current_schema_version, get_registry_version
from bopp.exceptions import BoppRegistryError


def test_get_current_schema_version():
    assert get_current_schema_version() == "1.0"


def test_get_registry_version_valid():
    assert get_registry_version() == "v1"
    assert get_registry_version("1.0") == "v1"
    assert get_registry_version("1.1") == "v1"
    assert get_registry_version("1.12") == "v1"
    assert get_registry_version("1") == "v1"
    assert get_registry_version("v1") == "v1"


def test_get_registry_version_invalid():
    with pytest.raises(BoppRegistryError):
        get_registry_version("2.0")

    with pytest.raises(BoppRegistryError):
        get_registry_version("0.1")

    with pytest.raises(BoppRegistryError):
        get_registry_version("1.a")
