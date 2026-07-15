from __future__ import annotations

import pytest

from bddk_mcp.release_yaml import ReleaseYamlError, ReleaseYamlReferenceError, load_bounded_release_yaml


def test_release_yaml_enforces_the_supplied_byte_limit() -> None:
    with pytest.raises(ReleaseYamlError, match="exceeds its byte limit"):
        load_bounded_release_yaml("key: value\n", maximum_bytes=4)


@pytest.mark.parametrize(
    "payload",
    (
        "key: &anchor value\n",
        "key: *missing\n",
    ),
)
def test_release_yaml_rejects_anchors_and_aliases_independently(payload: str) -> None:
    with pytest.raises(ReleaseYamlReferenceError, match="must not contain aliases or anchors"):
        load_bounded_release_yaml(payload, maximum_bytes=1_024)


def test_release_yaml_rejects_duplicate_keys_at_any_mapping_depth() -> None:
    with pytest.raises(ReleaseYamlError, match="release YAML is invalid"):
        load_bounded_release_yaml("outer:\n  key: first\n  key: second\n", maximum_bytes=1_024)
