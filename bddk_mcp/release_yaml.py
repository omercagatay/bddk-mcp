"""Bounded, unambiguous YAML parsing for release-significant evidence.

Callers retain ownership of file-system policy and translate these internal
errors into their existing content-free public error contracts.
"""

from __future__ import annotations

from typing import Any

import yaml


class ReleaseYamlError(ValueError):
    """Raised when release-significant YAML cannot be parsed safely."""


class ReleaseYamlReferenceError(ReleaseYamlError):
    """Raised when YAML anchors or aliases make signed content ambiguous."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a duplicate mapping key",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_bounded_release_yaml(payload: bytes | str, *, maximum_bytes: int) -> Any:
    """Parse one bounded YAML document without aliases, anchors, or duplicate keys."""

    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    try:
        payload_size = len(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    except UnicodeError as exc:
        raise ReleaseYamlError("release YAML encoding is invalid") from exc
    if payload_size > maximum_bytes:
        raise ReleaseYamlError("release YAML exceeds its byte limit")

    try:
        if any(isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for token in yaml.scan(payload)):
            raise ReleaseYamlReferenceError("release YAML must not contain aliases or anchors")
        return yaml.load(payload, Loader=_UniqueKeySafeLoader)
    except ReleaseYamlError:
        raise
    except (RecursionError, UnicodeError, yaml.YAMLError) as exc:
        raise ReleaseYamlError("release YAML is invalid") from exc
