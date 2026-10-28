"""Regulatory knowledge pilot: the ontology specification and its reviewed vocabularies.

``ontology/ontology.yml`` at the repository root is the single source of
truth for object types, link types, link sources, proposal statuses and
action types.  Code never redefines those vocabularies; it is tested against
the file (``tests/test_ontology_spec.py``), the same way the MCP tool
registry is tested against its reviewed name lists.
"""

from bddk_mcp.ontology.spec import DEFAULT_SPEC_PATH, OntologySpec, OntologySpecError, load_ontology_spec

__all__ = ("DEFAULT_SPEC_PATH", "OntologySpec", "OntologySpecError", "load_ontology_spec")
