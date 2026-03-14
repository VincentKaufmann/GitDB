"""GitDB Schema — dependency-free JSON Schema validation for metadata."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class SchemaError(ValueError):
    """Raised when metadata fails schema validation."""
    pass


class Schema:
    """Simple JSON Schema validator (subset). No external deps.

    Supports:
      - required: list of required field names
      - properties: per-field type/enum/minimum/maximum/minLength/maxLength/pattern
      - additionalProperties: if False, reject unknown fields
      - type: string, number, integer, boolean, array, object, null
      - enum: list of allowed values
      - minimum/maximum: numeric bounds
      - minLength/maxLength: string length bounds
      - pattern: regex match on strings
    """

    def __init__(self, definition: Dict[str, Any]):
        self.definition = definition
        self.required: List[str] = definition.get("required", [])
        self.properties: Dict[str, Any] = definition.get("properties", {})
        self.additional_properties: bool = definition.get("additionalProperties", True)

    def validate(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate metadata dict against the schema. Returns list of errors (empty = valid)."""
        errors = []

        # Required fields
        for field in self.required:
            if field not in metadata:
                errors.append(f"Missing required field: {field!r}")

        # Property validation
        for field, rules in self.properties.items():
            if field not in metadata:
                continue  # missing optionals handled by required check
            value = metadata[field]
            errors.extend(self._validate_field(field, value, rules))

        # Additional properties check
        if not self.additional_properties:
            allowed = set(self.properties.keys())
            extra = set(metadata.keys()) - allowed
            if extra:
                errors.append(f"Unknown fields not allowed: {sorted(extra)}")

        return errors

    def _validate_field(self, field: str, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate a single field value against its rules."""
        errors = []

        # Type check
        if "type" in rules:
            expected = rules["type"]
            if not _type_check(value, expected):
                errors.append(f"Field {field!r}: expected type {expected!r}, got {type(value).__name__}")
                return errors  # skip further checks if type is wrong

        # Enum
        if "enum" in rules:
            if value not in rules["enum"]:
                errors.append(f"Field {field!r}: value {value!r} not in {rules['enum']}")

        # Numeric bounds
        if "minimum" in rules and isinstance(value, (int, float)):
            if value < rules["minimum"]:
                errors.append(f"Field {field!r}: {value} < minimum {rules['minimum']}")
        if "maximum" in rules and isinstance(value, (int, float)):
            if value > rules["maximum"]:
                errors.append(f"Field {field!r}: {value} > maximum {rules['maximum']}")

        # String length
        if "minLength" in rules and isinstance(value, str):
            if len(value) < rules["minLength"]:
                errors.append(f"Field {field!r}: length {len(value)} < minLength {rules['minLength']}")
        if "maxLength" in rules and isinstance(value, str):
            if len(value) > rules["maxLength"]:
                errors.append(f"Field {field!r}: length {len(value)} > maxLength {rules['maxLength']}")

        # Pattern
        if "pattern" in rules and isinstance(value, str):
            import re
            if not re.search(rules["pattern"], value):
                errors.append(f"Field {field!r}: value {value!r} does not match pattern {rules['pattern']!r}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.definition)

    @classmethod
    def from_file(cls, path: str) -> "Schema":
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.definition, indent=2))


def _type_check(value: Any, expected: str) -> bool:
    """Check a value against a JSON Schema type string."""
    if expected == "string":
        return isinstance(value, str)
    elif expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected == "boolean":
        return isinstance(value, bool)
    elif expected == "array":
        return isinstance(value, list)
    elif expected == "object":
        return isinstance(value, dict)
    elif expected == "null":
        return value is None
    return True  # unknown type, pass
