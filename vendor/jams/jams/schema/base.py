"""Base schema definitions for JAMS."""

from typing import Any


class BaseSchema:
    """Base class for all JAMS schemas."""

    def __init__(self, name: str, description: str = ""):
        """Initialize the schema.

        Args:
            name: The schema name
            description: Optional description of the schema
        """
        self.name = name
        self.description = description

    def validate(self, data: Any) -> bool:
        """Validate data against this schema.

        Args:
            data: The data to validate

        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate()")


class ArraySchema(BaseSchema):
    """Schema for array data."""

    def __init__(self, name: str, description: str = "", min_items: int | None = None, max_items: int | None = None):
        """Initialize the array schema.

        Args:
            name: The schema name
            description: Optional description of the schema
            min_items: Minimum number of items (optional)
            max_items: Maximum number of items (optional)
        """
        super().__init__(name, description)
        self.min_items = min_items
        self.max_items = max_items

    def validate(self, data: list[Any]) -> bool:
        """Validate array data.

        Args:
            data: The array data to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, list):
            return False

        if self.min_items is not None and len(data) < self.min_items:
            return False

        if self.max_items is not None and len(data) > self.max_items:
            return False

        return True


class ObjectSchema(BaseSchema):
    """Schema for object data."""

    def __init__(
        self, name: str, properties: dict[str, BaseSchema], description: str = "", required: list[str] | None = None
    ):
        """Initialize the object schema.

        Args:
            name: The schema name
            properties: Dictionary mapping property names to their schemas
            description: Optional description of the schema
            required: List of required property names (optional)
        """
        super().__init__(name, description)
        self.properties = properties
        self.required = required or []

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate object data.

        Args:
            data: The object data to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False

        # Check required properties
        for prop in self.required:
            if prop not in data:
                return False

        # Validate each property
        for prop, value in data.items():
            if prop not in self.properties:
                return False

            if not self.properties[prop].validate(value):
                return False

        return True
