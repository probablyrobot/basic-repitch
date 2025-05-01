"""Schema definitions for GuitarSet dataset."""

from .base import BaseSchema, ObjectSchema


class GuitarSetSchema(ObjectSchema):
    """Schema for GuitarSet dataset annotations."""

    def __init__(self):
        """Initialize the GuitarSet schema."""
        super().__init__(
            name="guitarset",
            description="Schema for GuitarSet dataset annotations",
            properties={
                "audio_path": BaseSchema("audio_path", "Path to audio file"),
                "annotation_path": BaseSchema("annotation_path", "Path to annotation file"),
                "track_id": BaseSchema("track_id", "Track identifier"),
            },
            required=["audio_path", "annotation_path", "track_id"],
        )
