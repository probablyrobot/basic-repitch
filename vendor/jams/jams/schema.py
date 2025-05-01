"""JAMS schema module.

This module contains schema definitions for JAMS annotations.
"""

import importlib.util
import os
from types import ModuleType


def load_source(name: str, path: str) -> ModuleType:
    """Load a Python source file and return a module object.

    Args:
        name: The module name
        path: The path to the source file

    Returns:
        The loaded module object
    """
    if not os.path.exists(path):
        raise ImportError(f"No module named '{name}'")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not load module spec for '{path}'")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module loader for '{path}'")

    spec.loader.exec_module(module)
    return module


def namespace(name: str) -> ModuleType:
    """Load a namespace schema module.

    Args:
        name: The namespace name

    Returns:
        The loaded namespace module
    """
    # Get the directory containing this file
    schema_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for the schema file in the schema directory
    schema_path = os.path.join(schema_dir, "schema", f"{name}.py")

    if not os.path.exists(schema_path):
        raise ImportError(f"No schema found for namespace '{name}'")

    # Load the schema module
    return load_source(f"jams.schema.{name}", schema_path)
