"""Exception hierarchy for BOPP."""

__all__ = [
    "BoppArgumentError",
    "BoppArrayError",
    "BoppError",
    "BoppIOError",
    "BoppRegistryError",
    "BoppValidationError",
]


class BoppError(Exception):
    """Base exception class for all errors raised by the BOPP package."""


class BoppValidationError(BoppError):
    """Raised when object schema or structure validation fails."""


class BoppArrayError(BoppValidationError):
    """Raised for errors related to array structures or array length mismatches."""


class BoppRegistryError(BoppError, KeyError):
    """Raised when an unregistered kind identifier is requested."""


class BoppArgumentError(BoppError, ValueError):
    """Raised when invalid or unconsumed keyword arguments are passed."""


class BoppIOError(BoppError):
    """Raised for errors during reading, writing, or parsing BOPP data formats."""
