class CaptionAgentError(Exception):
    """Base exception for caption agent."""


class InvalidInputError(CaptionAgentError):
    """Raised when request input does not pass validation."""

