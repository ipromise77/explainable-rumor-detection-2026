"""Rumor detection package for the AI introduction course project."""

__all__ = ["RumourDetectClass", "FinalRumourDetectClass"]


def __getattr__(name):
    if name == "RumourDetectClass":
        from .rumor_detector import RumourDetectClass

        return RumourDetectClass
    if name == "FinalRumourDetectClass":
        from .final_detector import FinalRumourDetectClass

        return FinalRumourDetectClass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
