"""Local credit-bond trend modeling toolkit."""

from __future__ import annotations

import os


def _patch_windows_realpath_for_virtual_drives() -> None:
    """Avoid PyTorch import failures on Windows virtual or unusual drives."""
    if os.name != "nt":
        return

    import ntpath

    if getattr(ntpath.realpath, "_creditbond_ai_safe", False):
        return

    original_realpath = ntpath.realpath

    def safe_realpath(path, *args, **kwargs):
        try:
            return original_realpath(path, *args, **kwargs)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1005:
                return ntpath.abspath(path)
            raise

    safe_realpath._creditbond_ai_safe = True
    ntpath.realpath = safe_realpath
    os.path.realpath = safe_realpath


_patch_windows_realpath_for_virtual_drives()

__all__ = ["__version__"]

__version__ = "0.1.0"
