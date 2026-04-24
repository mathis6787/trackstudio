"""Runtime defaults for PyTorch-backed vision components."""

from __future__ import annotations

import os

MPS_UNSUPPORTED_OP_HINTS = (
    "not currently implemented for the MPS device",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


def configure_torch_runtime() -> None:
    """Set PyTorch environment defaults before torch is imported."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def is_mps_unsupported_op_error(error: BaseException) -> bool:
    message = str(error)
    return any(hint in message for hint in MPS_UNSUPPORTED_OP_HINTS)

