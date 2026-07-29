__version__ = "2.3.2.post1"

__all__ = [
    "Mamba",
    "Mamba2",
    "Mamba3",
    "MambaLMHeadModel",
    "mamba_inner_fn",
    "selective_scan_fn",
]


def __getattr__(name):
    if name in {"selective_scan_fn", "mamba_inner_fn"}:
        try:
            from mamba_ssm.ops.selective_scan_interface import (
                mamba_inner_fn,
                selective_scan_fn,
            )
        except ModuleNotFoundError:
            selective_scan_fn = None
            mamba_inner_fn = None
        globals()["selective_scan_fn"] = selective_scan_fn
        globals()["mamba_inner_fn"] = mamba_inner_fn
        return globals()[name]

    if name == "Mamba":
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ModuleNotFoundError:
            Mamba = None
        globals()[name] = Mamba
        return Mamba

    if name == "Mamba2":
        from mamba_ssm.modules.mamba2 import Mamba2

        globals()[name] = Mamba2
        return Mamba2

    if name == "Mamba3":
        from mamba_ssm.modules.mamba3 import Mamba3

        globals()[name] = Mamba3
        return Mamba3

    if name == "MambaLMHeadModel":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        globals()[name] = MambaLMHeadModel
        return MambaLMHeadModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
