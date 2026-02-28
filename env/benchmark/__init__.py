from .spec import BenchmarkSpec
from .presets import IPHYRE_SPEC


def get_spec(env_name: str) -> BenchmarkSpec:
    """Return the BenchmarkSpec for the given env_name prefix."""
    if env_name.startswith("Iphyre"):
        return IPHYRE_SPEC
    raise ValueError(f"No BenchmarkSpec registered for env '{env_name}'.")


def resolve_benchmark(args) -> BenchmarkSpec:
    """
    Convenience wrapper: look up the spec by args.env_name and apply
    defaults back into args (fills only unset / None fields).
    Returns the spec.
    """
    spec = get_spec(args.env_name)
    spec.apply_to(args)
    return spec
