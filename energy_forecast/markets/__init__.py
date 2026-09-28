from .base import Market, Zone


def get_market(name: str) -> Market:
    name = name.lower()
    if name == "france":
        from .france import France
        return France()
    if name == "italy":
        from .italy import Italy
        return Italy()
    raise ValueError(f"unknown market {name!r}")


__all__ = ["Market", "Zone", "get_market"]
