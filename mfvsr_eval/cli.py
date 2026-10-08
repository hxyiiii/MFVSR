from __future__ import annotations


def parse_models(values: list[str] | None, configured: dict) -> list[str]:
    if not values:
        raise ValueError("at least one --model is required")
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    if "all" in names:
        if len(names) != 1:
            raise ValueError("--model all cannot be combined with other model names")
        return list(configured)
    unknown = sorted(set(names) - set(configured))
    if unknown:
        raise ValueError(f"unknown models {unknown}; configured: {sorted(configured)}")
    return list(dict.fromkeys(names))


def parse_metrics(value: str | None, default: list[str]) -> set[str]:
    if value is None:
        return set(default)
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_gpus(value: str) -> list[str]:
    if value.strip().lower() == "cpu":
        return ["cpu"]
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or any(not item.isdigit() for item in result):
        raise ValueError("--gpus must be 'cpu' or a comma-separated list such as 0,1,2,3")
    return result

