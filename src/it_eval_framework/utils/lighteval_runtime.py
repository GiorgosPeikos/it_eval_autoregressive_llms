from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version


REQUIRED_LIGHTEVAL_VERSION = "0.13.0"
TESTED_DATASETS_VERSION = "3.6.0"
TESTED_TRANSFORMERS_VERSION = "4.57.1"


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def lighteval_environment_report() -> dict:
    versions = {
        "lighteval": installed_version("lighteval"),
        "datasets": installed_version("datasets"),
        "transformers": installed_version("transformers"),
        "accelerate": installed_version("accelerate"),
    }
    errors = []
    warnings = []
    if versions["lighteval"] is None:
        errors.append("LightEval is not installed.")
    elif versions["lighteval"] != REQUIRED_LIGHTEVAL_VERSION:
        errors.append(
            f"Expected lighteval=={REQUIRED_LIGHTEVAL_VERSION}, found {versions['lighteval']}."
        )
    if versions["datasets"] is None:
        errors.append("datasets is not installed.")
    elif versions["datasets"] != TESTED_DATASETS_VERSION:
        errors.append(
            f"Italian tasks are tested with datasets=={TESTED_DATASETS_VERSION}; found {versions['datasets']}."
        )
    if versions["transformers"] is None:
        errors.append("transformers is not installed.")
    elif versions["transformers"] != TESTED_TRANSFORMERS_VERSION:
        warnings.append(
            f"The compatibility path is tested with transformers=={TESTED_TRANSFORMERS_VERSION}; "
            f"found {versions['transformers']}."
        )
    if versions["accelerate"] is None:
        errors.append("accelerate is not installed.")
    return {"versions": versions, "errors": errors, "warnings": warnings}


def require_lighteval_environment() -> dict:
    report = lighteval_environment_report()
    if report["errors"]:
        install = (
            "Install the pinned runtime from the repository: "
            "python -m pip install 'lighteval[multilingual]==0.13.0' --no-deps && "
            "python -m pip install -r constraints/lighteval-python310-313.txt"
        )
        raise RuntimeError(" ".join([*report["errors"], install]))
    return report


def main() -> None:
    report = lighteval_environment_report()
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
