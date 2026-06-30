"""Runner for ARCHITECTURE-LEVEL (whole-repo / cross-file) static-analysis test cases.

The module-level runner (`run_fp_tests.py`) feeds ONE file to the per-module agent. This
runner exercises the `codebase_analyst` pipeline, which reasons across an entire mini-repo:
it ingests assets, classifies components, evaluates maturity dimensions, and synthesizes an
architecture review. Stages 1-3 are pure functions and Stage 4's `synthesize_report()` runs
in-memory, so NO database / Qdrant / GitHub is required.

Each test case is a directory under `architecture_cases/<case>/` containing:
  - a small multi-file project (real source + a manifest)
  - an `expected.json` describing the architecture-level assertions, e.g.:
      {
        "description": "agent with no tests / tracing / guardrail",
        "expect_low_dimensions": ["eval", "observability", "guardrail"],
        "expect_not_low_dimensions": [],
        "expect_external_llm_min": 1,
        "expect_unreachable_min": 0,
        "expect_component_min": 1,
        "expect_roles_include": ["agent_executor"]
      }

Usage (from repo root):
    DATABASE_URL='postgresql+asyncpg://ace:ace_password@localhost:5432/ace' \
      uv run python static-fp-testcases/run_architecture_tests.py [--case <name>]

Exit code is non-zero if any case fails an assertion (CI-friendly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://ace:ace_password@localhost:5432/ace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv("debug.env")

from backend.services.codebase_analyst.stages.compose_pattern_classifier import (  # noqa: E402
    classify_components,
)
from backend.services.codebase_analyst.stages.evaluate_rules import evaluate_run  # noqa: E402
from backend.services.codebase_analyst.stages.ingest_endpoint import resolve_endpoints  # noqa: E402
from backend.services.codebase_analyst.stages.ingest_llm_call import detect_llm_calls  # noqa: E402
from backend.services.codebase_analyst.stages.ingest_manifest import analyze_manifests  # noqa: E402
from backend.services.codebase_analyst.stages.ingest_non_code import scan_non_code  # noqa: E402
from backend.services.codebase_analyst.stages.ingest_prompt_template import (  # noqa: E402
    detect_prompt_templates,
)
from backend.services.codebase_analyst.stages.synthesize_report import synthesize_report  # noqa: E402

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "architecture_cases"


def run_pipeline(run_id: str, repo_root: Path) -> dict:
    """Run codebase_analyst Stages 1-4 on a directory; return facts for assertions."""
    assets = (
        analyze_manifests(run_id, repo_root)
        + detect_llm_calls(run_id, repo_root)
        + resolve_endpoints(run_id, repo_root)
        + detect_prompt_templates(run_id, repo_root)
        + scan_non_code(run_id, repo_root)
    )
    # Standalone (no-DB) runs leave ORM-assigned `id` as None on every asset, which
    # collapses `member_assets` to a single None key and breaks downstream lookups.
    # Assign stable synthetic ids before classification (the real pipeline gets these
    # from the DB on insert).
    for i, asset in enumerate(assets):
        if getattr(asset, "id", None) is None:
            asset.id = f"asset-{i}"
    components, edges = classify_components(run_id, assets, repo_root)
    maturities, gaps = evaluate_run(run_id, assets, components)
    report = synthesize_report(maturities, None, components=components, edges=edges, assets=assets)
    arch = report.get("architecture_review") or {}
    return {
        "tiers": {m.dimension: m.tier for m in maturities},
        "external_llm_clusters": arch.get("external_llm_clusters") or [],
        "unreachable_clusters": arch.get("unreachable_clusters") or [],
        "component_count": arch.get("component_count") or 0,
        "roles_present": arch.get("roles_present") or [],
        "gap_severities": [g.severity for g in gaps],
    }


def check(expected: dict, got: dict) -> list[str]:
    """Return a list of assertion-failure messages (empty == pass)."""
    fails: list[str] = []
    tiers = got["tiers"]
    for dim in expected.get("expect_low_dimensions", []):
        if tiers.get(dim) != "low":
            fails.append(f"{dim}: expected tier=low, got {tiers.get(dim)!r}")
    for dim in expected.get("expect_not_low_dimensions", []):
        if tiers.get(dim) == "low":
            fails.append(f"{dim}: expected tier!=low (gave credit), got 'low'")
    if len(got["external_llm_clusters"]) < expected.get("expect_external_llm_min", 0):
        fails.append(
            f"external_llm_clusters: expected >= {expected['expect_external_llm_min']}, "
            f"got {len(got['external_llm_clusters'])}"
        )
    if len(got["unreachable_clusters"]) < expected.get("expect_unreachable_min", 0):
        fails.append(
            f"unreachable_clusters: expected >= {expected['expect_unreachable_min']}, "
            f"got {len(got['unreachable_clusters'])}"
        )
    if got["component_count"] < expected.get("expect_component_min", 0):
        fails.append(
            f"component_count: expected >= {expected['expect_component_min']}, "
            f"got {got['component_count']}"
        )
    for role in expected.get("expect_roles_include", []):
        if role not in got["roles_present"]:
            fails.append(f"roles_present: missing {role!r} (got {got['roles_present']})")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run only this case directory")
    args = ap.parse_args()

    if not CASES_DIR.exists():
        print(f"No cases dir: {CASES_DIR}")
        return 0
    cases = sorted(d for d in CASES_DIR.iterdir() if d.is_dir() and (d / "expected.json").exists())
    if args.case:
        cases = [d for d in cases if d.name == args.case]
    if not cases:
        print("No architecture cases found.")
        return 0

    print(f"Running {len(cases)} architecture-level cases\n")
    passed, failed = 0, 0
    for case in cases:
        expected = json.loads((case / "expected.json").read_text())
        try:
            got = run_pipeline(f"arch-{case.name}", case)
        except Exception as e:  # a crash is itself a failure
            print(f"  ✗ [ERROR] {case.name}: {type(e).__name__}: {e}")
            failed += 1
            continue
        fails = check(expected, got)
        if fails:
            failed += 1
            print(f"  ✗ [FAIL ] {case.name} — {expected.get('description', '')}")
            for f in fails:
                print(f"            {f}")
        else:
            passed += 1
            low = [d for d, t in got["tiers"].items() if t == "low"]
            print(
                f"  ✓ [pass ] {case.name}  (components={got['component_count']}, "
                f"ext_llm={len(got['external_llm_clusters'])}, low_dims={low})"
            )

    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE cases: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
