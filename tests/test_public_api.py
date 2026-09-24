"""Contract checks for the package's explicitly supported public surface"""

from __future__ import annotations

import modems


def test_all_exports_exist_and_are_unique() -> None:
    """Every declared public name resolves, and accidental duplicates are rejected"""
    assert len(modems.__all__) == len(set(modems.__all__))
    assert all(hasattr(modems, name) for name in modems.__all__)


def test_ablation_reference_variants_are_not_public() -> None:
    """Benchmark-only V0-V2 insertion implementations stay out of the public API"""
    private_variants = {
        "variant_v0_naive",
        "variant_v1_prefiltered",
        "variant_v2_incremental",
        "variant_v3_production",
    }

    assert private_variants.isdisjoint(modems.__all__)
