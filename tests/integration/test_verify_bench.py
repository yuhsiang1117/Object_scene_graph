"""Regression gate for TargetVerifier: run the labeled offline benchmark
(tests/fixtures/verify_bench) and require accuracy on the non-ambiguous
cases to stay at or above the level measured when the set was built. See
scripts/verify_bench.py for the full per-image report and
docs/DESIGN_AND_ROADMAP.md P1c for how the labels were derived.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

# qwen2.5vl:7b measured accuracy=0.700 (precision=0.846, recall=0.733;
# TP=11 FP=2 FN=4 TN=3 on the 20 non-ambiguous cases) when this set was
# built — lower than hand-picked spot checks suggested, dominated by
# sampling variance on near-identical crops (e.g. 2 of 4 armchair crops of
# the SAME chair were rejected). Floor sits a bit below the measured value
# so normal run-to-run variance doesn't fail this on its own; a real
# regression (prompt/model change that meaningfully hurts accuracy) should
# still trip it.
MIN_ACCURACY = 0.65


@pytest.mark.timeout(600)  # 24 images x up to 2 VLM calls each, ~8-10 min observed
def test_verifier_meets_accuracy_floor():
    from scripts.verify_bench import report, run

    results = run(model="qwen2.5vl:7b", accept_confidence=0.5)
    accuracy = report(results)
    assert accuracy >= MIN_ACCURACY, (
        f"verifier accuracy {accuracy:.3f} on tests/fixtures/verify_bench "
        f"dropped below the {MIN_ACCURACY} floor"
    )
