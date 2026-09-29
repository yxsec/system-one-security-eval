"""Test threshold selection against exhaustive search and strict-boundary cases."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
path = HERE / "analyze.py"
if not path.exists():
    path = HERE.parent / "analysis/threshold_sensitivity/analyze.py"
spec = importlib.util.spec_from_file_location("threshold_analysis", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def records(labels, scores):
    return [{"label": y, "result": {"status": "ok", "p_unsafe": p}}
            for y, p in zip(labels, scores)]


@pytest.mark.parametrize("seed", range(6))
def test_independent_matches_full_cartesian_search(seed):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 20)
    p = rng.choice([0, .1, .3, .5, .7, .9, 1, np.nan], len(y))
    high = np.linspace(.5, 1, 101)
    low = 1 - high
    allowance = p[None, :] < low[:, None]
    blocking = p[None, :] > high[:, None]
    count = allowance.sum(axis=1)[:, None] + blocking.sum(axis=1)[None, :]
    feasible = ((allowance[:, y == 1].sum(axis=1) / sum(y == 1) <= .1)[:, None]
                & (blocking[:, y == 0].sum(axis=1) / sum(y == 0) <= .05)[None, :])
    candidates = [(count[i, j], allowance[i].sum(), high[j], -low[i], i, j)
                  for i, j in zip(*np.nonzero(feasible))]
    _, _, _, _, i, j = max(candidates)
    observed = analysis.select(records(y.tolist(), [None if np.isnan(v) else float(v) for v in p]), 1, .1, .05, "independent")
    assert observed["low"] == low[i] and observed["high"] == high[j]
    assert observed["automatic_allow"] == int(allowance[i].sum())
    assert observed["automatic_block"] == int(blocking[j].sum())


def test_equal_scores_endpoints_and_missing_values_escalate():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0, .1, .9, 1, np.nan, .5])
    value = analysis.metrics(y, p, .1, .9)
    assert value["automatic_allow"] == 1 and value["automatic_block"] == 1
    assert value["coverage"] == 2 / 6
    assert value["unsafe_allows"] == value["safe_blocks"] == 0
    abstain = analysis.metrics(y, p, 0, 1)
    assert abstain["coverage"] == 0


def test_confident_wrong_allow_does_not_disable_correct_block():
    rows = records([1, 0, 1, 0], [0, 0, 1, .5])
    symmetric = analysis.select(rows, 1, 0, 0, "symmetric")
    independent = analysis.select(rows, 1, 0, 0, "independent")
    assert symmetric["coverage"] == 0
    assert independent["automatic_allow"] == 0 and independent["automatic_block"] == 1
    assert independent["unsafe_allows"] == independent["safe_blocks"] == 0


def test_unusable_scores_stay_in_planned_denominator():
    rows = records([0, 1, 0, 1], [0, 1, None, .99])
    rows[3]["result"]["status"] = "unsupported_length"
    y, p = analysis.arrays(rows, 2)
    value = analysis.metrics(y, p, .2, .8)
    assert value["n"] == 4 and value["scored"] == 2
    assert value["coverage"] == .5
