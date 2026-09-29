"""Recompute the clustered intervals displayed in the paper from exported scores."""

from collections import defaultdict
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from security_eval.paper_analysis import cluster_intervals
from reproduce_results import read, rows


def main():
    data = read('results/heldout_current.json')
    cache = defaultdict(list)
    for row in rows('data/predictions/primary_classification.jsonl.gz'):
        cache[row['dataset'],row['model']].append(row)
    checks = 0

    def compare(actual, expected):
        nonlocal checks
        for key in ('estimate','low','high'):
            a, b = actual[key], expected[key]
            assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11)), (key,a,b)
            checks += 1

    for dataset, value in data['datasets'].items():
        common = set(value['common_ids'])
        for model, saved in value['models'].items():
            matched = [row for row in cache[dataset,model] if row['id'] in common]
            observed = cluster_intervals(matched)
            for key, expected in saved['common_intervals']['metrics'].items():
                compare(observed['metrics'][key],expected)
    from security_eval.paper_analysis import paired_comparison, holm_adjust
    expanded = read('results/added_models.json')
    panel_cache = defaultdict(list)
    for filename in ('primary_classification', 'glm_completed', 'additional_judges'):
        for record in rows(f'data/predictions/{filename}.jsonl.gz'):
            panel_cache[record['dataset'], record['model']].append(record)

    def check_scalar(actual, expected):
        nonlocal checks
        assert (actual is None and expected is None) or (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11)), (actual, expected)
        checks += 1

    for dataset, data in expanded['datasets'].items():
        common = set(data['shared_ids'])
        for model, saved in data['models'].items():
            matched = [r for r in panel_cache[dataset, model] if r['id'] in common]
            observed = cluster_intervals(matched)
            for key, expected in saved['shared_intervals']['metrics'].items():
                compare(observed['metrics'][key], expected)
        contrasts = []
        for saved in data['comparisons_added_minus_jev']:
            observed = paired_comparison(panel_cache[dataset, saved['model']], panel_cache[dataset, 'jev'])
            for key in ('common_n', 'common_probability_n', 'cluster_count', 'left_only_errors', 'right_only_errors', 'mcnemar_exact_iid_p'):
                check_scalar(observed[key], saved[key])
            for metric, fields in observed['metrics'].items():
                for key, actual in fields.items():
                    check_scalar(actual, saved['metrics'][metric][key])
            contrasts.append((observed, saved))
        for metric in ('fnr', 'fpr'):
            adjusted = holm_adjust([observed['metrics'][metric]['cluster_randomization_p'] for observed, _ in contrasts])
            for p, (_, saved) in zip(adjusted, contrasts, strict=True):
                check_scalar(p, saved['metrics'][metric]['holm_p_added_four_family'])
    print(f'Passed {checks} interval estimate/bound checks with 2,000 grouped resamples per cohort.')


if __name__=='__main__':
    main()
