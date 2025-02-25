# Generated by devtools/yamaker.

PY3TEST()

SUBSCRIBER(g:python-contrib)

VERSION(1.11.4)

ORIGINAL_SOURCE(mirror://pypi/s/scipy/scipy-1.11.4.tar.gz)

SIZE(MEDIUM)

FORK_SUBTESTS()

PEERDIR(
    contrib/python/scipy/py3/tests
)

NO_LINT()

DATA(
    arcadia/contrib/python/scipy/py3
)

SRCDIR(contrib/python/scipy/py3)

TEST_SRCS(
    scipy/stats/tests/__init__.py
    scipy/stats/tests/common_tests.py
    scipy/stats/tests/data/fisher_exact_results_from_r.py
    scipy/stats/tests/test_axis_nan_policy.py
    scipy/stats/tests/test_binned_statistic.py
    scipy/stats/tests/test_boost_ufuncs.py
    scipy/stats/tests/test_censored_data.py
    scipy/stats/tests/test_contingency.py
    scipy/stats/tests/test_continuous_basic.py
    scipy/stats/tests/test_continuous_fit_censored.py
    scipy/stats/tests/test_crosstab.py
    scipy/stats/tests/test_discrete_basic.py
    scipy/stats/tests/test_discrete_distns.py
    scipy/stats/tests/test_entropy.py
    scipy/stats/tests/test_fit.py
    scipy/stats/tests/test_hypotests.py
    scipy/stats/tests/test_morestats.py
    scipy/stats/tests/test_mstats_basic.py
    scipy/stats/tests/test_mstats_extras.py
    scipy/stats/tests/test_multicomp.py
    scipy/stats/tests/test_odds_ratio.py
    scipy/stats/tests/test_qmc.py
    scipy/stats/tests/test_rank.py
    scipy/stats/tests/test_relative_risk.py
    scipy/stats/tests/test_resampling.py
    scipy/stats/tests/test_sampling.py
    scipy/stats/tests/test_sensitivity_analysis.py
    scipy/stats/tests/test_stats.py
    scipy/stats/tests/test_survival.py
    scipy/stats/tests/test_tukeylambda_stats.py
    scipy/stats/tests/test_variation.py
)

END()
