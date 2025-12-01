import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import ttest_ind
from scipy.stats import t

from dataclasses import dataclass
from scipy.stats import permutation_test
import argparse

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None  # We'll fall back if SciPy isn't available

@dataclass
class TwoSampleStats:
    n1: int
    n2: int
    mean1: float
    mean2: float
    sd1: float
    sd2: float
    mean_diff: float
    mean_diff_CI95: tuple
    t: float
    df: float
    p_two_sided: float
    cohen_d: float
    cohen_d_CI95: tuple

def diff_mean(a, b):
    return np.mean(a) - np.mean(b)

def _t_ppf_975(df: float) -> float:
    """Return the 97.5% quantile of Student-t(df)."""
    if student_t is None:
        # Simple normal fallback if SciPy is unavailable (slightly optimistic CI)
        return 1.959963984540054  # ~N(0,1) 97.5% quantile
    return student_t.ppf(0.975, df)

def _t_sf(abs_t: float, df: float) -> float:
    """Survival function (1-CDF) for t, fallback to normal if SciPy missing."""
    if student_t is None:
        # Normal fallback
        import math
        from math import erf, sqrt, exp, pi
        # tail ≈ 0.5 * erfc(|t|/sqrt(2))
        return 0.5 * (1 - erf(abs_t / np.sqrt(2)))
    return student_t.sf(abs_t, df)

def ci95_unweighted_mean(deltas):
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.size
    mean = deltas.mean()
    # sd = deltas.std(ddof=1)
    sd = deltas.std()
    se = sd / np.sqrt(n)
    tcrit = t.ppf(0.975, df=n-1)
    lo, hi = mean - tcrit*se, mean + tcrit*se
    print('se', sd, se, tcrit)
    return mean, (lo, hi), sd

def two_sample_stats(x, y, epsilon: float = 1e-8) -> TwoSampleStats:
    """
    Compute Welch two-sample stats + Cohen's d with CI (independent samples).
    If a sample variance is zero, it is floored to `epsilon` to avoid numerical issues.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")

    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample must have at least 2 observations")

    mean1, mean2 = float(np.mean(x)), float(np.mean(y))
    # sample std (ddof=1)
    s1 = float(np.std(x, ddof=1))
    s2 = float(np.std(y, ddof=1))

    # Floor variances to avoid zeros
    v1 = max(s1**2, epsilon)
    v2 = max(s2**2, epsilon)

    # --- Welch t-test (default) ---
    se2 = v1 / n1 + v2 / n2
    se = np.sqrt(se2)
    t_val = (mean1 - mean2) / se
    # Welch-Satterthwaite df
    df_num = se2**2
    df_den = (v1**2) / (n1**2 * (n1 - 1)) + (v2**2) / (n2**2 * (n2 - 1))
    df = df_num / df_den
    # CI for mean difference (Welch)
    tcrit = _t_ppf_975(df)
    ci_md = ((mean1 - mean2) - tcrit * se, (mean1 - mean2) + tcrit * se)

    p_two_sided = 2.0 * _t_sf(abs(t_val), df)
    
    res = permutation_test(
            (x, y),
            statistic=diff_mean,
            vectorized=False,
            n_resamples=10000,
            alternative="two-sided",
            random_state=0,
            )
    p_perm = res.pvalue

    # --- Cohen's d (pooled SD) and its CI (Hedges & Olkin approx) ---
    # Even if Welch above, Cohen's d is standardly defined with pooled SD.
    df_pooled = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df_pooled
    sp = np.sqrt(sp2)
    d = (mean1 - mean2) / sp

    # Variance of d (Hedges & Olkin, 1985) for independent groups
    # var(d) ≈ (n1+n2)/(n1*n2) + d^2 / (2*(n1+n2-2))
    var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2.0 * df_pooled)
    zcrit = 1.959963984540054  # 97.5% normal quantile
    se_d = np.sqrt(var_d)
    ci_d = (d - zcrit * se_d, d + zcrit * se_d)

    if v1 == epsilon and v2 == epsilon:
        p_value = float(p_perm)
    else:
        p_value = float(p_two_sided)

    return TwoSampleStats(
        n1=n1, n2=n2,
        mean1=mean1, mean2=mean2,
        sd1=np.sqrt(v1), sd2=np.sqrt(v2),
        mean_diff=mean1 - mean2,
        mean_diff_CI95=ci_md,
        t=float(t_val),
        df=float(df),
        p_two_sided=p_value,
        cohen_d=float(d),
        cohen_d_CI95=tuple(map(float, ci_d)),
    )


parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--task', type=str, default='mp1') # 'navigation' or 'mp1'
parser.add_argument('--method', type=str, default='language') # 'usc' or 'language'
args = parser.parse_args()

plot_type = args.task
method_type = args.method

if plot_type == 'navigation':
    plt.rcParams['font.size'] = 15
    col_titles = ['Pepper', 'Drone', 'Wheeled biped', 'Carter']
    col_titles_2 = ['Pepper', 'Drone', 'Wheeled\nbiped', 'Carter']
    dim = 4
    our_result = np.loadtxt('./simulation-result/navigation_ours_single_score.txt', delimiter=',')
    language_result = np.loadtxt('./simulation-result/navigation_language_single_score.txt', delimiter=',')
    density_result = np.loadtxt('./simulation-result/navigation_usc_single_score.txt', delimiter=',')
elif plot_type == 'mp1':
    plt.rcParams['font.size'] = 16
    col_titles = ['UR5_L (5 joints)', 'UR5_F (5 joints)', 'UR5_R (5 joints)']
    col_titles_2 = ['UR5_L\n(5 joints)', 'UR5_F\n(5 joints)', 'UR5_R\n(5 joints)']
    dim = 3
    our_result = np.loadtxt('./simulation-result/mp1_ours_single_score.txt', delimiter=',')
    language_result = np.loadtxt('./simulation-result/mp1_language_single_score.txt', delimiter=',')
    density_result = np.loadtxt('./simulation-result/mp1_usc_single_score.txt', delimiter=',')

navi_names = ['Pepper', 'Drone', 'Wheeled biped', 'Carter']
mp_names = ['UR5_L', 'UR5_F', 'UR5_R']

if plot_type == 'navigation':
    robot_names = navi_names
else:
    robot_names = mp_names

for i in range(len(our_result)):
    ours = our_result[i]
    language = language_result[i]
    usc = density_result[i]

    if method_type == 'language':
        baseline = language
    else:
        baseline = usc

    name_len = len(robot_names)
    demo_name = int(i/name_len)
    learner_name = int(i%name_len)

    epsilon = 1e-16
    result = two_sample_stats(ours, baseline, epsilon)
    
    if result.sd1 == np.sqrt(epsilon) and result.sd2 == np.sqrt(epsilon):
        print(robot_names[demo_name], '-', robot_names[learner_name], 
              'mean_diff:', result.mean_diff, result.mean_diff_CI95, 
              't:', '-', 'df:', '-', 'p:', result.p_two_sided, 'd:', '-')
    else:
        print(robot_names[demo_name], '-', robot_names[learner_name], 
              'mean_diff:', result.mean_diff, result.mean_diff_CI95, 
              't:', result.t, 'df:', result.df, 'p:', result.p_two_sided, 'd:', result.cohen_d, result.cohen_d_CI95)

