"""
Belief Inertia Analysis (v2)
============================

Tests VFE prediction: high belief inertia → resistance to attitude change.

Fixed issues from v1:
- Regression to mean artifact (extreme scores have more room to regress)
- Better inertia proxy: partisan strength (not attitude extremity)
- New metric: directional stability (staying on same side)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# =============================================================================
# AUTO-DETECT DATA FILE
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
data_files = list(SCRIPT_DIR.glob('*.dta')) + list(SCRIPT_DIR.glob('*.sav')) + list(SCRIPT_DIR.glob('*.csv'))

if data_files:
    DATA_PATH = max(data_files, key=lambda f: f.stat().st_size)
    print(f"Auto-detected: {DATA_PATH.name}")
else:
    print("No data file found!")
    DATA_PATH = None

# =============================================================================
# Load Data
# =============================================================================

def load_data(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.dta':
        return pd.read_stata(path)
    elif ext == '.sav':
        return pd.read_spss(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    return None

if DATA_PATH is None:
    raise SystemExit("No data file found.")

print("Loading data...")
df = load_data(DATA_PATH)
print(f"Loaded {len(df)} cases")

# =============================================================================
# ANES Variable Mappings
# =============================================================================

# Party ID (7-point: 1=Strong Dem, 4=Independent, 7=Strong Rep)
PARTY_ID_VARS = ['V201231x', 'V201228']  # Try both

# Thermometers with pre/post (0-100)
THERMO_VARS = {
    'thermo_dems': ('V201156', 'V202145'),
    'thermo_reps': ('V201157', 'V202146'),
}

# =============================================================================
# Get Party ID (inertia proxy)
# =============================================================================

def convert_to_numeric(series):
    """Convert ANES variable to numeric, handling Stata categories."""
    # Try direct numeric conversion first
    vals = pd.to_numeric(series, errors='coerce')

    # If all NaN, try extracting from categorical codes
    if vals.isna().all():
        if hasattr(series, 'cat'):
            # Categorical - use codes
            vals = series.cat.codes.astype(float)
            vals[vals < 0] = np.nan  # -1 codes are missing
        elif series.dtype == object:
            # String labels - extract leading digits
            vals = series.astype(str).str.extract(r'^(\d+)')[0].astype(float)

    return vals.values

party_id = None
for var in PARTY_ID_VARS:
    if var in df.columns:
        party_id = convert_to_numeric(df[var])
        valid_count = (~np.isnan(party_id)).sum()
        print(f"Party ID variable: {var}")
        print(f"  Valid cases: {valid_count}")
        print(f"  Sample values: {party_id[~np.isnan(party_id)][:10]}")
        if valid_count > 100:
            break
        else:
            party_id = None  # Try next variable

if party_id is None:
    print("No usable party ID variable found")
    partisan_strength = None
else:
    # Partisan STRENGTH (distance from center) = inertia proxy
    # ANES: 1=Strong Dem, 4=Independent, 7=Strong Rep
    partisan_strength = np.abs(party_id - 4)  # 0 = independent, 3 = strong partisan
    valid_ps = partisan_strength[~np.isnan(partisan_strength)]
    print(f"Partisan strength range: {valid_ps.min():.0f} to {valid_ps.max():.0f}")

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_with_regression_control(w1, w2, inertia, name, scale_min=0, scale_max=100):
    """
    Analyze belief change controlling for regression to mean.

    Key insight: Instead of raw |change|, use:
    1. Directional stability: Did they stay on same side?
    2. Proportional change: change / distance_from_boundary
    """
    midpoint = (scale_min + scale_max) / 2

    # Valid cases
    valid = ~np.isnan(w1) & ~np.isnan(w2) & ~np.isnan(inertia)
    valid &= (w1 >= scale_min) & (w1 <= scale_max)
    valid &= (w2 >= scale_min) & (w2 <= scale_max)

    w1, w2, inertia = w1[valid], w2[valid], inertia[valid]
    n = len(w1)

    if n < 100:
        print(f"  Skipping {name}: only {n} valid cases")
        return None

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name} (n={n})")
    print(f"{'='*60}")

    # ----- Test 1: Directional Stability -----
    # Did they stay on the same side of the midpoint?
    side_w1 = np.sign(w1 - midpoint)  # -1 = below, 0 = at, +1 = above
    side_w2 = np.sign(w2 - midpoint)
    same_side = (side_w1 == side_w2) | (side_w1 == 0) | (side_w2 == 0)

    # Correlation: high inertia → more likely same side
    r1, p1 = stats.pointbiserialr(same_side, inertia)

    print(f"\n[Test 1] Directional Stability")
    print(f"  Do high-inertia people stay on same side of midpoint?")
    print(f"  Correlation(inertia, same_side): r = {r1:.3f}, p = {p1:.4f}")
    print(f"  Prediction: r > 0")
    print(f"  Result: {'✓ SUPPORTED' if r1 > 0 and p1 < 0.05 else '✗ NOT SUPPORTED'}")

    # ----- Test 2: Regression-Controlled Change -----
    # For each person, what's their max possible change TOWARD the mean?
    # Then: proportional_change = actual_change / max_regression_change

    distance_from_mid = np.abs(w1 - midpoint)
    raw_change = np.abs(w2 - w1)

    # Avoid division by zero for people at midpoint
    distance_from_mid_safe = np.maximum(distance_from_mid, 1)
    proportional_change = raw_change / distance_from_mid_safe

    # Only analyze people NOT at midpoint (they have room to regress)
    not_at_mid = distance_from_mid > 5  # At least 5 points from center

    if not_at_mid.sum() > 100:
        r2, p2 = stats.pearsonr(inertia[not_at_mid], proportional_change[not_at_mid])

        print(f"\n[Test 2] Regression-Controlled Change (excluding middle)")
        print(f"  Does high inertia predict less proportional change?")
        print(f"  Correlation(inertia, prop_change): r = {r2:.3f}, p = {p2:.4f}")
        print(f"  Prediction: r < 0")
        print(f"  Result: {'✓ SUPPORTED' if r2 < 0 and p2 < 0.05 else '✗ NOT SUPPORTED'}")
    else:
        r2, p2 = np.nan, np.nan
        print(f"\n[Test 2] Skipped: not enough non-centrist cases")

    # ----- Test 3: Tertile Comparison -----
    low_thresh, high_thresh = np.percentile(inertia, [33, 67])
    low_inertia = inertia <= low_thresh
    high_inertia = inertia >= high_thresh

    stability_low = same_side[low_inertia].mean()
    stability_high = same_side[high_inertia].mean()

    # Chi-square test
    contingency = np.array([
        [same_side[low_inertia].sum(), (~same_side[low_inertia]).sum()],
        [same_side[high_inertia].sum(), (~same_side[high_inertia]).sum()]
    ])
    chi2, p3 = stats.chi2_contingency(contingency)[:2]

    print(f"\n[Test 3] Tertile Comparison")
    print(f"  Low inertia (independents): {stability_low:.1%} stayed same side")
    print(f"  High inertia (strong partisans): {stability_high:.1%} stayed same side")
    print(f"  Chi-square: χ² = {chi2:.1f}, p = {p3:.4f}")
    print(f"  Prediction: high > low")
    print(f"  Result: {'✓ SUPPORTED' if stability_high > stability_low and p3 < 0.05 else '✗ NOT SUPPORTED'}")

    return {
        'name': name,
        'n': n,
        'r_directional': r1,
        'p_directional': p1,
        'r_proportional': r2,
        'p_proportional': p2,
        'stability_low': stability_low,
        'stability_high': stability_high,
        'chi2_p': p3,
    }

# =============================================================================
# Run Analysis
# =============================================================================

print("\n" + "#"*60)
print("# BELIEF INERTIA ANALYSIS (v2)")
print("# Using partisan strength as inertia proxy")
print("#"*60)

all_results = []

for name, (w1_col, w2_col) in THERMO_VARS.items():
    if w1_col not in df.columns or w2_col not in df.columns:
        print(f"Skipping {name}: variables not found")
        continue

    w1 = convert_to_numeric(df[w1_col])
    w2 = convert_to_numeric(df[w2_col])

    print(f"\n{name}: w1 valid={np.sum(~np.isnan(w1))}, w2 valid={np.sum(~np.isnan(w2))}")

    if partisan_strength is None:
        print("  Skipping: no partisan strength data")
        continue

    result = analyze_with_regression_control(w1, w2, partisan_strength, name, 0, 100)
    if result:
        all_results.append(result)

# =============================================================================
# INDEPENDENT VOTER ANALYSIS
# =============================================================================

print("\n" + "#"*60)
print("# INDEPENDENT VOTER DEEP DIVE")
print("# VFE predicts: low inertia → more susceptible to influence")
print("#"*60)

# Independents = party_id == 4 (pure independents)
# Leaners = party_id in [3, 5] (independent but lean D/R)
pure_independent = (party_id == 4)
leaner = (party_id == 3) | (party_id == 5)
strong_partisan = (party_id <= 2) | (party_id >= 6)

print(f"\nGroups:")
print(f"  Pure independents (PID=4): {pure_independent.sum()}")
print(f"  Leaners (PID=3,5): {leaner.sum()}")
print(f"  Strong partisans (PID≤2 or ≥6): {strong_partisan.sum()}")

# For each thermometer, compare change across groups
for name, (w1_col, w2_col) in THERMO_VARS.items():
    if w1_col not in df.columns or w2_col not in df.columns:
        continue

    w1 = convert_to_numeric(df[w1_col])
    w2 = convert_to_numeric(df[w2_col])
    change = np.abs(w2 - w1)

    # Valid cases for each group
    valid = ~np.isnan(w1) & ~np.isnan(w2) & ~np.isnan(party_id)

    change_indep = change[valid & pure_independent]
    change_leaner = change[valid & leaner]
    change_partisan = change[valid & strong_partisan]

    if len(change_indep) < 30:
        continue

    print(f"\n{name}:")
    print(f"  Pure independents: mean Δ = {np.mean(change_indep):.1f} (n={len(change_indep)})")
    print(f"  Leaners:           mean Δ = {np.mean(change_leaner):.1f} (n={len(change_leaner)})")
    print(f"  Strong partisans:  mean Δ = {np.mean(change_partisan):.1f} (n={len(change_partisan)})")

    # Test: independents vs partisans
    t_stat, p_val = stats.ttest_ind(change_indep, change_partisan)
    print(f"  Independent vs Partisan: t={t_stat:.2f}, p={p_val:.4f}")

    if np.mean(change_indep) > np.mean(change_partisan) and p_val < 0.05:
        print(f"  ✓ VFE SUPPORTED: Independents change MORE than partisans")
    else:
        print(f"  Result: {'Indep > Partisan' if np.mean(change_indep) > np.mean(change_partisan) else 'Indep ≤ Partisan'}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)

if all_results:
    n_directional = sum(1 for r in all_results if r['r_directional'] > 0 and r['p_directional'] < 0.05)
    n_tertile = sum(1 for r in all_results if r['stability_high'] > r['stability_low'] and r['chi2_p'] < 0.05)

    print(f"Directional stability test: {n_directional}/{len(all_results)} supported")
    print(f"Tertile comparison test: {n_tertile}/{len(all_results)} supported")

    print("\nInterpretation:")
    if n_directional > 0 or n_tertile > 0:
        print("  Strong partisans are MORE stable in their attitudes.")
        print("  This supports the VFE belief inertia prediction:")
        print("  High precision (strong priors) → resistance to social influence")
    else:
        print("  No support found for belief inertia prediction.")
else:
    print("No results to summarize.")
