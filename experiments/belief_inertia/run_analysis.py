"""
Belief Inertia Analysis
=======================

Run this script directly in Spyder/Jupyter.
Just set DATA_PATH below to your ANES data file.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# =============================================================================
# AUTO-DETECT DATA FILE (looks in same folder as this script)
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# Find any .dta, .sav, .csv file in this folder
data_files = list(SCRIPT_DIR.glob('*.dta')) + list(SCRIPT_DIR.glob('*.sav')) + list(SCRIPT_DIR.glob('*.csv'))

if data_files:
    DATA_PATH = max(data_files, key=lambda f: f.stat().st_size)
    print(f"Auto-detected: {DATA_PATH.name}")
else:
    print("No data file found!")
    print(f"Put your ANES .dta or .csv file in: {SCRIPT_DIR}")
    DATA_PATH = None

# =============================================================================
# Load Data
# =============================================================================

def load_data(path):
    """Load data file (.dta, .sav, or .csv)"""
    path = Path(path)

    # If directory, find largest data file
    if path.is_dir():
        files = list(path.glob('*.dta')) + list(path.glob('*.sav')) + list(path.glob('*.csv'))
        if not files:
            print(f"No data files found in {path}")
            print(f"Contents: {[f.name for f in path.iterdir()][:10]}")
            return None
        path = max(files, key=lambda f: f.stat().st_size)
        print(f"Using: {path.name}")

    ext = path.suffix.lower()
    if ext == '.dta':
        return pd.read_stata(path)
    elif ext == '.sav':
        return pd.read_spss(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    else:
        print(f"Unknown format: {ext}")
        return None

if DATA_PATH is None:
    raise SystemExit("No data file found. Download ANES from https://electionstudies.org/data-center/")

print("Loading data...")
df = load_data(DATA_PATH)

if df is not None:
    print(f"Loaded {len(df)} cases, {len(df.columns)} variables")
    print(f"Columns preview: {list(df.columns)[:20]}")

# =============================================================================
# Find Available Variables
# =============================================================================

def find_attitude_pairs(df):
    """Find attitude variables that appear in both waves."""

    # ANES 2020 variable patterns
    patterns = {
        'govt_services': ('V201246', 'V202231'),
        'defense': ('V201249', 'V202234'),
        'health_insurance': ('V201252', 'V202237'),
        'environment': ('V201255', 'V202240'),
        'aid_blacks': ('V201258', 'V202243'),
        'thermometer_dems': ('V201156', 'V202145'),
        'thermometer_reps': ('V201157', 'V202146'),
        'thermometer_biden': ('V201151', 'V202141'),
        'thermometer_trump': ('V201152', 'V202142'),
    }

    available = {}
    for name, (w1, w2) in patterns.items():
        if w1 in df.columns and w2 in df.columns:
            available[name] = (w1, w2)
            print(f"  Found: {name} ({w1} → {w2})")

    return available

print("\nSearching for attitude variables...")
attitude_vars = find_attitude_pairs(df) if df is not None else {}

# =============================================================================
# Compute Inertia and Stability
# =============================================================================

def analyze_variable(df, w1_col, w2_col, name, scale_range=(1, 7)):
    """Analyze belief inertia for one attitude variable."""

    scale_min, scale_max = scale_range
    midpoint = (scale_min + scale_max) / 2

    # Get waves
    w1 = pd.to_numeric(df[w1_col], errors='coerce').values
    w2 = pd.to_numeric(df[w2_col], errors='coerce').values

    # Filter valid (in range, not missing)
    valid = (
        ~np.isnan(w1) & ~np.isnan(w2) &
        (w1 >= scale_min) & (w1 <= scale_max) &
        (w2 >= scale_min) & (w2 <= scale_max)
    )
    w1, w2 = w1[valid], w2[valid]

    if len(w1) < 100:
        print(f"  Skipping {name}: only {len(w1)} valid cases")
        return None

    # Inertia proxy = attitude extremity (distance from midpoint)
    extremity = np.abs(w1 - midpoint) / (scale_max - midpoint)

    # Change and stability
    change = np.abs(w2 - w1)
    stability = 1 - change / (scale_max - scale_min)

    # Correlations
    r_stab, p_stab = stats.pearsonr(extremity, stability)
    r_change, p_change = stats.pearsonr(extremity, change)

    # Tertile comparison
    low_thresh, high_thresh = np.percentile(extremity, [33, 67])
    low_inertia = extremity < low_thresh
    high_inertia = extremity > high_thresh

    mean_change_low = np.mean(change[low_inertia])
    mean_change_high = np.mean(change[high_inertia])
    t_stat, p_ttest = stats.ttest_ind(change[low_inertia], change[high_inertia])

    results = {
        'name': name,
        'n': len(w1),
        'r_stability': r_stab,
        'p_stability': p_stab,
        'r_change': r_change,
        'p_change': p_change,
        'mean_change_low_inertia': mean_change_low,
        'mean_change_high_inertia': mean_change_high,
        'tertile_p': p_ttest,
    }

    return results

# =============================================================================
# Run Analysis
# =============================================================================

print("\n" + "="*60)
print("BELIEF INERTIA ANALYSIS")
print("="*60)

all_results = []

for name, (w1_col, w2_col) in attitude_vars.items():
    # Determine scale (thermometers are 0-100, issues are 1-7)
    if 'thermometer' in name:
        scale = (0, 100)
    else:
        scale = (1, 7)

    result = analyze_variable(df, w1_col, w2_col, name, scale)
    if result:
        all_results.append(result)

# Display results
print("\n" + "-"*60)
print("RESULTS BY VARIABLE")
print("-"*60)

for r in all_results:
    supported = (r['r_stability'] > 0 and r['p_stability'] < 0.05)

    print(f"\n{r['name']} (n={r['n']})")
    print(f"  Inertia-Stability correlation: r={r['r_stability']:.3f}, p={r['p_stability']:.4f}")
    print(f"  Inertia-Change correlation:    r={r['r_change']:.3f}, p={r['p_change']:.4f}")
    print(f"  Low inertia mean Δ:  {r['mean_change_low_inertia']:.3f}")
    print(f"  High inertia mean Δ: {r['mean_change_high_inertia']:.3f}")
    print(f"  VFE Prediction: {'✓ SUPPORTED' if supported else '✗ NOT SUPPORTED'}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

n_supported = sum(1 for r in all_results if r['r_stability'] > 0 and r['p_stability'] < 0.05)
print(f"Variables tested: {len(all_results)}")
print(f"VFE prediction supported: {n_supported}/{len(all_results)}")

# Create summary dataframe
if all_results:
    summary_df = pd.DataFrame(all_results)
    print("\n")
    print(summary_df[['name', 'n', 'r_stability', 'p_stability', 'mean_change_low_inertia', 'mean_change_high_inertia']].to_string(index=False))
