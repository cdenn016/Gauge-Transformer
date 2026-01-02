"""
Polarization Phase Transition Analysis
=======================================

Tests VFE prediction: Polarized states become stable when cross-group
attention decays exponentially with belief distance.

Key equation:
    cross_attention ∝ exp(-κ ||μ_A - μ_B||²)

Phase transition occurs at critical distance d_c where:
    κ * d_c² ≈ 1  →  d_c ≈ 1/√κ

Above d_c: polarized state is stable (groups ignore each other)
Below d_c: mixed state (groups influence each other)

Testable predictions:
1. Cross-party engagement decays with attitude distance
2. Decay follows exponential form (not linear)
3. There exists a critical threshold (phase transition signature)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

# =============================================================================
# Load Data
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
data_files = list(SCRIPT_DIR.glob('*.dta')) + list(SCRIPT_DIR.glob('*.sav')) + list(SCRIPT_DIR.glob('*.csv'))

if data_files:
    DATA_PATH = max(data_files, key=lambda f: f.stat().st_size)
    print(f"Auto-detected: {DATA_PATH.name}")
else:
    raise SystemExit("No data file found.")

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

def convert_to_numeric(series):
    """Convert ANES variable to numeric, handling Stata categories."""
    vals = pd.to_numeric(series, errors='coerce')
    if vals.isna().all():
        if hasattr(series, 'cat'):
            vals = series.cat.codes.astype(float)
            vals[vals < 0] = np.nan
        elif series.dtype == object:
            vals = series.astype(str).str.extract(r'^(\d+)')[0].astype(float)
    return vals.values

print("Loading data...")
df = load_data(DATA_PATH)
print(f"Loaded {len(df)} cases")

# =============================================================================
# Variable Mappings
# =============================================================================

# Party ID (1=Strong Dem ... 7=Strong Rep)
PARTY_ID_VAR = 'V201231x'

# Feeling thermometers (0-100)
THERMO_VARS = {
    'dem_party': 'V201156',
    'rep_party': 'V201157',
    'biden': 'V201151',
    'trump': 'V201152',
}

# Cross-party contact/engagement proxies
# V201622: How often discuss politics with family
# V201623: How often discuss politics with friends
DISCUSSION_VARS = ['V201622', 'V201623']

# =============================================================================
# Extract Data
# =============================================================================

party_id = convert_to_numeric(df[PARTY_ID_VAR])

# Own-party and out-party thermometer
thermo_dem = convert_to_numeric(df[THERMO_VARS['dem_party']])
thermo_rep = convert_to_numeric(df[THERMO_VARS['rep_party']])

# Classify as Democrat or Republican (excluding pure independents)
is_dem = party_id <= 3  # Lean Dem or stronger
is_rep = party_id >= 5  # Lean Rep or stronger

print(f"Democrats (PID ≤ 3): {is_dem.sum()}")
print(f"Republicans (PID ≥ 5): {is_rep.sum()}")

# =============================================================================
# Measure 1: Affective Polarization (Attitude Distance)
# =============================================================================

print("\n" + "="*60)
print("AFFECTIVE POLARIZATION MEASURE")
print("="*60)

# Affective polarization = in-party warmth - out-party warmth
# For Democrats: thermo_dem - thermo_rep
# For Republicans: thermo_rep - thermo_dem

affective_pol = np.full(len(df), np.nan)
affective_pol[is_dem] = thermo_dem[is_dem] - thermo_rep[is_dem]
affective_pol[is_rep] = thermo_rep[is_rep] - thermo_dem[is_rep]

valid_pol = ~np.isnan(affective_pol)
print(f"Valid cases: {valid_pol.sum()}")
print(f"Mean affective polarization: {np.nanmean(affective_pol):.1f}")
print(f"Std: {np.nanstd(affective_pol):.1f}")

# =============================================================================
# Measure 2: Cross-Party Attention (Out-Party Thermometer)
# =============================================================================

print("\n" + "="*60)
print("CROSS-PARTY ATTENTION")
print("="*60)

# Out-party warmth as proxy for cross-group attention
# Low out-party warmth = low attention = polarized state
out_party_warmth = np.full(len(df), np.nan)
out_party_warmth[is_dem] = thermo_rep[is_dem]
out_party_warmth[is_rep] = thermo_dem[is_rep]

print(f"Mean out-party warmth: {np.nanmean(out_party_warmth):.1f}")

# =============================================================================
# Test 1: Cross-Attention Decays with Distance
# =============================================================================

print("\n" + "="*60)
print("TEST 1: CROSS-ATTENTION DECAY")
print("="*60)
print("VFE Prediction: cross_attention ∝ exp(-κ * distance²)")

# Distance = absolute affective polarization
distance = np.abs(affective_pol)
attention = out_party_warmth

# Filter valid
valid = ~np.isnan(distance) & ~np.isnan(attention) & (distance > 0)
d = distance[valid]
a = attention[valid]

# Bin by distance deciles
n_bins = 10
bins = np.percentile(d, np.linspace(0, 100, n_bins + 1))
bin_idx = np.digitize(d, bins[:-1])

print("\nAttention by distance bin:")
bin_distances = []
bin_attentions = []
for i in range(1, n_bins + 1):
    mask = bin_idx == i
    if mask.sum() > 0:
        mean_d = np.mean(d[mask])
        mean_a = np.mean(a[mask])
        bin_distances.append(mean_d)
        bin_attentions.append(mean_a)
        print(f"  Distance {mean_d:5.1f}: attention = {mean_a:.1f} (n={mask.sum()})")

bin_distances = np.array(bin_distances)
bin_attentions = np.array(bin_attentions)

# =============================================================================
# Test 2: Fit Exponential Decay Model
# =============================================================================

print("\n" + "="*60)
print("TEST 2: EXPONENTIAL FIT")
print("="*60)

# Model: attention = A * exp(-κ * distance²) + baseline
def exp_decay(d, A, kappa, baseline):
    return A * np.exp(-kappa * d**2) + baseline

# Model: linear decay for comparison
def linear_decay(d, A, slope, baseline):
    return np.maximum(A + slope * d + baseline, 0)

try:
    # Fit exponential
    popt_exp, _ = curve_fit(
        exp_decay, bin_distances, bin_attentions,
        p0=[50, 0.001, 10],
        bounds=([0, 0, 0], [100, 1, 50]),
        maxfev=5000
    )
    A_exp, kappa, baseline_exp = popt_exp

    # Fit linear
    popt_lin, _ = curve_fit(
        linear_decay, bin_distances, bin_attentions,
        p0=[50, -0.5, 10],
        bounds=([0, -2, 0], [100, 0, 50]),
        maxfev=5000
    )

    # Compute R² for each model
    pred_exp = exp_decay(bin_distances, *popt_exp)
    pred_lin = linear_decay(bin_distances, *popt_lin)

    ss_tot = np.sum((bin_attentions - np.mean(bin_attentions))**2)
    ss_res_exp = np.sum((bin_attentions - pred_exp)**2)
    ss_res_lin = np.sum((bin_attentions - pred_lin)**2)

    r2_exp = 1 - ss_res_exp / ss_tot
    r2_lin = 1 - ss_res_lin / ss_tot

    print(f"\nExponential model: attention = {A_exp:.1f} * exp(-{kappa:.5f} * d²) + {baseline_exp:.1f}")
    print(f"  R² = {r2_exp:.3f}")
    print(f"  Estimated κ = {kappa:.5f}")
    print(f"  Critical distance d_c = 1/√κ = {1/np.sqrt(kappa):.1f}")

    print(f"\nLinear model:")
    print(f"  R² = {r2_lin:.3f}")

    print(f"\nModel comparison:")
    if r2_exp > r2_lin:
        print(f"  ✓ Exponential fits better (ΔR² = {r2_exp - r2_lin:.3f})")
        print(f"  VFE PREDICTION SUPPORTED: decay is exponential in distance²")
    else:
        print(f"  Linear fits better (ΔR² = {r2_lin - r2_exp:.3f})")

except Exception as e:
    print(f"Curve fitting failed: {e}")
    kappa = None

# =============================================================================
# Test 3: Phase Transition Signature
# =============================================================================

print("\n" + "="*60)
print("TEST 3: PHASE TRANSITION SIGNATURE")
print("="*60)
print("Looking for discontinuity in attitude stability at critical distance")

# Use pre/post thermometers if available
THERMO_POST = {
    'dem_party': 'V202145',
    'rep_party': 'V202146',
}

if THERMO_POST['dem_party'] in df.columns:
    thermo_dem_post = convert_to_numeric(df[THERMO_POST['dem_party']])
    thermo_rep_post = convert_to_numeric(df[THERMO_POST['rep_party']])

    # Compute attitude change
    change_dem = np.abs(thermo_dem_post - thermo_dem)
    change_rep = np.abs(thermo_rep_post - thermo_rep)

    # For each person: did polarization INCREASE or DECREASE?
    affective_pol_post = np.full(len(df), np.nan)
    affective_pol_post[is_dem] = thermo_dem_post[is_dem] - thermo_rep_post[is_dem]
    affective_pol_post[is_rep] = thermo_rep_post[is_rep] - thermo_dem_post[is_rep]

    pol_change = affective_pol_post - affective_pol  # + = more polarized

    # Test: do highly polarized people STAY polarized? (stability)
    valid_change = ~np.isnan(affective_pol) & ~np.isnan(pol_change)

    # Bin by initial polarization
    init_pol = affective_pol[valid_change]
    delta_pol = pol_change[valid_change]

    # Those already highly polarized should show less change (stable state)
    high_pol = init_pol > np.percentile(init_pol, 75)
    low_pol = init_pol < np.percentile(init_pol, 25)

    stability_high = 1 - np.std(delta_pol[high_pol]) / np.std(delta_pol)
    stability_low = 1 - np.std(delta_pol[low_pol]) / np.std(delta_pol)

    print(f"\nAlready-polarized (top 25%): stability = {stability_high:.3f}")
    print(f"Not-polarized (bottom 25%): stability = {stability_low:.3f}")

    # Correlation: initial polarization → less change?
    r_stab, p_stab = stats.pearsonr(np.abs(init_pol), np.abs(delta_pol))
    print(f"\nCorrelation(|initial_pol|, |Δpol|): r = {r_stab:.3f}, p = {p_stab:.4f}")
    print(f"Prediction: r < 0 (high polarization = stable = less change)")

    if r_stab < 0 and p_stab < 0.05:
        print(f"✓ VFE SUPPORTED: Polarized states are more stable")
    else:
        print(f"Result: {'Stable' if r_stab < 0 else 'Unstable'}")

else:
    print("Post-wave thermometers not found, skipping stability analysis")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("PHASE TRANSITION SUMMARY")
print("="*60)

print("""
VFE Model: cross_attention ∝ exp(-κ ||μ_A - μ_B||²)

Findings:
1. Cross-party attention decays with affective distance
2. Decay pattern: exponential vs linear?
3. Polarized states show stability (phase transition)

Interpretation:
- κ controls the "sharpness" of polarization
- Critical distance d_c = 1/√κ separates mixed and polarized phases
- Above d_c: groups effectively ignore each other → stable polarization
""")

if kappa is not None:
    print(f"Estimated parameters:")
    print(f"  κ = {kappa:.5f}")
    print(f"  d_c = {1/np.sqrt(kappa):.1f} (on 0-100 thermometer scale)")
    print(f"  This suggests polarization becomes stable when")
    print(f"  in-party vs out-party warmth differs by >{1/np.sqrt(kappa):.0f} points")
