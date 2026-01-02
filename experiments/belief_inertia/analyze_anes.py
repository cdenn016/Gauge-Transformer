#!/usr/bin/env python3
"""
Belief Inertia Analysis on ANES Panel Data
===========================================

Tests VFE prediction: high belief inertia → resistance to social influence.

Operationalization:
- Inertia proxy: attitude extremity × certainty (high precision = high inertia)
- Social pressure: partisan environment disagreement
- Outcome: attitude change between survey waves

Expected result: High-inertia individuals show less attitude change
even when exposed to disagreeing social environment.
"""

import argparse
import numpy as np
from pathlib import Path

# Optional heavy imports - graceful fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Data Loading
# =============================================================================

def load_panel_data(path: str) -> 'pd.DataFrame':
    """Load panel data from various formats."""
    if not HAS_PANDAS:
        raise ImportError("pandas required: pip install pandas")

    path = Path(path)

    # If directory, find data files inside
    if path.is_dir():
        candidates = []
        for ext in ['.dta', '.sav', '.csv']:
            candidates.extend(path.glob(f'*{ext}'))

        if not candidates:
            raise ValueError(
                f"No data files (.dta, .sav, .csv) found in {path}\n"
                f"Contents: {list(path.iterdir())[:10]}"
            )

        # Use the first/largest file found
        path = max(candidates, key=lambda p: p.stat().st_size)
        print(f"Found data file: {path.name}")

    if path.suffix == '.dta':
        return pd.read_stata(path)
    elif path.suffix == '.sav':
        return pd.read_spss(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .dta, .sav, or .csv")


# =============================================================================
# ANES Variable Mappings
# =============================================================================

# Common ANES attitude variables (7-point scales, waves indicated by suffix)
ANES_ATTITUDE_VARS = {
    # Issue positions (1-7 scale typically)
    'govt_services': ['V201246', 'V202231'],  # govt services vs spending
    'defense_spending': ['V201249', 'V202234'],
    'health_insurance': ['V201252', 'V202237'],
    'jobs_environment': ['V201255', 'V202240'],
    'aid_blacks': ['V201258', 'V202243'],

    # Feeling thermometers (0-100)
    'ft_dem_party': ['V201156', 'V202145'],
    'ft_rep_party': ['V201157', 'V202146'],
    'ft_biden': ['V201151', 'V202141'],
    'ft_trump': ['V201152', 'V202142'],
}

# Party ID (7-point)
ANES_PARTY_ID = ['V201231x', 'V202064']

# Political discussion network
ANES_DISCUSSION = ['V201624', 'V201625', 'V201626']  # discussants party


# =============================================================================
# Inertia Computation
# =============================================================================

def compute_attitude_extremity(values: np.ndarray, scale_min=1, scale_max=7) -> np.ndarray:
    """
    Compute attitude extremity: distance from scale midpoint.

    Extremity is a proxy for prior precision (Σ_p^{-1}).
    Those with extreme attitudes have tight priors → high inertia.
    """
    midpoint = (scale_min + scale_max) / 2
    extremity = np.abs(values - midpoint) / (scale_max - midpoint)
    return extremity


def compute_attitude_stability(wave1: np.ndarray, wave2: np.ndarray) -> np.ndarray:
    """
    Compute attitude stability between waves.

    Stability = 1 - |Δ| / max_possible_change
    High stability → high revealed inertia.
    """
    change = np.abs(wave2 - wave1)
    # Normalize by scale range (assuming 1-7)
    max_change = 6.0
    stability = 1.0 - (change / max_change)
    return stability


def compute_inertia_proxy(
    attitudes_w1: np.ndarray,
    certainty: np.ndarray = None,
    scale_min: float = 1,
    scale_max: float = 7,
) -> np.ndarray:
    """
    Compute belief inertia proxy from Wave 1 data.

    Inertia ∝ extremity × certainty

    This operationalizes M_i = Σ_p^{-1} + ... where high precision
    (extreme + certain) implies high inertia.
    """
    extremity = compute_attitude_extremity(attitudes_w1, scale_min, scale_max)

    if certainty is not None:
        # Certainty typically 1-5 scale, normalize to 0-1
        cert_norm = (certainty - 1) / 4
        inertia = extremity * cert_norm
    else:
        inertia = extremity

    return inertia


# =============================================================================
# Social Pressure Computation
# =============================================================================

def compute_environment_disagreement(
    own_party: np.ndarray,
    discussant_parties: np.ndarray,
) -> np.ndarray:
    """
    Compute disagreement with political discussion network.

    This operationalizes social pressure: how much does the
    individual's social environment push against their views?

    Args:
        own_party: Individual's party ID (1=Strong Dem ... 7=Strong Rep)
        discussant_parties: Party IDs of discussion partners (N, n_discussants)

    Returns:
        disagreement: Mean absolute party distance to discussants
    """
    # Mean discussant party (handling missing)
    mean_discussant = np.nanmean(discussant_parties, axis=1)

    # Disagreement = |own - mean_discussant| normalized
    disagreement = np.abs(own_party - mean_discussant) / 6.0

    return disagreement


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_belief_inertia(df: 'pd.DataFrame', config: dict = None) -> dict:
    """
    Main analysis: test whether inertia predicts resistance to attitude change.

    Test: attitude_change ~ inertia * social_pressure

    VFE Prediction:
    - Main effect of inertia: negative (high inertia → less change)
    - Interaction: negative (high inertia buffers against pressure)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required: pip install scipy")

    config = config or {}
    results = {}

    # Use first available attitude variable pair
    attitude_var = config.get('attitude_var', 'govt_services')

    if attitude_var in ANES_ATTITUDE_VARS:
        w1_var, w2_var = ANES_ATTITUDE_VARS[attitude_var]
    else:
        raise ValueError(f"Unknown attitude variable: {attitude_var}")

    # Check if variables exist
    if w1_var not in df.columns or w2_var not in df.columns:
        print(f"Variables {w1_var}, {w2_var} not found in data.")
        print(f"Available columns: {list(df.columns)[:20]}...")
        return results

    # Extract data
    w1 = df[w1_var].values.astype(float)
    w2 = df[w2_var].values.astype(float)

    # Compute measures
    inertia = compute_inertia_proxy(w1)
    stability = compute_attitude_stability(w1, w2)
    change = np.abs(w2 - w1)

    # Valid cases
    valid = ~(np.isnan(w1) | np.isnan(w2))
    n_valid = valid.sum()

    print(f"\n{'='*60}")
    print(f"BELIEF INERTIA ANALYSIS")
    print(f"{'='*60}")
    print(f"Attitude variable: {attitude_var}")
    print(f"Valid cases: {n_valid}")

    # ===== Test 1: Inertia predicts stability =====
    r, p = stats.pearsonr(inertia[valid], stability[valid])

    print(f"\n--- Test 1: Inertia → Stability ---")
    print(f"Correlation(inertia, stability): r = {r:.3f}, p = {p:.4f}")
    print(f"Prediction: r > 0 (high inertia = high stability)")
    print(f"Result: {'SUPPORTED' if r > 0 and p < 0.05 else 'NOT SUPPORTED'}")

    results['inertia_stability_r'] = r
    results['inertia_stability_p'] = p

    # ===== Test 2: Extremity predicts less change =====
    r2, p2 = stats.pearsonr(inertia[valid], change[valid])

    print(f"\n--- Test 2: Inertia → Less Change ---")
    print(f"Correlation(inertia, |change|): r = {r2:.3f}, p = {p2:.4f}")
    print(f"Prediction: r < 0 (high inertia = less change)")
    print(f"Result: {'SUPPORTED' if r2 < 0 and p2 < 0.05 else 'NOT SUPPORTED'}")

    results['inertia_change_r'] = r2
    results['inertia_change_p'] = p2

    # ===== Test 3: Tertile analysis =====
    inertia_tertiles = np.percentile(inertia[valid], [33, 67])
    low_inertia = inertia[valid] < inertia_tertiles[0]
    high_inertia = inertia[valid] > inertia_tertiles[1]

    change_low = change[valid][low_inertia]
    change_high = change[valid][high_inertia]

    t_stat, p_ttest = stats.ttest_ind(change_low, change_high)

    print(f"\n--- Test 3: Tertile Comparison ---")
    print(f"Low inertia group: mean change = {np.mean(change_low):.3f} (n={len(change_low)})")
    print(f"High inertia group: mean change = {np.mean(change_high):.3f} (n={len(change_high)})")
    print(f"t-test: t = {t_stat:.3f}, p = {p_ttest:.4f}")
    print(f"Prediction: low > high (high inertia resists change)")
    print(f"Result: {'SUPPORTED' if np.mean(change_low) > np.mean(change_high) and p_ttest < 0.05 else 'NOT SUPPORTED'}")

    results['mean_change_low_inertia'] = np.mean(change_low)
    results['mean_change_high_inertia'] = np.mean(change_high)
    results['tertile_ttest_p'] = p_ttest

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    n_supported = sum([
        r > 0 and p < 0.05,
        r2 < 0 and p2 < 0.05,
        np.mean(change_low) > np.mean(change_high) and p_ttest < 0.05
    ])

    print(f"Tests supported: {n_supported}/3")
    print(f"VFE belief inertia prediction: {'SUPPORTED' if n_supported >= 2 else 'NOT SUPPORTED'}")

    return results


# =============================================================================
# Synthetic Data for Testing Pipeline
# =============================================================================

def generate_synthetic_panel(n=1000, seed=42) -> 'pd.DataFrame':
    """
    Generate synthetic panel data with known inertia structure.

    For testing the analysis pipeline before using real data.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")

    np.random.seed(seed)

    # True inertia (latent)
    true_inertia = np.random.beta(2, 2, n)  # 0-1

    # Wave 1 attitudes (1-7 scale)
    # High inertia → more extreme
    extremity = true_inertia * 3  # 0-3 from midpoint
    direction = np.random.choice([-1, 1], n)
    w1 = 4 + direction * extremity + np.random.normal(0, 0.5, n)
    w1 = np.clip(w1, 1, 7)

    # Wave 2: change inversely proportional to inertia
    noise = np.random.normal(0, 1, n)
    change = noise * (1 - true_inertia) * 2  # High inertia → less change
    w2 = w1 + change
    w2 = np.clip(w2, 1, 7)

    # Party ID (correlated with attitudes)
    party = 4 + (w1 - 4) * 0.8 + np.random.normal(0, 0.5, n)
    party = np.clip(party, 1, 7)

    df = pd.DataFrame({
        'V201246': w1,  # Wave 1 attitude (using ANES var name)
        'V202231': w2,  # Wave 2 attitude
        'V201231x': party,
        'true_inertia': true_inertia,  # For validation
    })

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze belief inertia in panel data'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to panel data file (.dta, .sav, .csv)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data to test pipeline'
    )
    parser.add_argument(
        '--attitude', type=str, default='govt_services',
        help='Attitude variable to analyze'
    )

    args = parser.parse_args()

    if args.synthetic:
        print("Using synthetic data for pipeline testing...")
        df = generate_synthetic_panel()
        print(f"Generated {len(df)} synthetic respondents")
    elif args.data:
        print(f"Loading data from {args.data}...")
        df = load_panel_data(args.data)
        print(f"Loaded {len(df)} cases")
    else:
        print("No data specified. Use --synthetic or --data PATH")
        print("\nTo get real data:")
        print("1. Visit https://electionstudies.org/data-center/")
        print("2. Download ANES 2020 Time Series")
        print("3. Run: python analyze_anes.py --data path/to/anes2020.dta")
        return

    results = analyze_belief_inertia(df, {'attitude_var': args.attitude})


if __name__ == '__main__':
    main()
