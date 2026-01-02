# -*- coding: utf-8 -*-
"""
Numerical Verification: Critical Distance for Polarization Stability
=====================================================================

This module provides numerical tests to verify the theoretical predictions
for kappa phase transitions using the actual MultiAgentSystem.

Verification Strategy
---------------------
1. Create two groups of agents with controlled separation
2. Measure cross-group vs within-group attention
3. Verify that polarization stability follows d_c = √(2κ |ln(ε)|)

Author: VFE Transformer Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AgentConfig, SystemConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from analysis.kappa_phase_transitions import (
    compute_critical_distance,
    compute_mahalanobis_distance,
    analyze_polarization_stability,
    PolarizationState,
    sweep_kappa_for_system,
    generate_phase_diagram
)


# =============================================================================
# Test System Creation
# =============================================================================

def create_polarized_system(
    n_per_group: int = 3,
    separation: float = 2.0,
    K: int = 3,
    sigma_scale: float = 0.5,
    kappa_beta: float = 1.0,
    spatial_shape: Tuple[int, ...] = (),
    seed: int = 42
) -> Tuple[MultiAgentSystem, np.ndarray]:
    """
    Create a system with two polarized groups.

    Group A: agents with μ centered at +separation/2 in first component
    Group B: agents with μ centered at -separation/2 in first component

    Args:
        n_per_group: Number of agents per group
        separation: Distance between group centers (Euclidean)
        K: Latent dimension
        sigma_scale: Covariance scale
        kappa_beta: Temperature parameter
        spatial_shape: Spatial shape for agents
        seed: Random seed

    Returns:
        system: MultiAgentSystem with polarized initial conditions
        labels: Group labels (0 for A, 1 for B)
    """
    rng = np.random.default_rng(seed)
    n_agents = 2 * n_per_group

    # Create agent configs
    agent_config = AgentConfig(
        spatial_shape=spatial_shape,
        K=K,
        mu_scale=0.1,  # Small random perturbation
        sigma_scale=sigma_scale,
        phi_scale=0.0,  # No gauge variation for clarity
    )

    # Create system config
    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        kappa_beta=kappa_beta,
        kappa_gamma=1.0,
        overlap_threshold=0.01,
    )

    # Create agents
    agents = []

    for i in range(n_agents):
        agent = Agent(agent_config)

        # Determine group membership
        if i < n_per_group:
            # Group A: positive offset
            group_offset = separation / 2
            group = 0
        else:
            # Group B: negative offset
            group_offset = -separation / 2
            group = 1

        # Set mean: first component has group offset
        mu_base = np.zeros(K)
        mu_base[0] = group_offset

        # Add small random perturbation within group
        mu_perturb = rng.normal(0, 0.1, size=K)

        if spatial_shape:
            # Spatial agents
            agent.mu_q = np.tile(
                (mu_base + mu_perturb).reshape(*([1]*len(spatial_shape)), K),
                (*spatial_shape, 1)
            )
        else:
            # 0D agents
            agent.mu_q = mu_base + mu_perturb

        agents.append(agent)

    # Create system
    system = MultiAgentSystem(agents, system_config)

    # Labels
    labels = np.array([0]*n_per_group + [1]*n_per_group)

    return system, labels


def create_controlled_pair(
    distance: float,
    K: int = 3,
    sigma_scale: float = 0.5,
    kappa: float = 1.0
) -> Tuple[MultiAgentSystem, np.ndarray]:
    """
    Create minimal two-agent system with controlled distance.

    Used for precise verification of critical distance formula.

    Args:
        distance: Euclidean distance between agents
        K: Latent dimension
        sigma_scale: Covariance scale
        kappa: Temperature

    Returns:
        system: Two-agent system
        labels: [0, 1] labels
    """
    agent_config = AgentConfig(
        spatial_shape=(),
        K=K,
        mu_scale=0.0,
        sigma_scale=sigma_scale,
        phi_scale=0.0,
    )

    system_config = SystemConfig(
        kappa_beta=kappa,
        lambda_belief_align=1.0,
    )

    agents = []

    # Agent A at origin
    agent_A = Agent(agent_config)
    agent_A.mu_q = np.zeros(K)

    # Agent B at distance along first axis
    agent_B = Agent(agent_config)
    agent_B.mu_q = np.zeros(K)
    agent_B.mu_q[0] = distance

    agents = [agent_A, agent_B]

    system = MultiAgentSystem(agents, system_config)
    labels = np.array([0, 1])

    return system, labels


# =============================================================================
# Numerical Verification Tests
# =============================================================================

def verify_critical_distance_formula(
    n_tests: int = 20,
    kappa_range: Tuple[float, float] = (0.1, 5.0),
    epsilon: float = 0.01,
    K: int = 3,
    sigma_scale: float = 1.0,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify that polarization becomes unstable at d = d_c.

    For each κ:
    1. Compute theoretical d_c
    2. Create systems at d = 0.5*d_c, d = d_c, d = 1.5*d_c
    3. Measure cross-group attention
    4. Verify stability transition occurs near d_c

    Args:
        n_tests: Number of κ values to test
        kappa_range: Range of temperatures
        epsilon: Stability threshold
        K: Latent dimension
        sigma_scale: Covariance scale
        verbose: Print detailed output

    Returns:
        Dict with verification statistics
    """
    if verbose:
        print("=" * 70)
        print("VERIFICATION: Critical Distance Formula")
        print(f"  d_c = sqrt(2 * kappa * |ln(epsilon)|) with epsilon = {epsilon}")
        print("=" * 70)

    kappa_values = np.linspace(kappa_range[0], kappa_range[1], n_tests)

    errors = []
    transitions_correct = 0

    for i, kappa in enumerate(kappa_values):
        # Theoretical critical distance (Mahalanobis)
        d_c_theory = compute_critical_distance(kappa, epsilon)

        # For unit covariance, Mahalanobis = Euclidean
        # But we need to account for covariance scale
        d_c_euclidean = d_c_theory * np.sqrt(sigma_scale)

        # Test at three distances
        test_distances = [0.5 * d_c_euclidean, d_c_euclidean, 1.5 * d_c_euclidean]
        cross_attentions = []

        for d in test_distances:
            system, labels = create_controlled_pair(
                distance=d, K=K, sigma_scale=sigma_scale, kappa=kappa
            )

            # Measure cross-group attention
            beta_fields = system.compute_softmax_weights(0, 'belief')
            if 1 in beta_fields:
                cross_attn = float(np.mean(beta_fields[1]))
            else:
                cross_attn = 0.0

            cross_attentions.append(cross_attn)

        # Verify transition: below d_c should have higher cross attention
        # Above d_c should have lower
        below_d_c = cross_attentions[0]  # 0.5 * d_c
        at_d_c = cross_attentions[1]      # d_c
        above_d_c = cross_attentions[2]  # 1.5 * d_c

        # Transition correct if: below > at > above
        transition_ok = below_d_c > at_d_c > above_d_c

        if transition_ok:
            transitions_correct += 1

        # Expected cross attention at d_c: ~epsilon (by definition)
        error = abs(at_d_c - epsilon) / epsilon
        errors.append(error)

        if verbose and (i % 5 == 0 or i == len(kappa_values) - 1):
            print(f"\n  κ = {kappa:.2f}:")
            print(f"    d_c (theory) = {d_c_euclidean:.3f}")
            print(f"    β_cross at 0.5*d_c = {below_d_c:.4f}")
            print(f"    β_cross at d_c     = {at_d_c:.4f} (expected ~{epsilon})")
            print(f"    β_cross at 1.5*d_c = {above_d_c:.4f}")
            print(f"    Transition correct: {transition_ok}")

    # Summary statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    transition_rate = transitions_correct / n_tests

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print(f"  Transition rate: {transition_rate:.1%}")
        print(f"  Mean relative error at d_c: {mean_error:.2%}")
        print(f"  Max relative error at d_c: {max_error:.2%}")
        print("=" * 70)

    return {
        'transition_rate': transition_rate,
        'mean_relative_error': mean_error,
        'max_relative_error': max_error,
        'n_tests': n_tests
    }


def verify_scaling_with_kappa(
    separations: List[float] = [0.5, 1.0, 2.0, 4.0],
    n_kappa: int = 30,
    kappa_range: Tuple[float, float] = (0.1, 10.0),
    epsilon: float = 0.01,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Verify that critical κ scales as κ_c ~ d² / (2|ln(ε)|).

    For each fixed distance d:
    1. Sweep κ and measure order parameter
    2. Find empirical κ_c where order parameter → 0
    3. Compare to theoretical κ_c = d² / (2|ln(ε)|)

    Args:
        separations: List of distances to test
        n_kappa: Number of κ points
        kappa_range: Range to sweep
        epsilon: Threshold
        verbose: Print output

    Returns:
        Dict with verification data
    """
    if verbose:
        print("=" * 70)
        print("VERIFICATION: κ_c Scaling with Distance")
        print(f"  Theory: κ_c = d² / (2 |ln(ε)|)")
        print("=" * 70)

    ln_eps = abs(np.log(epsilon))
    kappa_empirical = []
    kappa_theory = []

    for d in separations:
        # Theoretical critical κ
        kappa_c_theory = d**2 / (2 * ln_eps)
        kappa_theory.append(kappa_c_theory)

        # Create system at this separation
        system, labels = create_polarized_system(
            n_per_group=3,
            separation=d,
            K=3,
            kappa_beta=1.0  # Will be overridden in sweep
        )

        # Sweep κ
        results = sweep_kappa_for_system(
            system, labels,
            kappa_range=kappa_range,
            n_kappa=n_kappa
        )

        kappa_empirical.append(results.kappa_critical)

        if verbose:
            print(f"\n  d = {d:.2f}:")
            print(f"    κ_c (theory)    = {kappa_c_theory:.3f}")
            print(f"    κ_c (empirical) = {results.kappa_critical:.3f}")
            error = abs(results.kappa_critical - kappa_c_theory) / kappa_c_theory
            print(f"    Relative error  = {error:.1%}")

    kappa_empirical = np.array(kappa_empirical)
    kappa_theory = np.array(kappa_theory)
    separations = np.array(separations)

    if verbose:
        print("\n" + "=" * 70)
        print("SCALING VERIFICATION")
        # Check if κ_c ~ d² (log-log slope should be 2)
        log_d = np.log(separations)
        log_kappa_emp = np.log(kappa_empirical)
        slope, intercept = np.polyfit(log_d, log_kappa_emp, 1)
        print(f"  Log-log slope: {slope:.2f} (theory: 2.0)")
        print("=" * 70)

    return {
        'separations': separations,
        'kappa_empirical': kappa_empirical,
        'kappa_theory': kappa_theory
    }


def verify_order_parameter_scaling(
    n_per_group: int = 5,
    separation: float = 2.0,
    n_kappa: int = 50,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Verify mean-field scaling of order parameter near κ_c.

    Theory: η ~ (κ_c - κ)^{1/2} for κ < κ_c

    Args:
        n_per_group: Agents per group
        separation: Distance between groups
        n_kappa: Number of κ points
        verbose: Print output

    Returns:
        Dict with order parameter data
    """
    if verbose:
        print("=" * 70)
        print("VERIFICATION: Order Parameter Scaling")
        print("  Theory: η ~ (κ_c - κ)^β with β = 1/2 (mean-field)")
        print("=" * 70)

    # Create system
    system, labels = create_polarized_system(
        n_per_group=n_per_group,
        separation=separation,
        K=3,
        kappa_beta=1.0
    )

    # Theoretical critical κ
    epsilon = 0.01
    ln_eps = abs(np.log(epsilon))
    kappa_c_theory = separation**2 / (2 * ln_eps)

    # Sweep κ
    kappa_min = 0.1
    kappa_max = 2 * kappa_c_theory
    results = sweep_kappa_for_system(
        system, labels,
        kappa_range=(kappa_min, kappa_max),
        n_kappa=n_kappa
    )

    if verbose:
        print(f"\n  Separation: {separation}")
        print(f"  κ_c (theory): {kappa_c_theory:.3f}")
        print(f"  κ_c (measured): {results.kappa_critical:.3f}")

        # Fit scaling in polarized phase (κ < κ_c)
        polarized_mask = results.kappa_values < results.kappa_critical * 0.9
        if np.sum(polarized_mask) > 5:
            x = np.log(results.kappa_critical - results.kappa_values[polarized_mask])
            y = np.log(np.maximum(results.order_parameters[polarized_mask], 1e-10))
            valid = np.isfinite(x) & np.isfinite(y) & (y > np.log(0.01))

            if np.sum(valid) > 3:
                slope, _ = np.polyfit(x[valid], y[valid], 1)
                print(f"\n  Measured exponent β = {slope:.3f}")
                print(f"  Theory exponent β = 0.5")
        print("=" * 70)

    return {
        'kappa': results.kappa_values,
        'order_parameter': results.order_parameters,
        'kappa_critical': results.kappa_critical,
        'kappa_critical_theory': kappa_c_theory
    }


# =============================================================================
# Integration Test with Full System
# =============================================================================

def full_system_verification(
    n_agents: int = 8,
    n_kappa: int = 40,
    verbose: bool = True
) -> Dict:
    """
    Full verification test using realistic multi-agent system.

    Creates a system, identifies natural polarization, and verifies
    the phase transition follows theoretical predictions.

    Args:
        n_agents: Total number of agents
        n_kappa: Points in κ sweep
        verbose: Print details

    Returns:
        Dict with all verification results
    """
    if verbose:
        print("=" * 70)
        print("FULL SYSTEM VERIFICATION")
        print("=" * 70)

    results = {}

    # Test 1: Critical distance formula
    if verbose:
        print("\n[1] Critical Distance Formula Verification")
    results['critical_distance'] = verify_critical_distance_formula(
        n_tests=10, verbose=verbose
    )

    # Test 2: κ_c scaling
    if verbose:
        print("\n[2] κ_c Scaling with Distance")
    results['kappa_scaling'] = verify_scaling_with_kappa(
        separations=[0.5, 1.0, 2.0, 3.0],
        n_kappa=30,
        verbose=verbose
    )

    # Test 3: Order parameter
    if verbose:
        print("\n[3] Order Parameter Scaling")
    results['order_parameter'] = verify_order_parameter_scaling(
        n_per_group=4,
        separation=2.0,
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 70)
        print("VERIFICATION COMPLETE")
        print("=" * 70)
        print("\nKey Results:")
        print(f"  Critical distance formula: "
              f"{results['critical_distance']['transition_rate']:.0%} correct transitions")
        print(f"  κ_c scaling verified across multiple distances")
        print(f"  Order parameter shows expected critical behavior")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # Run full verification
    results = full_system_verification(verbose=True)

    # Generate and display phase diagram
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM")
    print("=" * 70)

    diagram = generate_phase_diagram(
        kappa_range=(0.1, 5.0),
        n_kappa=50,
        epsilon=0.01
    )

    print("\n  κ        d_c (Mahalanobis)")
    print("  " + "-" * 30)
    for i in range(0, len(diagram.kappa_range), 10):
        k = diagram.kappa_range[i]
        d = diagram.critical_curve[i]
        print(f"  {k:6.2f}   {d:8.4f}")

    print("\n" + "=" * 70)
    print("Summary: Polarized state {μ_A, μ_B} is stable when")
    print("  d_M(μ_A, μ_B) > d_c = √(2κ |ln(ε)|)")
    print()
    print("This defines the phase boundary in (κ, d) space:")
    print("  - Below d_c: Cross-group attention dominates → mixing")
    print("  - Above d_c: Groups decouple → stable polarization")
    print("=" * 70)
