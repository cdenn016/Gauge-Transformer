# Belief Inertia Experiment

Testing the VFE prediction: individuals with high belief inertia M_i resist social influence.

## Theoretical Prediction

From the VFE framework:
```
M_i = Σ_p^{-1} + Σ_o^{-1} + Σ_j β_{ij} Ω_{ij} Σ_{q,j}^{-1} Ω_{ij}^T + Σ_j β_{ji} Σ_{q,i}^{-1}
```

High M → slow belief change → attitude stability over time.

## Operationalization

| Theory | Observable |
|--------|------------|
| Belief inertia M_i | Attitude stability across survey waves |
| Prior precision Σ_p^{-1} | Attitude extremity + certainty |
| Social pressure | Disagreement with partisan environment |
| Belief change | Δ attitude between waves |

## Data Sources

### Option 1: ANES Panel (Recommended)
- **URL**: https://electionstudies.org/data-center/
- **Files**: 2016-2020 Social Media Study or 2020-2022 Panel
- **Format**: Stata (.dta) or SPSS (.sav)
- **Registration**: Free academic account required

### Option 2: GSS Panels
- **URL**: https://gssdataexplorer.norc.org/
- **Panels**: 2006-2014 panel component

### Option 3: Pew American Trends Panel
- **URL**: https://www.pewresearch.org/american-trends-panel-datasets/

## Quick Start

```bash
# 1. Download ANES 2020 panel from electionstudies.org
# 2. Place in experiments/belief_inertia/data/

# 3. Run analysis
python analyze_anes.py --data data/anes_2020_panel.dta
```
