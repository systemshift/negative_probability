# Quasi-Probability Reinforcement Learning

A research implementation investigating the use of negative probabilities (quasi-probabilities) for counterfactual reasoning and bidirectional temporal navigation in reinforcement learning.

## Overview

This project explores whether **negative Q-values** can enable better learning through counterfactual reasoning - asking "what if I had acted differently?" without physically resetting the environment.

**Inspiration**: Feynman path integrals in quantum mechanics, where paths going "backward in time" contribute to probability amplitudes.

**Key Question**: Can RL agents benefit from reasoning about past decisions and hypothetical alternatives?

## What's Included

This repository contains:

1. **`qp_rl_v2/`** - Clean, research-grade implementation
   - Quasi-probability agents (with/without backward reasoning)
   - Classical Q-learning baseline
   - Test environments (trap mazes, corridors, four rooms)
   - Visualization suite
   - Experiment runner

2. **`qp_rl_project/`** - Original implementation (historical)

3. **Documentation**:
   - `THEORY.md` - Mathematical framework
   - `INSIGHT.md` - Why counterfactual inference is hard
   - `STATUS.md` - Project status and next steps
   - `FINAL_SUMMARY.md` - Complete analysis and recommendations

## Approaches Tested

### 1. Backward Jumps
- Reset environment to past states
- Try different actions with current knowledge
- **Result**: -331% vs. classical Q-learning (wastes samples)

### 2. Counterfactual Inference
- Don't reset, just update beliefs about past
- Infer "what if I had acted differently?"
- **Result**: -355% vs. classical Q-learning (fabricates experience)

### 3. Model-Based Extension (future work)
- Learn dynamics model P(s'|s,a)
- Use model for accurate counterfactuals
- Could actually work!

## Key Finding

**Counterfactual reasoning in model-free RL is challenging** because you don't know what would have happened on alternative paths. Both backward jumps and pure inference corrupt learning.

However, this could work with:
- Model-based RL (learned dynamics)
- Simulators (can reset to any state)
- Different applications (exploration bonuses, not action selection)

## Quick Start

```bash
# Install dependencies
pip3 install numpy matplotlib --break-system-packages

# Run main experiments (compares QP-RL vs. Classical)
python3 run_experiments.py --episodes 500 --runs 3

# Test counterfactual reasoning
python3 test_counterfactual.py

# Results saved to results_v2/
```

## Experimental Results

| Agent | Trap Maze Performance | Success Rate |
|-------|----------------------|--------------|
| Classical Q-Learning | 0.86 reward, 9 steps | 98% |
| QP-RL (backward jumps) | -1.35 reward, 42 steps | 55% |
| Counterfactual (inference) | -2.18 reward, 62 steps | 72% |

**Conclusion**: Quasi-probability approaches underperform classical Q-learning in model-free settings.

## Research Value

Despite negative empirical results, this work contributes:

1. **Theoretical framework** for quasi-probabilities in RL
2. **Rigorous experimental methodology** with proper baselines
3. **Understanding of why** counterfactual reasoning is hard in model-free RL
4. **Path forward**: Model-based extensions, different applications

## Publication Potential

✅ **Workshop paper**: Negative results with theoretical contribution
⚠️ **Conference paper**: Would need model-based extension or theoretical proofs
✅ **Learning experience**: Research-quality codebase and methodology

## Files

- `qp_rl_v2/` - Main implementation
- `THEORY.md` - Mathematical framework
- `INSIGHT.md` - Analysis of challenges
- `FINAL_SUMMARY.md` - Complete summary
- `run_experiments.py` - Experiment runner
- `test_counterfactual.py` - Test counterfactual approach

## Future Directions

1. **Model-based RL**: Learn dynamics, use for counterfactuals
2. **Different applications**: Exploration bonuses, not action selection
3. **Theoretical analysis**: Convergence proofs, sample complexity

## License

MIT License
