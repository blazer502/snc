# Three-factor neuromodulated learning (reward-only)

A structure-friendly, brain-like training rule for the substrate — created
because the existing schemes (BPTT, e-prop/DFA) don't fit the model's nature:
they need a per-output **target vector** and (BPTT) a global backward pass, and
they treat the structure as a masked dense matrix. This rule instead learns from
a **single scalar reward** with the **local eligibility traces already in the
trainer**, and it is the *same* neuromodulatory form the two-timescale structural
plasticity uses. (Opt-in: `--reward-mode 1`; supervised e-prop is unchanged.)

## The rule

Per sample, after the LIF window produces output spike counts:

```
p        = softmax(counts)                 # output policy over classes
a        ~ Categorical(p)                  # sample an action
R        = 1[a == label]                   # scalar reward (right / wrong)
adv      = R - baseline                     # reward advantage (the neuromodulator)
delta_k  = adv * (p_k - 1[k == a])         # REINFORCE signal at the readout
w_s     -= lr * L_post(s) * E_s            # same eligibility update as e-prop
baseline = ema(baseline, R)                 # running reward reference
```

`E_s` is the unchanged eligibility (`Σ_t ψ_post · tr_pre`); the hidden learning
signal still spreads via the random feedback `B`. The **only** new ingredient is
the three-factor gating: **eligibility × dopamine-like reward advantage × the
sampled action**. The label is used *only to score the action* — there is no
target vector and no backward pass. This is dopamine-gated reinforcement of
eligibility traces, the brain's actual mechanism.

## Results (MNIST, static-snc, 256 hidden, budget 40k, 40 epochs, 2 seeds)

| structure | supervised e-prop | **reward-only (three-factor)** | gap |
|---|---|---|---|
| static-snc | 0.894 | **0.853** | +0.041 |
| random-sparse | 0.847 | 0.691 | +0.156 |
| **structure advantage** | **+0.047** | **+0.162** | |

Two findings:

1. **The substrate learns from scalar reward alone.** Reward-only reaches 0.853
   — just 4 pts below supervised — with no target vector and no backprop, a
   capability the supervised baselines don't have. The gap is the expected
   variance cost of REINFORCE vs a full gradient.

2. **Structure helps *more* under the reward rule.** The structure-aware prior
   (static-snc over matched random-sparse) is worth +4.7 pts supervised but
   **+16.2 pts under reward learning — 3.4× larger.** The interpretation: a
   noisy, global scalar reward is a much harder credit-assignment problem than a
   per-output gradient, and the structural prior gives a better-conditioned
   credit landscape — so it pays off most exactly where learning is hardest.
   Structure isn't just an efficiency knob; it's a *learnability* prior.

## Honest scope

MNIST is a static task and REINFORCE is higher-variance than the gradient, so
reward-only sits below supervised in absolute terms — as expected. The point is
the *capability* (label-free, backprop-free, on-graph learning) and the
*structure × learnability* interaction, both demonstrated cleanly. Natural next
steps: genuinely reward-shaped / delayed-reward tasks (where a target vector
doesn't exist), and unifying this neuromodulator with the structural-plasticity
clock so weights and wiring learn from one reward signal.
