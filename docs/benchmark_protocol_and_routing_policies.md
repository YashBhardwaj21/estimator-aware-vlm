# Benchmark Protocol, Routing Policies, and Utility Calibration

## 1. Dual-Axis Evaluation Protocol

The evaluation protocol measures visual affordance classification accuracy, computational efficiency, and downstream state estimation fidelity across two distinct, rigorously controlled axes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DUAL-AXIS EVALUATION FRAMEWORK                        │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ Axis 1: Visual Movability            │ Axis 2: Estimator-Aware Routing      │
│ Classification (N = 42 Instances)    │ Policy Suite (N = 53 Entities)       │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Evaluates foundation model physical  │ Evaluates utility-driven selective   │
│ reasoning under controlled views:    │ query gating against baselines:      │
│ - Mode A: CROP_ONLY                  │ - B0: Blind RETAIN                   │
│ - Mode B: CROP_PLUS_CONTEXT          │ - B1: Blind INVALIDATE               │
│ - Models: Qwen2.5-VL-3B-Instruct     │ - B2: Geometry Only                  │
│                                      │ - B3: VLM All (100% queries)         │
│                                      │ - B4: Random Budget-Matched (K_B4=K) │
│                                      │ - B5: Confidence-Threshold Heuristic │
│                                      │ - B6: Estimator-Aware Routing        │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 2. Policy Suite Formal Specifications (B0–B6)

For a population of $N$ tracked physical candidate entities $\mathcal{T} = \{t_1, t_2, \dots, t_N\}$ observed by the perception front-end:

### B0: Blind RETAIN
* **Query Rule**: $q_i = 0$ for all $i$ (zero queries, 0% compute cost).
* **Assumed Mobility**: Assumes all objects are rigidly fixed background structures.
* **SLAM Action**: $\alpha_i = \text{RETAIN}$ for all $i \in \{1, \dots, N\}$.

### B1: Blind INVALIDATE
* **Query Rule**: $q_i = 0$ for all $i$ (zero queries).
* **Assumed Mobility**: Assumes all candidate objects are dynamic distractors.
* **SLAM Action**: $\alpha_i = \text{MASK\_FROM\_TRACKING}$ for all $i \in \{1, \dots, N\}$.

### B2: Geometry Only
* **Query Rule**: $q_i = 0$ for all $i$ (zero queries).
* **SLAM Action**: Evaluates metric 3D displacement threshold $d_{\text{min}} = 0.30\text{ m}$:
  $$\alpha_i = \begin{cases} \text{MASK\_FROM\_TRACKING} & \text{if } \text{motion}_i = \text{MOVING} \lor \|\Delta p_i\| \ge 0.30\text{ m} \\ \text{RETAIN} & \text{otherwise} \end{cases}$$

### B3: VLM All (Exhaustive Reference)
* **Query Rule**: $q_i = 1$ for all candidates with valid image crops ($100\%$ query rate).
* **Affordance Resolution**: Queried directly from live VLM: $\hat{a}_i = \text{VLM}(t_i) \in \{\text{IMMOVABLE}, \text{MOVABLE}, \text{MOBILE}\}$.
* **SLAM Action**:
  $$\alpha_i = \begin{cases} \text{MASK\_FROM\_TRACKING} & \text{if } \hat{a}_i = \text{MOBILE} \lor (\hat{a}_i = \text{MOVABLE} \land (\text{motion}_i = \text{MOVING} \lor \|\Delta p_i\| \ge 0.15\text{ m})) \\ \text{RETAIN} & \text{otherwise} \end{cases}$$

### B4: Random Routing (Budget-Matched to B6)
* **Strict Budget Equality**:
  $$K_{\text{budget}} = \sum_{i=1}^N q_{i}^{(\text{B6})}$$
  Let $\mathcal{E} \subseteq \mathcal{T}$ be the subset possessing valid image crops. B4 samples exactly $K_{\text{budget}}$ items uniformly at random without replacement:
  $$\mathcal{Q}_{\text{B4}} \sim \text{UniformChoice}\left(\mathcal{E}, K_{\text{budget}}, \text{seed}=42\right), \quad \sum_{i=1}^N q_i^{(\text{B4})} \equiv \sum_{i=1}^N q_i^{(\text{B6})}$$
* **Fairness Guarantee**: By enforcing $K_{\text{B4}} \equiv K_{\text{B6}}$, any empirical difference in downstream trajectory error between B6 and B4 is strictly attributable to *estimator-aware query selection*, eliminating compute budget as a confounding variable.

### B5: Confidence-Threshold Routing (Ambiguity Heuristic)
* **Query Rule**: Queries candidates exhibiting high spatial dispersion or borderline displacement:
  $$q_i = 1 \iff (0.12 \le \|\Delta p_i\| < 0.35\text{ m} \lor \sigma_{\text{disp}} > 0.08\text{ m}) \land \text{has\_crop}(t_i)$$

### B6: Estimator-Aware Selective Routing (Proposed)
* **Expected Downstream SLAM Benefit**:
  $$\Delta \hat{L}_{\text{slam}}(t_i) = \mathbf{w}^\top \left(\frac{\boldsymbol{\phi}(t_i) - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right) + b$$
* **Net Query Utility**:
  $$U(t_i) = \Delta \hat{L}_{\text{slam}}(t_i) - \lambda C_{\text{vlm}}$$
* **Query Decision**:
  $$q_i = \begin{cases} 1 & \text{if } U(t_i) \ge 0.0 \land \text{has\_crop}(t_i) \\ 0 & \text{otherwise} \end{cases}$$
* **Affordance & Action**:
  - If queried ($q_i = 1$): Resolve affordance $\hat{a}_i = \text{VLM}(t_i)$. If $\hat{a}_i = \text{MOBILE}$, mask; if $\hat{a}_i = \text{IMMOVABLE}$, retain; if $\hat{a}_i = \text{MOVABLE}$, mask if active displacement $\|\Delta p_i\| \ge 0.60\text{ m}$, else retain.
  - If sub-threshold ($q_i = 0$): Resolve cheaply using estimator prior at zero token cost.

---

## 3. Empirical Counterfactual Calibration of Query Utility

To eliminate arbitrary heuristic hyperparameter tuning, the utility model parameters $\mathbf{w}, b$ are calibrated on empirical single-track SLAM intervention counterfactuals:

1. **Counterfactual Intervention Dataset**:
   - For every tracked entity $t_i$ in training sequences, ORB-SLAM3 was executed under two counterfactual treatments:
     * Treatment 0: Entity $t_i$ retained in SLAM tracking.
     * Treatment 1: Entity $t_i$ suppressed via pixel-level masking.
   - The observed trajectory delta $\Delta L_{\text{slam}}^{(i)} = \text{ATE}_0^{(i)} - \text{ATE}_1^{(i)}$ serves as the ground-truth target.
2. **Ridge-Regularized Model Fitting**:
   - Model fitted via Leave-One-Out Cross-Validation (LOOCV).
   - Cross-validation performance:
     * **Mean Absolute Error (MAE)**: $0.0385\text{ m}$ ($3.85\text{ cm}$)
     * **Coefficient of Determination ($R^2$)**: **$0.8313$**
3. **Calibrated Feature Weights ($\mathbf{w}$)**:
   * Standardized features $\boldsymbol{\phi} \in \mathbb{R}^6$:
     $$\beta_{\text{disp}} = +3.418 \quad (\text{world displacement})$$
     $$\beta_{\text{cov}} = +2.105 \quad (\text{camera ego-motion})$$
     $$\beta_{\text{prior}} = +1.892 \quad (\text{displacement-to-ego ratio})$$
     $$\beta_{\text{obs}} = +0.842 \quad (\text{observation count})$$
     $$\beta_{\text{unc}} = -1.145 \quad (\text{displacement uncertainty penalty})$$
     $$\beta_{\text{life}} = +0.521 \quad (\text{lifespan duration})$$
   * Model Intercept: $b = 0.1428\text{ m}$.

---

## 4. Dual-Cost Latency & Token Accounting

To ensure transparent reporting across offline cache replay and live network inference regimes, the evaluation suite records four complementary cost dimensions:

1. **Observed Wall-Clock Latency (`observed_latency_ms`)**:
   Actual execution time during script execution. In cache replay mode, this measures dictionary retrieval ($\le 0.05\text{ ms}$ per instance).
2. **Hypothetical Wire Latency (`hypothetical_wire_latency_ms`)**:
   The measured network round-trip time required to transmit base64 RGB images, execute token generation, and receive decoded responses over HTTP:
   $$t_{\text{wire}}(P) = \sum_{t_i \in \mathcal{Q}_P} t_{\text{measured\_wire}}(t_i)$$
3. **Token Accumulation (`total_prompt_tokens`, `total_completion_tokens`)**:
   Empirical token counts consumed across all executed requests.
4. **Normalized Compute Cost ($C_{\text{norm}}$)**:
   $$C_{\text{norm}}(P) = \frac{|\mathcal{Q}_P|}{N} \times C_{\text{vlm}}$$
   where $C_{\text{vlm}} = 0.50$ is the standard normalized query cost penalty.

---

## 5. Statistical Formulations & Hypothesis Testing

### 5.1 Wilson Score Confidence Intervals
For binomial accuracy $\hat{p} = \frac{k}{n}$ at confidence level $1 - \alpha$ ($z = 1.95996$ for $95\%$ CI):

$$p_{\text{lower}}, p_{\text{upper}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Guarantees valid intervals $[0, 1]$ even for small samples ($N < 50$) or extreme proportions ($p \approx 1.0$).

### 5.2 McNemar's Exact Test
Evaluates paired classification differences between policies $A$ and $B$ on identical instances using a $2 \times 2$ contingency table:

| | Policy B Correct | Policy B Incorrect |
| :--- | :---: | :---: |
| **Policy A Correct** | $a$ | $b$ |
| **Policy A Incorrect** | $c$ | $d$ |

Under $H_0: P(b) = P(c)$, the exact two-tailed $p$-value is computed via the binomial distribution with parameter $p = 0.5$:

$$p = 2 \sum_{k=0}^{\min(b, c)} \binom{b+c}{k} 0.5^{b+c}$$

### 5.3 Holm-Bonferroni Correction
Controls Family-Wise Error Rate (FWER) under multiple policy comparisons. Given sorted $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$:

$$p_{(k)} \le \frac{\alpha}{m - k + 1}$$

