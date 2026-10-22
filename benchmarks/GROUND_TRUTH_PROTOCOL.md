# Ground-Truth Movability & Dynamic Scene Graph Benchmark Protocol

## 1. Objective and Theoretical Framework

Evaluating object movability and motion estimation in dynamic 3D scene graphs requires establishing empirical ground truth independently of the perceptual models being evaluated. Without verified physical ground truth, model outputs remain uncalibrated predictions rather than verified measurements.

Following the methodological discipline of Rüdt et al. (*Towards Dynamic 3D Scene Graphs: Finding and Tracking Objects in the Wild*, arXiv:2404.09904), this benchmark:
1. Manually annotates physical object instances, semantic categories, and contextual movability across indoor robotics sequences.
2. Implements a controlled multi-session physical protocol to verify displacements against independent ground-truth measurements.
3. Provides an oracle-substitution framework to isolate upstream perception bottlenecks from downstream reasoning.

---

## 2. Disentangled Three-Question Ontology

To prevent the common conflation of physical affordance with momentary observation, the benchmark evaluates three fundamentally distinct questions for every object instance:

| Question | Conceptual Target | Benchmark Field | Permissible Ground-Truth Values |
| :--- | :--- | :--- | :--- |
| **Q1: Can this object be relocated?** | Contextual physical affordance / movability prior | `movability_gt` | `IMMOVABLE`, `MOVABLE`, `MOBILE`, `UNKNOWN` |
| **Q2: Did it move during this experiment?** | Observed geometric displacement during the interval | `current_motion_gt` | `STATIC`, `MOVING`, `UNCERTAIN` |
| **Q3: Did its movement make old SLAM observations unreliable?** | Epistemic landmark validity for geometric SLAM | `slam_invalidated` | `true`, `false` |

### A. Contextual Movability (`movability_gt`)
The three categories established by Rüdt et al. are maintained:
- **`IMMOVABLE`**: Permanently fixed, architectural, or structural elements that cannot ordinarily be relocated without tools, demolition, or structural building alterations.
  - *Examples*: Architectural load-bearing walls, structural columns, interior passage doors framed and hung in door jambs, ceiling fans securely anchored and wired to overhead concrete slabs.
- **`MOVABLE`**: Freestanding physical objects that can ordinarily be relocated, pushed, or carried by external human or robotic agents without structural disassembly.
  - *Examples*: Caster-wheeled office chairs, freestanding workstation desks, storage boxes, laptops, backpacks, computer monitors, keyboards, pedestal fans.
  - *Principle*: A stationary chair is still `MOVABLE`. Movability is an intrinsic physical affordance and contextual prior, not a function of whether motion occurred during the observation window.
- **`MOBILE`**: Contextual category involving intrinsic locomotion or an active mobile carrier.
  - *Examples*: Autonomous robotic platforms (e.g. the 4-motor RC car), human occupants/pedestrians, quadcopter drones, or objects actively carried by an agent.
  - *Principle*: `MOBILE` is a contextual capability category involving intrinsic propulsion or carrier dynamics; it is strictly not a synonym for "displaced in this frame".
- **`UNKNOWN` / Abstention**: Explicit category reserved for the model when perceptual evidence (e.g. heavy occlusion, truncation, extreme distance) is insufficient to establish affordance.

### B. Observed Motion (`current_motion_gt`)
- **`STATIC`**: The physical object maintained a constant world-relative 3D position within measurement uncertainty ($\Delta \mathbf{p} \le 3\sigma$) throughout the recorded interval.
- **`MOVING`**: The physical object underwent a verified spatial displacement ($\Delta \mathbf{p} > 3\sigma$) during the recording interval.
- **`UNCERTAIN`**: Sensor noise or tracking sparsity prevents distinguishing physical motion from localization jitter.

### C. SLAM Landmark Invalidation (`slam_invalidated`)
- **`true`**: The physical object changed position between observation epochs such that past geometric landmark observations are inconsistent with current reality and must be culled or downweighted in SLAM optimization.
- **`false`**: The object remained static; prior geometric observations remain valid metric constraints for map optimization and loop closure.

---

## 3. Controlled Data Collection Protocol

Ground truth is established using a multi-phase controlled experiment:

```
[Phase 1: Baseline Traversal]
   └── Sensor platform records initial stationary environment (Session 1)
   └── Model predictions logged prior to physical intervention
   └── Baseline 3D centroids computed for all physical objects

[Phase 2: Controlled Physical Intervention]
   └── Selective relocations executed with calibrated physical measurements
   └── Calibrated steel tape measure records metric displacement (sigma = 0.02 m)
   └── Stationary reference anchors maintained untouched

[Phase 3: Re-observation Traversal]
   └── Sensor platform re-traverses the environment (Session 2)
   └── Re-observation of displaced objects, stationary anchors, and novel entities

[Phase 4: Rigid Reference Registration & Verification]
   └── Distance-preserving RANSAC SE(2) registration solves coordinate alignment
   └── Alignment residual evaluated across rigid anchors (inlier residual sigma = 0.1198 m, all-anchor sigma = 0.3054 m)
   └── Measured displacement independently corroborated against tape-measure ground truth
```

### Measurement Uncertainty Quantification
- **Physical Tape Measure**: Calibrated steel tape precision: $\sigma_{\text{tape}} = \pm 0.020\text{ m}$.
- **SE(2) Anchor Residual**: RANSAC SE(2) inlier registration residual across verified static anchors: $\sigma_{\text{reg}} = \pm 0.1198\text{ m}$ (all-anchor RMS = $0.3054\text{ m}$).
- **Total Combined Uncertainty**:
  $$\sigma_{\text{total}} = \sqrt{\sigma_{\text{tape}}^2 + \sigma_{\text{reg}}^2} = \sqrt{0.020^2 + 0.1198^2} \approx \pm 0.1215\text{ m}$$
Displacements exceeding $3\sigma_{\text{total}} \approx 0.364\text{ m}$ are categorized as statistically significant motion events.

---

## 4. Track Resolution & Physical Instance Deduplication

Raw perception tracks from causal tracking front-ends do not correspond 1-to-1 with physical objects due to re-detections, occlusions, and multi-session mapping.

### Acceptance Gate Formulation
The benchmark establishes ground truth at the **physical instance level** rather than the raw tracking track level:
- If 78 confirmed entity tracks exist across sessions, they must be clustered into distinct physical instances via:
  1. Multi-session SE(2) spatial proximity ($\Delta d_{xy} < 0.40\text{ m}$).
  2. Bounding box 3D Intersection-over-Union (3D IoU).
  3. Semantic class compatibility and multi-view feature consistency.
- Evaluation metrics are reported per verified physical object instance, preventing repeated track counts from skewing validation results.

---

## 5. Dual-Annotator Verification & Inter-Annotator Agreement

To eliminate individual annotator bias:
1. Two independent annotators annotate every physical object instance:
   - **Annotator A**: Robotics & State Estimation Specialist.
   - **Annotator B**: Computer Vision & Scene Understanding Specialist.
2. Annotations are independently conducted for `movability_gt`, `current_motion_gt`, and `semantic_class`.
3. Inter-annotator agreement is quantified using:
   - **Percent Agreement**: $P_0 = \frac{\sum \text{agreements}}{N}$
   - **Cohen's Kappa**:
     $$\kappa = \frac{P_0 - P_e}{1 - P_e}$$
     where $P_e$ is the expected hypothetical probability of chance agreement.
4. Any annotation discrepancies are resolved through joint physical inspection of the environment.

---

## 6. Oracle-Substitution Study Design

Following Rüdt et al., system bottlenecks are identified by selectively substituting ground-truth intermediate outputs into the inference pipeline:

1. **Oracle Instance Bounding Boxes**: Evaluates whether movability reasoning errors stem from imperfect 3D segmentation or VLM classification.
2. **Oracle Semantic Classes**: Isolates semantic misclassification from affordance inference.
3. **Oracle Movability Prior**: Substitutes ground-truth `movability_gt` into SLAM landmark weighting to assess downstream mapping gain.
4. **Oracle Motion State**: Evaluates SLAM trajectory error when dynamic objects are perfectly masked out vs. estimated dynamically.
