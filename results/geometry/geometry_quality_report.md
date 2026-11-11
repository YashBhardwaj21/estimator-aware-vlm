# Reliable RGB-D Object Geometry Audit Report

## Executive Summary

This report verifies the conversion of 2D object observations into metric 3D camera-frame and world-frame measurements across all eight benchmark sequences.
Every 3D measurement is derived strictly from real calibrated depth sensors and estimated ORB-SLAM3 poses, adhering strictly to role integrity with zero ground-truth leakage and zero identity fallbacks.

### Global Summary Metrics
- **Total 3D Observations Produced**: 1055
- **Total World-Transformed Observations**: 1055
- **Global World Transformation Rate**: 100.00%
- **Global Mean Valid Depth Fraction**: 72.97%
- **Global Mean Depth MAD (Dispersion)**: 0.0951 m
- **Global Mean Boundary Discontinuity**: 0.2281 m
- **Global Mean Depth Confidence**: 0.8432

## Per-Sequence Geometric Quality Breakdown

| Sequence ID | Input Obs | Valid 3D Obs | World Valid | World Trans Rate | Valid Depth % | Mean Depth MAD (m) | Mean Disc (m) | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_xyz` | 90 | 90 | 90 | 100.0% | 80.3% | 0.0599 | 0.1509 | 0.8864 |
| `tum_halfsphere` | 110 | 109 | 109 | 100.0% | 80.3% | 0.0617 | 0.2222 | 0.9008 |
| `tum_static` | 65 | 64 | 64 | 100.0% | 83.6% | 0.0654 | 0.0895 | 0.8974 |
| `openloris` | 30 | 26 | 26 | 100.0% | 78.0% | 0.1470 | 0.1069 | 0.7793 |
| `bonn_crowd` | 100 | 100 | 100 | 100.0% | 78.4% | 0.0690 | 0.1415 | 0.8767 |
| `bonn_box` | 55 | 55 | 55 | 100.0% | 83.5% | 0.0861 | 0.0548 | 0.8773 |
| `d455_01` | 189 | 188 | 188 | 100.0% | 67.0% | 0.1146 | 0.2556 | 0.8152 |
| `d455_02` | 424 | 423 | 423 | 100.0% | 67.6% | 0.1113 | 0.3052 | 0.8149 |

## Methodology and Geometric Formulation

### 1. Vectorized Inverse Pinhole Back-Projection
For each valid depth pixel $(u_i, v_i)$ within the instance mask contour:
$$\mathbf{P}_{cam, i} = Z_i \mathbf{K}^{-1} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{u_i - c_x}{f_x} Z_i \\ \frac{v_i - c_y}{f_y} Z_i \\ Z_i \end{bmatrix}$$

### 2. Robust Centroid and Covariance
The object centroid $\bar{\mathbf{P}}_{cam} = [\bar{X}, \bar{Y}, \bar{Z}]$ is computed using coordinate-wise medians for outlier immunity against specular depth dropouts.
The 3D point dispersion is captured by the empirical $3 \times 3$ covariance matrix $\mathbf{\Sigma}_{cam}$.

### 3. Spatial Bounding Dimensions
Dimensions $[dx, dy, dz]$ are estimated using 2nd-to-98th percentile filtering across the back-projected point cluster to eliminate flying pixel artifacts.

### 4. Boundary Depth Discontinuity
Evaluates the median depth step across the object boundary between an inner eroded contour and an outer dilated perimeter ring:
$$\Delta Z_{boundary} = |\text{median}(Z_{outer}) - \text{median}(Z_{inner})|$$

### 5. Camera-to-World Ego-Motion Compensation
Transformed into world coordinates using strictly estimated ORB-SLAM3 baseline poses:
$$\mathbf{P}_{world} = \mathbf{T}_{world \leftarrow cam} \bar{\mathbf{P}}_{cam} = \mathbf{R}_{wc} \bar{\mathbf{P}}_{cam} + \mathbf{t}_{wc}$$
Pose role integrity is strictly asserted (`TrajectoryRole.ESTIMATED`). Ground truth poses are prohibited from entering geometric observation features.

## Quality Verdict

- **Coordinate Frame Provenance**: Verified 100% compliant with `ESTIMATED_SLAM` provenance.
- **Depth Validity**: All observations exhibit valid metric depth within calibrated sensor envelopes.
- **Ego-Motion Compensation**: Identity fallback is strictly prohibited and disabled for all moving cameras.
- **Deliverable State**: Verified complete and ready for physical track building.
