# Track Annotation Audit

Source run: `C:\Users\Yash Bhardwaj\Desktop\vlm\results\perception_harvesting\run_20261101_234859`
Annotator: rule-based (scripts/auto_annotate_tracks.py)
Tracks: 177

## Distribution Summary

### Observed Motion

| Label | Count |
| --- | --- |
| MOVING | 102 |
| STATIONARY | 57 |
| UNCERTAIN | 18 |

### Physical Mobility

| Label | Count |
| --- | --- |
| MOVABLE | 174 |
| STRUCTURALLY_IMMOVABLE | 3 |

### Track Identity Validity

| Label | Count |
| --- | --- |
| VALID | 124 |
| QUESTIONABLE | 53 |

### Annotation Confidence

| Label | Count |
| --- | --- |
| MEDIUM | 71 |
| HIGH | 69 |
| LOW | 37 |

### Evidence Sufficiency

| Label | Count |
| --- | --- |
| SUFFICIENT | 109 |
| INSUFFICIENT | 68 |

## Per-Track Rationale

### tum_xyz / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3549 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.355m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.3549 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.81 mean_area=14113 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9616 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.962m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.9616 → UNCERTAIN (LOW) | [visibility] mean_conf=0.76 mean_area=5150 → CLEAR | [track_identity] QUESTIONABLE: static class 'monitor' shows world_disp=0.962m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### tum_xyz / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8279 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.8279 → MOVING (HIGH) | [visibility] mean_conf=0.69 mean_area=22409 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 3.0845 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=3.0845 → MOVING (HIGH) | [visibility] mean_conf=0.89 mean_area=72577 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.6928 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.6928 → MOVING (HIGH) | [visibility] mean_conf=0.89 mean_area=66822 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.1446 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.1446 → MOVING (HIGH) | [visibility] mean_conf=0.79 mean_area=72211 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7894 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7894 → MOVING (HIGH) | [visibility] mean_conf=0.52 mean_area=20255 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.5306 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.5306 → MOVING (HIGH) | [visibility] mean_conf=0.80 mean_area=55401 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_05

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.458 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.458 → MOVING (HIGH) | [visibility] mean_conf=0.63 mean_area=74834 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / keyboard_01

| Field | Value |
| --- | --- |
| object_class | keyboard |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5697 |

**Rationale:** [physical_mobility] class='keyboard' → MOVABLE (rule-based by class) | [observed_motion] disp=0.570m unexpectedly large for 'keyboard' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.5697 → UNCERTAIN (LOW) | [visibility] mean_conf=0.70 mean_area=2525 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### tum_xyz / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2469 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2469 → MOVING (MEDIUM) | [visibility] mean_conf=0.62 mean_area=31165 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / person_06

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.728 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.728 → MOVING (HIGH) | [visibility] mean_conf=0.55 mean_area=1998 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_xyz / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.526 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.526 → MOVING (HIGH) | [visibility] mean_conf=0.52 mean_area=10549 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### tum_xyz / chair_05

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1455 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1455 → STATIONARY (HIGH) | [visibility] mean_conf=0.49 mean_area=14902 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### tum_xyz / person_07

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.131 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.131 → MOVING (MEDIUM) | [visibility] mean_conf=0.79 mean_area=20910 → CLEAR | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### tum_halfsphere / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.7133 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.7133 → MOVING (HIGH) | [visibility] mean_conf=0.77 mean_area=33417 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4815 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.481m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.4815 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.76 mean_area=7993 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.0624 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=1.062m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=1.0624 → UNCERTAIN (LOW) | [visibility] mean_conf=0.80 mean_area=11959 → CLEAR | [track_identity] QUESTIONABLE: static class 'monitor' shows world_disp=1.062m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### tum_halfsphere / keyboard_01

| Field | Value |
| --- | --- |
| object_class | keyboard |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2324 |

**Rationale:** [physical_mobility] class='keyboard' → MOVABLE (rule-based by class) | [observed_motion] disp=0.232m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2324 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.82 mean_area=2941 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9526 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9526 → MOVING (HIGH) | [visibility] mean_conf=0.70 mean_area=30468 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / monitor_03

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5479 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.548m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.5479 → UNCERTAIN (LOW) | [visibility] mean_conf=0.56 mean_area=19858 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### tum_halfsphere / keyboard_02

| Field | Value |
| --- | --- |
| object_class | keyboard |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4733 |

**Rationale:** [physical_mobility] class='keyboard' → MOVABLE (rule-based by class) | [observed_motion] disp=0.473m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.4733 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.57 mean_area=3338 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 3.2174 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=3.2174 → MOVING (HIGH) | [visibility] mean_conf=0.64 mean_area=33102 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.7324 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.7324 → MOVING (HIGH) | [visibility] mean_conf=0.69 mean_area=27769 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 3.3195 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=3.3195 → MOVING (HIGH) | [visibility] mean_conf=0.88 mean_area=51180 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / person_05

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.6247 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.6247 → MOVING (HIGH) | [visibility] mean_conf=0.91 mean_area=46115 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4361 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4361 → MOVING (MEDIUM) | [visibility] mean_conf=0.60 mean_area=16768 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / person_06

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.8999 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.8999 → MOVING (HIGH) | [visibility] mean_conf=0.61 mean_area=10801 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3647 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.3647 → MOVING (MEDIUM) | [visibility] mean_conf=0.51 mean_area=33931 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / laptop_01

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1521 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1521 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.64 mean_area=11140 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_halfsphere / mouse_01

| Field | Value |
| --- | --- |
| object_class | mouse |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7371 |

**Rationale:** [physical_mobility] class='mouse' → MOVABLE (rule-based by class) | [observed_motion] disp=0.737m unexpectedly large for 'mouse' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.7371 → UNCERTAIN (LOW) | [visibility] mean_conf=0.53 mean_area=1211 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### tum_halfsphere / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.06 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.06 → STATIONARY (HIGH) | [visibility] mean_conf=0.57 mean_area=9562 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### tum_halfsphere / monitor_04

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0339 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0339 → STATIONARY (HIGH) | [visibility] mean_conf=0.68 mean_area=5906 → CLEAR | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### tum_halfsphere / laptop_02

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0355 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0355 → STATIONARY (HIGH) | [visibility] mean_conf=0.59 mean_area=21951 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### tum_static / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3534 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.353m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.3534 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.87 mean_area=5366 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4417 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.442m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.4417 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.76 mean_area=8456 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5721 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.5721 → MOVING (HIGH) | [visibility] mean_conf=0.77 mean_area=23387 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.724 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.724 → MOVING (HIGH) | [visibility] mean_conf=0.58 mean_area=36374 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.3332 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.3332 → MOVING (HIGH) | [visibility] mean_conf=0.88 mean_area=77931 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.638 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.638 → MOVING (HIGH) | [visibility] mean_conf=0.54 mean_area=26206 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7808 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7808 → MOVING (HIGH) | [visibility] mean_conf=0.81 mean_area=49049 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.6577 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.6577 → MOVING (HIGH) | [visibility] mean_conf=0.83 mean_area=40074 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2307 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2307 → MOVING (MEDIUM) | [visibility] mean_conf=0.56 mean_area=34153 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### tum_static / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.1054 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.1054 → MOVING (HIGH) | [visibility] mean_conf=0.52 mean_area=7719 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### openloris / plant_01

| Field | Value |
| --- | --- |
| object_class | plant |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.8469 |

**Rationale:** [physical_mobility] class='plant' → MOVABLE (rule-based by class) | [observed_motion] disp=1.847m unexpectedly large for 'plant' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=1.8469 → UNCERTAIN (LOW) | [visibility] mean_conf=0.68 mean_area=6190 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### openloris / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.5954 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.5954 → MOVING (HIGH) | [visibility] mean_conf=0.62 mean_area=3920 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### openloris / plant_02

| Field | Value |
| --- | --- |
| object_class | plant |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6706 |

**Rationale:** [physical_mobility] class='plant' → MOVABLE (rule-based by class) | [observed_motion] disp=0.671m unexpectedly large for 'plant' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.6706 → UNCERTAIN (LOW) | [visibility] mean_conf=0.51 mean_area=5663 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### bonn_crowd / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 3.8698 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=3.8698 → MOVING (HIGH) | [visibility] mean_conf=0.89 mean_area=41809 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9111 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9111 → MOVING (HIGH) | [visibility] mean_conf=0.80 mean_area=25346 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7156 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7156 → MOVING (HIGH) | [visibility] mean_conf=0.83 mean_area=21866 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.6888 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.6888 → MOVING (HIGH) | [visibility] mean_conf=0.82 mean_area=51382 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.1791 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.1791 → MOVING (HIGH) | [visibility] mean_conf=0.91 mean_area=32482 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8104 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.8104 → MOVING (HIGH) | [visibility] mean_conf=0.88 mean_area=27321 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.4566 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.4566 → MOVING (HIGH) | [visibility] mean_conf=0.78 mean_area=78876 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.2165 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.2165 → MOVING (HIGH) | [visibility] mean_conf=0.78 mean_area=72939 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1612 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1612 → STATIONARY (HIGH) | [visibility] mean_conf=0.64 mean_area=4122 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_05

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.3336 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.3336 → MOVING (HIGH) | [visibility] mean_conf=0.77 mean_area=39921 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5137 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.514m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.5137 → UNCERTAIN (LOW) | [visibility] mean_conf=0.65 mean_area=4654 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### bonn_crowd / person_06

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.7942 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.7942 → MOVING (HIGH) | [visibility] mean_conf=0.81 mean_area=70432 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_07

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.4943 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.4943 → MOVING (HIGH) | [visibility] mean_conf=0.69 mean_area=57581 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_08

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.6542 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.6542 → MOVING (HIGH) | [visibility] mean_conf=0.71 mean_area=30462 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / desk_01

| Field | Value |
| --- | --- |
| object_class | desk |
| physical_mobility | STRUCTURALLY_IMMOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4147 |

**Rationale:** [physical_mobility] class='desk' → STRUCTURALLY_IMMOVABLE (rule-based by class) | [observed_motion] desk shows disp=0.415m — ego-comp residual or poor detection; classified STATIONARY | [observed_motion] pipe_motion=MOVING world_disp=0.4147 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.46 mean_area=25300 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / monitor_03

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0809 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0809 → STATIONARY (HIGH) | [visibility] mean_conf=0.50 mean_area=4478 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_09

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8751 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.8751 → MOVING (HIGH) | [visibility] mean_conf=0.86 mean_area=81628 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / monitor_04

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2593 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.259m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2593 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.54 mean_area=3747 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_crowd / person_10

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4796 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4796 → MOVING (MEDIUM) | [visibility] mean_conf=0.78 mean_area=95646 → CLEAR | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### bonn_crowd / laptop_01

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.1067 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.1067 → STATIONARY (HIGH) | [visibility] mean_conf=0.52 mean_area=8100 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### bonn_box / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.1925 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.1925 → MOVING (HIGH) | [visibility] mean_conf=0.74 mean_area=56344 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_box / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5468 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.5468 → MOVING (HIGH) | [visibility] mean_conf=0.88 mean_area=37092 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_box / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9682 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.968m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.9682 → UNCERTAIN (LOW) | [visibility] mean_conf=0.53 mean_area=4568 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: static class 'monitor' shows world_disp=0.968m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### bonn_box / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1961 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1961 → STATIONARY (HIGH) | [visibility] mean_conf=0.94 mean_area=41550 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_box / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9847 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9847 → MOVING (HIGH) | [visibility] mean_conf=0.92 mean_area=38978 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### bonn_box / laptop_01

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0719 |

**Rationale:** [class] YOLO label 'laptop' in bonn_box sequence — may be cardboard box misclassified. | [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0719 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.68 mean_area=10559 → CLEAR | [track_identity] QUESTIONABLE: laptop in bonn_box with near-zero disp — may be misclassified box fragment | [evidence] → SUFFICIENT

### d455_01 / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6644 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.6644 → MOVING (HIGH) | [visibility] mean_conf=0.46 mean_area=2033 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_01 / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.7556 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.7556 → MOVING (HIGH) | [visibility] mean_conf=0.59 mean_area=15507 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.6273 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.6273 → MOVING (HIGH) | [visibility] mean_conf=0.62 mean_area=12727 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4749 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4749 → MOVING (MEDIUM) | [visibility] mean_conf=0.75 mean_area=16582 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_05

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9087 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9087 → MOVING (HIGH) | [visibility] mean_conf=0.49 mean_area=4437 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_06

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6182 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.6182 → MOVING (HIGH) | [visibility] mean_conf=0.53 mean_area=4480 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_07

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4103 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4103 → MOVING (MEDIUM) | [visibility] mean_conf=0.81 mean_area=30270 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_08

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.9389 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.9389 → MOVING (HIGH) | [visibility] mean_conf=0.48 mean_area=37779 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_09

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.7601 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.7601 → MOVING (HIGH) | [visibility] mean_conf=0.52 mean_area=32510 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2444 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.244m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2444 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.64 mean_area=2324 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6474 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.6474 → MOVING (HIGH) | [visibility] mean_conf=0.47 mean_area=1489 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_01 / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3269 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.327m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.3269 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.68 mean_area=9892 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 4.4697 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=4.4697 → MOVING (HIGH) | [visibility] mean_conf=0.85 mean_area=22211 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_10

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.2238 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.2238 → MOVING (HIGH) | [visibility] mean_conf=0.49 mean_area=67230 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_11

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5368 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.5368 → MOVING (HIGH) | [visibility] mean_conf=0.46 mean_area=29218 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_12

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4721 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4721 → MOVING (MEDIUM) | [visibility] mean_conf=0.55 mean_area=15867 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_01 / chair_13

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1209 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1209 → STATIONARY (HIGH) | [visibility] mean_conf=0.56 mean_area=11717 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_14

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5184 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.5184 → MOVING (HIGH) | [visibility] mean_conf=0.76 mean_area=20695 → CLEAR | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_15

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9484 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9484 → MOVING (HIGH) | [visibility] mean_conf=0.48 mean_area=7787 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / monitor_03

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6729 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.673m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.6729 → UNCERTAIN (LOW) | [visibility] mean_conf=0.65 mean_area=2182 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3086 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.3086 → MOVING (MEDIUM) | [visibility] mean_conf=0.47 mean_area=1128 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2186 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2186 → MOVING (MEDIUM) | [visibility] mean_conf=0.48 mean_area=1036 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_16

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2426 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2426 → MOVING (MEDIUM) | [visibility] mean_conf=0.64 mean_area=74306 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (8 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_17

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3062 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.3062 → MOVING (MEDIUM) | [visibility] mean_conf=0.44 mean_area=6440 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_18

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1604 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1604 → STATIONARY (HIGH) | [visibility] mean_conf=0.49 mean_area=15284 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_19

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7588 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7588 → MOVING (HIGH) | [visibility] mean_conf=0.50 mean_area=40400 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_20

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.0809 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.0809 → MOVING (HIGH) | [visibility] mean_conf=0.81 mean_area=86242 → CLEAR | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_21

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3682 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.3682 → MOVING (MEDIUM) | [visibility] mean_conf=0.51 mean_area=24120 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / monitor_04

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0543 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0543 → STATIONARY (HIGH) | [visibility] mean_conf=0.49 mean_area=12393 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / laptop_01

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0206 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0206 → STATIONARY (HIGH) | [visibility] mean_conf=0.61 mean_area=4485 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_22

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9364 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9364 → MOVING (HIGH) | [visibility] mean_conf=0.62 mean_area=34887 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_23

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8232 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.8232 → MOVING (HIGH) | [visibility] mean_conf=0.47 mean_area=4403 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_24

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.3942 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.3942 → MOVING (HIGH) | [visibility] mean_conf=0.46 mean_area=39497 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / chair_25

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2314 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2314 → MOVING (MEDIUM) | [visibility] mean_conf=0.72 mean_area=156942 → CLEAR | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / monitor_05

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0532 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0532 → STATIONARY (HIGH) | [visibility] mean_conf=0.50 mean_area=1197 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_01 / monitor_06

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8855 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.885m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.8855 → UNCERTAIN (LOW) | [visibility] mean_conf=0.54 mean_area=2745 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10); static class 'monitor' shows world_disp=0.885m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### d455_02 / chair_01

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.3249 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.3249 → MOVING (HIGH) | [visibility] mean_conf=0.94 mean_area=63126 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_02

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.2118 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.2118 → MOVING (HIGH) | [visibility] mean_conf=0.52 mean_area=2451 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_01

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.1288 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=1.129m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=1.1288 → UNCERTAIN (LOW) | [visibility] mean_conf=0.49 mean_area=2658 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: static class 'monitor' shows world_disp=1.129m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### d455_02 / chair_03

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.0286 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.0286 → MOVING (HIGH) | [visibility] mean_conf=0.76 mean_area=24217 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_04

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.4822 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.4822 → MOVING (HIGH) | [visibility] mean_conf=0.58 mean_area=10037 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_05

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.0061 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.0061 → MOVING (HIGH) | [visibility] mean_conf=0.60 mean_area=7596 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_02

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3919 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.392m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.3919 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.81 mean_area=4688 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_03

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1999 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1999 → STATIONARY (HIGH) | [visibility] mean_conf=0.54 mean_area=1640 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_06

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.0986 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.0986 → MOVING (HIGH) | [visibility] mean_conf=0.66 mean_area=16244 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_04

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3859 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.386m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.3859 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.60 mean_area=1460 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_07

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.278 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.278 → MOVING (MEDIUM) | [visibility] mean_conf=0.70 mean_area=4955 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_01

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4238 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4238 → MOVING (MEDIUM) | [visibility] mean_conf=0.71 mean_area=2904 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_05

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4877 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.488m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.4877 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.65 mean_area=1728 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_08

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.866 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.866 → MOVING (HIGH) | [visibility] mean_conf=0.61 mean_area=5474 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_09

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.2708 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.2708 → MOVING (HIGH) | [visibility] mean_conf=0.48 mean_area=2136 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / chair_10

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4635 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4635 → MOVING (MEDIUM) | [visibility] mean_conf=0.45 mean_area=1752 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / monitor_06

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3369 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.337m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.3369 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.86 mean_area=20892 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_07

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4552 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.455m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.4552 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.70 mean_area=1268 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / laptop_01

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5715 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] laptop disp=0.572m unexpectedly large — likely ego-comp residual or misclassification | [observed_motion] pipe_motion=MOVING world_disp=0.5715 → UNCERTAIN (LOW) | [visibility] mean_conf=0.56 mean_area=1876 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / monitor_08

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2973 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.297m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2973 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.77 mean_area=1355 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_09

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2487 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.249m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2487 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.62 mean_area=1315 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_11

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.1967 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.1967 → MOVING (HIGH) | [visibility] mean_conf=0.49 mean_area=8879 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_10

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2518 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.252m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.2518 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.58 mean_area=4626 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_11

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3836 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.384m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=MOVING world_disp=0.3836 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.71 mean_area=22631 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_12

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3499 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.350m for static fixture — likely ego-comp residual; classified STATIONARY with MEDIUM confidence | [observed_motion] pipe_motion=DISPLACED world_disp=0.3499 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.61 mean_area=2669 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / laptop_02

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.3295 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.3295 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.48 mean_area=3222 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_02

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.8579 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.8579 → MOVING (HIGH) | [visibility] mean_conf=0.82 mean_area=13617 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_03

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7221 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7221 → MOVING (HIGH) | [visibility] mean_conf=0.51 mean_area=5013 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / laptop_03

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4272 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4272 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.55 mean_area=4664 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_13

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1749 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1749 → STATIONARY (HIGH) | [visibility] mean_conf=0.54 mean_area=1104 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_12

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.34 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.34 → MOVING (MEDIUM) | [visibility] mean_conf=0.56 mean_area=6835 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / monitor_14

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1822 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1822 → STATIONARY (HIGH) | [visibility] mean_conf=0.48 mean_area=1652 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / monitor_15

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7466 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.747m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.7466 → UNCERTAIN (LOW) | [visibility] mean_conf=0.52 mean_area=1647 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / chair_13

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4114 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4114 → MOVING (MEDIUM) | [visibility] mean_conf=0.46 mean_area=12252 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_04

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.409 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.409 → MOVING (HIGH) | [visibility] mean_conf=0.48 mean_area=1761 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / monitor_16

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7408 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.741m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.7408 → UNCERTAIN (LOW) | [visibility] mean_conf=0.67 mean_area=3096 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / laptop_04

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.4729 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.4729 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.60 mean_area=1792 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_14

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9753 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9753 → MOVING (HIGH) | [visibility] mean_conf=0.56 mean_area=6121 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_15

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9638 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.9638 → MOVING (HIGH) | [visibility] mean_conf=0.77 mean_area=66131 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_16

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7623 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7623 → MOVING (HIGH) | [visibility] mean_conf=0.72 mean_area=23910 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_17

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.8954 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.8954 → MOVING (HIGH) | [visibility] mean_conf=0.46 mean_area=18497 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / chair_18

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.3401 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.3401 → MOVING (HIGH) | [visibility] mean_conf=0.72 mean_area=22489 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_05

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1344 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1344 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.58 mean_area=2579 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_06

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | HIGH |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 2.4399 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=2.4399 → MOVING (HIGH) | [visibility] mean_conf=0.78 mean_area=11320 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_07

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2381 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2381 → MOVING (MEDIUM) | [visibility] mean_conf=0.66 mean_area=1715 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / person_08

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.342 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=1.342 → MOVING (HIGH) | [visibility] mean_conf=0.42 mean_area=2086 → HEAVILY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / laptop_05

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6918 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] laptop disp=0.692m unexpectedly large — likely ego-comp residual or misclassification | [observed_motion] pipe_motion=MOVING world_disp=0.6918 → UNCERTAIN (LOW) | [visibility] mean_conf=0.57 mean_area=2249 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → INSUFFICIENT

### d455_02 / laptop_06

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3929 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.3929 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.65 mean_area=3171 → CLEAR | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / desk_01

| Field | Value |
| --- | --- |
| object_class | desk |
| physical_mobility | STRUCTURALLY_IMMOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | VALID |
| evidence_sufficiency | SUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.5456 |

**Rationale:** [physical_mobility] class='desk' → STRUCTURALLY_IMMOVABLE (rule-based by class) | [observed_motion] desk shows disp=0.546m — ego-comp residual or poor detection; classified STATIONARY | [observed_motion] pipe_motion=MOVING world_disp=0.5456 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.52 mean_area=59029 → PARTIALLY_OCCLUDED | [track_identity] VALID — no suspicious patterns detected | [evidence] → SUFFICIENT

### d455_02 / laptop_07

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.3812 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.3812 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.49 mean_area=2950 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / person_09

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.6089 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.6089 → MOVING (HIGH) | [visibility] mean_conf=0.50 mean_area=6882 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (9 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / person_10

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.156 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.156 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.49 mean_area=2462 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (8 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / monitor_17

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0644 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0644 → STATIONARY (HIGH) | [visibility] mean_conf=0.54 mean_area=4235 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / laptop_08

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2407 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2407 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.52 mean_area=1526 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / laptop_09

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1474 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1474 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.58 mean_area=2383 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / chair_19

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1404 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1404 → STATIONARY (HIGH) | [visibility] mean_conf=0.46 mean_area=12967 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / person_11

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.7105 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=MOVING world_disp=0.7105 → MOVING (HIGH) | [visibility] mean_conf=0.69 mean_area=4450 → CLEAR | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / person_12

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0398 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0398 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.77 mean_area=5898 → CLEAR | [track_identity] QUESTIONABLE: short track (7 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / monitor_18

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | CLEAR |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1555 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1555 → STATIONARY (HIGH) | [visibility] mean_conf=0.65 mean_area=3355 → CLEAR | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / monitor_19

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 0.9632 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] disp=0.963m unexpectedly large for 'monitor' — possible ego-comp error or ID switch; classified UNCERTAIN | [observed_motion] pipe_motion=MOVING world_disp=0.9632 → UNCERTAIN (LOW) | [visibility] mean_conf=0.58 mean_area=1464 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10); static class 'monitor' shows world_disp=0.963m > 0.8m — possible ego-comp failure or identity switch | [evidence] → INSUFFICIENT

### d455_02 / laptop_10

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0913 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0913 → STATIONARY (HIGH) | [visibility] mean_conf=0.61 mean_area=3019 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / laptop_11

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1926 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1926 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.45 mean_area=2257 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / desk_02

| Field | Value |
| --- | --- |
| object_class | desk |
| physical_mobility | STRUCTURALLY_IMMOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.1592 |

**Rationale:** [physical_mobility] class='desk' → STRUCTURALLY_IMMOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.1592 → STATIONARY (HIGH) | [visibility] mean_conf=0.44 mean_area=77470 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / chair_20

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2413 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2413 → MOVING (MEDIUM) | [visibility] mean_conf=0.44 mean_area=4706 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (6 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / monitor_20

| Field | Value |
| --- | --- |
| object_class | monitor |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | MEDIUM |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0951 |

**Rationale:** [physical_mobility] class='monitor' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0951 → STATIONARY (HIGH) | [visibility] mean_conf=0.49 mean_area=6221 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / laptop_12

| Field | Value |
| --- | --- |
| object_class | laptop |
| physical_mobility | MOVABLE |
| observed_motion | UNCERTAIN |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | MOVING |
| _pipeline_world_displacement_m | 1.204 |

**Rationale:** [physical_mobility] class='laptop' → MOVABLE (rule-based by class) | [observed_motion] laptop disp=1.204m unexpectedly large — likely ego-comp residual or misclassification | [observed_motion] pipe_motion=MOVING world_disp=1.204 → UNCERTAIN (LOW) | [visibility] mean_conf=0.52 mean_area=6747 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / chair_21

| Field | Value |
| --- | --- |
| object_class | chair |
| physical_mobility | MOVABLE |
| observed_motion | MOVING |
| visibility | PARTIALLY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | DISPLACED |
| _pipeline_world_displacement_m | 0.2293 |

**Rationale:** [physical_mobility] class='chair' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=DISPLACED world_disp=0.2293 → MOVING (MEDIUM) | [visibility] mean_conf=0.56 mean_area=42857 → PARTIALLY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT

### d455_02 / person_13

| Field | Value |
| --- | --- |
| object_class | person |
| physical_mobility | MOVABLE |
| observed_motion | STATIONARY |
| visibility | HEAVILY_OCCLUDED |
| track_identity_validity | QUESTIONABLE |
| evidence_sufficiency | INSUFFICIENT |
| annotation_confidence | LOW |
| _pipeline_observed_motion | STATIC |
| _pipeline_world_displacement_m | 0.0828 |

**Rationale:** [physical_mobility] class='person' → MOVABLE (rule-based by class) | [observed_motion] pipe_motion=STATIC world_disp=0.0828 → STATIONARY (MEDIUM) | [visibility] mean_conf=0.48 mean_area=1613 → HEAVILY_OCCLUDED | [track_identity] QUESTIONABLE: short track (5 total detections < 10) | [evidence] → INSUFFICIENT
