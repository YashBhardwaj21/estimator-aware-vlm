# Tracking Consistency Report

Source run: `C:\Users\Yash Bhardwaj\Desktop\vlm\results\perception_harvesting\run_20261101_234859`
Tracks checked: 177
Observations checked: 885

## Summary

| Severity | Count |
| --- | --- |
| ERROR   | 0   |
| WARNING | 0 |
| TOTAL   | 0   |

## Result

No consistency issues detected. All checks passed.

## Checks Performed

1. Track IDs unique within each sequence.
2. Observation timestamps monotonically ordered.
3. Frame indices are non-negative.
4. No duplicate obs_id within a track.
5. Declared observation count matches actual jsonl count.
6. Crop files exist on disk (first + last obs per track sampled).

> Issues are reported only, never silently corrected. Investigate each WARNING before drawing conclusions about temporal evidence.