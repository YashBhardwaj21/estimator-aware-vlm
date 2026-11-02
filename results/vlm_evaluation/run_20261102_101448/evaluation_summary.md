# Controlled VLM Evaluation Results

Run: `20261102_101448`
Tracks evaluated: 30
Split: development
Confidence filter: ['HIGH', 'MEDIUM']

## Results by Configuration

| Config | n | Bal.Acc | Macro F1 | Lat.Median | Parse Fail |
| --- | --- | --- | --- | --- | --- |
| V1 | 27 | 0.500 | 0.233 | 18574ms | 0 |
| V2 | 26 | 0.606 | 0.283 | 31251ms | 0 |
| V3 | 26 | 0.576 | 0.258 | 32621ms | 0 |
| V4 | 26 | 0.364 | 0.073 | 21471ms | 0 |
| Baseline_0A | 30 | 0.667 | 0.655 | 0ms | 0 |
| Baseline_0B | 30 | 1.000 | 1.000 | 0ms | 0 |

## Per-Class Breakdown

### V1

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.053 | 1.000 | 0.100 | 1 |
| MOVABLE | 0.750 | 0.500 | 0.600 | 12 |
| MOBILE | 0.000 | 0.000 | 0.000 | 14 |

### V2

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.143 | 1.000 | 0.250 | 1 |
| MOVABLE | 0.474 | 0.818 | 0.600 | 11 |
| MOBILE | 0.000 | 0.000 | 0.000 | 14 |

### V3

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.100 | 1.000 | 0.182 | 1 |
| MOVABLE | 0.500 | 0.727 | 0.593 | 11 |
| MOBILE | 0.000 | 0.000 | 0.000 | 14 |

### V4

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.045 | 1.000 | 0.087 | 1 |
| MOVABLE | 0.250 | 0.091 | 0.133 | 11 |
| MOBILE | 0.000 | 0.000 | 0.000 | 14 |

### Baseline_0A

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.000 | 0.000 | 0.000 | 1 |
| MOVABLE | 0.933 | 1.000 | 0.966 | 14 |
| MOBILE | 1.000 | 1.000 | 1.000 | 15 |

### Baseline_0B

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 1.000 | 1.000 | 1.000 | 1 |
| MOVABLE | 1.000 | 1.000 | 1.000 | 14 |
| MOBILE | 1.000 | 1.000 | 1.000 | 15 |
