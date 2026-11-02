# Stage 3 VLM Evaluation Results

Run: `20261101_193504`
Tracks evaluated: 30
Split: development
Confidence filter: ['HIGH', 'MEDIUM']

## Results by Configuration

| Config | n | Bal.Acc | Macro F1 | Lat.Median | Parse Fail |
| --- | --- | --- | --- | --- | --- |
| V1 | 30 | 0.567 | 0.365 | 18686ms | 0 |
| V2 | 30 | 0.379 | 0.109 | 30848ms | 0 |
| V3 | 30 | 0.444 | 0.192 | 32895ms | 0 |
| V4 | 30 | 0.544 | 0.297 | 18983ms | 0 |

## Per-Class Breakdown

### V1

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.050 | 1.000 | 0.095 | 1 |
| MOVABLE | 1.000 | 0.500 | 0.667 | 14 |
| MOBILE | 1.000 | 0.200 | 0.333 | 15 |

### V2

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.036 | 1.000 | 0.069 | 1 |
| MOVABLE | 1.000 | 0.071 | 0.133 | 14 |
| MOBILE | 1.000 | 0.067 | 0.125 | 15 |

### V3

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.040 | 1.000 | 0.077 | 1 |
| MOVABLE | 0.000 | 0.000 | 0.000 | 14 |
| MOBILE | 1.000 | 0.333 | 0.500 | 15 |

### V4

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| IMMOVABLE | 0.062 | 1.000 | 0.118 | 1 |
| MOVABLE | 0.583 | 0.500 | 0.538 | 14 |
| MOBILE | 1.000 | 0.133 | 0.235 | 15 |
