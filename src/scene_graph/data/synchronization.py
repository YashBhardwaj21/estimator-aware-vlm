"""Timestamp synchronization for TUM RGB-D sequences.

Implements a dynamic-programming based monotonic matching algorithm that explicitly:
1. Maximizes the total number of matches |M|.
2. Minimizes the total temporal error sum(|t_i - t_j|) as a tie-breaker.
Ensures strict monotonic chronological ordering.
"""

from typing import List, Tuple


def associate(
    primary_timestamps: List[float], 
    secondary_timestamps: List[float], 
    max_dt: float = 0.02
) -> List[Tuple[int, int]]:
    """Optimal monotonic one-to-one timestamp matching via Dynamic Programming.

    Objective:
        max |M|
    Subject to:
        |primary[i] - secondary[j]| <= max_dt
        i1 < i2 => j1 < j2
    Tie-break:
        min sum(|primary[i] - secondary[j]|)

    Args:
        primary_timestamps: List of timestamps (e.g., RGB).
        secondary_timestamps: List of timestamps (e.g., Depth).
        max_dt: Maximum allowed time difference in seconds.

    Returns:
        List of matched index pairs (primary_idx, secondary_idx) sorted chronologically.
    """
    N = len(primary_timestamps)
    M = len(secondary_timestamps)
    
    if N == 0 or M == 0:
        return []
        
    # DP table storing tuples: (matches_count, negative_total_error)
    # We maximize this tuple lexicographically.
    dp = [[(0, 0.0)] * (M + 1) for _ in range(N + 1)]
    # Parent pointers: 0=diag (match), 1=up (skip primary), 2=left (skip secondary)
    parent = [[-1] * (M + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        p_t = primary_timestamps[i - 1]
        for j in range(1, M + 1):
            s_t = secondary_timestamps[j - 1]
            
            # Option 1: Skip primary i
            best_val = dp[i-1][j]
            best_dir = 1
            
            # Option 2: Skip secondary j
            if dp[i][j-1] > best_val:
                best_val = dp[i][j-1]
                best_dir = 2
                
            # Option 3: Match primary i with secondary j
            diff = abs(p_t - s_t)
            if diff <= max_dt:
                prev_matches, prev_neg_err = dp[i-1][j-1]
                match_val = (prev_matches + 1, prev_neg_err - diff)
                if match_val > best_val:
                    best_val = match_val
                    best_dir = 0
                    
            dp[i][j] = best_val
            parent[i][j] = best_dir
            
    # Backtrack to find the optimal matches
    matches = []
    i, j = N, M
    while i > 0 and j > 0:
        direction = parent[i][j]
        if direction == 0:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            break
            
    matches.reverse()
    return matches
