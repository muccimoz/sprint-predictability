import numpy as np
from typing import List, Dict, Any


def compute_predictability(values: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all predictability metrics from a list of completed values.

    values  : completed points or issues per sprint, already filtered (excluded sprints removed)
    config  : dict with keys matching team_config table columns
    """
    n = len(values)
    result = {
        "sprints_in_analysis":  n,
        "effective_grouping_size": 0,
        "rolling_sprint_groups":   0,
        "windows":             [],
        "avg_typical":         None,
        "avg_conservative":    None,
        "avg_ratio":           None,
        "rating":              None,
        "most_recent_ratio":   None,
        "ratio_n_periods_ago": None,
        "trend_delta":         None,
        "recent_trend":        None,
        "recent_avg_ratio":    None,
        "prior_avg_ratio":     None,
        "smoothed_trend_delta": None,
        "smoothed_trend":      None,
        "std_dev":             None,
        "min_ratio":           None,
        "max_ratio":           None,
        "data_volume_warning": None,
    }

    min_warn = config.get("min_sprints_warning", 10)
    if n < min_warn:
        result["data_volume_warning"] = (
            f"Warning: fewer than {min_warn} sprints — results may not be reliable."
        )
    else:
        result["data_volume_warning"] = "Sufficient data"

    if n == 0:
        return result

    analysis_mode       = config.get("analysis_mode", "Rolling")
    w                   = int(config.get("sprints_per_window", 5))
    conservative_pct    = float(config.get("conservative_percentile", 0.15))
    strong_thr          = float(config.get("strong_threshold", 0.5))
    moderate_thr        = float(config.get("moderate_threshold", 0.33))
    needs_attention_thr = float(config.get("needs_attention_threshold", 0.25))
    trend_lookback      = int(config.get("trend_lookback", 5))

    # Effective grouping size
    effective_w = n if analysis_mode == "All" else min(w, n)
    result["effective_grouping_size"] = effective_w

    # Build windows
    if analysis_mode == "All":
        windows_values = [values]
        result["rolling_sprint_groups"] = 1
    else:
        if n < effective_w:
            result["rolling_sprint_groups"] = 0
            return result
        num_windows = n - effective_w + 1
        result["rolling_sprint_groups"] = num_windows
        windows_values = [values[i: i + effective_w] for i in range(num_windows)]

    # Per-window metrics
    window_results = []
    for i, window in enumerate(windows_values):
        arr         = np.array(window, dtype=float)
        typical     = float(np.median(arr))
        conservative = float(np.percentile(arr, conservative_pct * 100))
        ratio       = (conservative / typical) if typical > 0 else 0.0
        window_results.append({
            "window":       i + 1,
            "typical":      typical,
            "conservative": conservative,
            "ratio":        ratio,
        })

    result["windows"] = window_results

    ratios       = [w["ratio"]       for w in window_results]
    typicals     = [w["typical"]     for w in window_results]
    conservatives = [w["conservative"] for w in window_results]

    result["avg_typical"]      = float(np.mean(typicals))
    result["avg_conservative"] = float(np.mean(conservatives))
    result["avg_ratio"]        = float(np.mean(ratios))
    result["min_ratio"]        = float(np.min(ratios))
    result["max_ratio"]        = float(np.max(ratios))
    result["std_dev"]          = float(np.std(values, ddof=1)) if n > 1 else 0.0

    # Rating
    avg = result["avg_ratio"]
    if avg >= strong_thr:
        result["rating"] = "Strong"
    elif avg >= moderate_thr:
        result["rating"] = "Moderate"
    elif avg >= needs_attention_thr:
        result["rating"] = "Needs Attention"
    else:
        result["rating"] = "Very Weak"

    # Most recent ratio
    result["most_recent_ratio"] = ratios[-1]

    # Point-to-point trend
    if len(ratios) >= trend_lookback + 1:
        result["ratio_n_periods_ago"] = ratios[-(trend_lookback + 1)]
        result["trend_delta"]         = ratios[-1] - result["ratio_n_periods_ago"]
        d = result["trend_delta"]
        result["recent_trend"] = "Improving" if d > 0.05 else ("Declining" if d < -0.05 else "Stable")
    else:
        result["recent_trend"] = "Not enough windows"

    # Smoothed trend (3-window averages)
    if len(ratios) >= 6:
        result["recent_avg_ratio"]     = float(np.mean(ratios[-3:]))
        result["prior_avg_ratio"]      = float(np.mean(ratios[-6:-3]))
        result["smoothed_trend_delta"] = result["recent_avg_ratio"] - result["prior_avg_ratio"]
        d = result["smoothed_trend_delta"]
        result["smoothed_trend"] = "Improving" if d > 0.05 else ("Declining" if d < -0.05 else "Stable")
    else:
        result["smoothed_trend"] = "Not enough windows"

    return result
