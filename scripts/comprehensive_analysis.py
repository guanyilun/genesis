#!/usr/bin/env python3
"""
Comprehensive Genesis Analysis Pipeline
Computes every available metric from experimental data and produces
a publication-quality report.

Data structure: list of replicates, each with census_data array.
Each census point has: tick, population, avg_genome_length, load_before_move_frac,
load_gene_freq, trace_mean, trace_std, genome_length_stdev, avg_energy, etc.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = "/Users/yilun/Workspace/scratch/genesis/analysis_output"
DATA_DIR = "/tmp"

# ─── Loading ──────────────────────────────────────────────────────────────

def load_replicates(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_all():
    files = {
        "early_window": "early_window_valid.json",
        "patchy": "replicates_patchy_valid.json",
        "uniform": "replicates_uniform_valid.json",
    }
    datasets = {}
    for label, fname in files.items():
        data = load_replicates(fname)
        if data:
            n_census = len(data[0]["census_data"]) if data else 0
            print(f"  [OK] {fname}: {len(data)} replicates, {n_census} census points each")
        else:
            print(f"  [SKIP] {fname} not found")
        datasets[label] = data
    return datasets

# ─── Per-Replicate Metrics ───────────────────────────────────────────────

def extract_series(replicate):
    """Extract all time series from a single replicate."""
    census = replicate["census_data"]
    ticks = np.array([c["tick"] for c in census])
    population = np.array([c.get("population", 0) for c in census])
    genome_len = np.array([c.get("avg_genome_length", 0) for c in census])
    genome_std = np.array([c.get("genome_length_stdev", 0) for c in census])
    energy = np.array([c.get("avg_energy", 0) for c in census])
    age = np.array([c.get("avg_age", 0) for c in census])
    
    # Stigmergy / communication / LOAD gene frequency fields
    load_move_frac = np.array([c.get("load_before_move_frac", 0) for c in census])
    load_gene_freq = np.array([c.get("load_gene_freq", 0) for c in census])
    store_gene_freq = np.array([c.get("store_gene_freq", 0) for c in census])
    stigmergy_gene_freq = np.array([c.get("stigmergy_gene_freq", 0) for c in census])
    comm_gene_freq = np.array([c.get("communication_gene_freq", 0) for c in census])
    store_after_compute = np.array([c.get("store_after_compute_frac", 0) for c in census])
    directional_load_div = np.array([c.get("directional_load_diversity", 0) for c in census])
    
    # Trace / environment
    trace_util = np.array([c.get("trace_utilization", 0) for c in census])
    trace_mean = np.array([c.get("avg_trace_value", c.get("trace_mean", 0)) for c in census])
    
    # Parasites
    parasite_count = np.array([c.get("parasite_count", 0) for c in census])
    parasite_frac = np.array([c.get("parasite_frac", 0) for c in census])
    
    # Mutual information
    mutual_info = np.array([c.get("mutual_information", 0) for c in census])
    mutual_info_raw = np.array([c.get("mutual_information_raw", 0) for c in census])
    
    # Throughput
    tps = np.array([c.get("tps", 0) for c in census])
    
    # Opcode distributions over time (top_opcodes per census point)
    opcode_series = []
    for c in census:
        top_ops = c.get("top_opcodes", [])
        if isinstance(top_ops, list) and top_ops:
            if isinstance(top_ops[0], list):
                opcode_series.append({str(k): v for k, v in top_ops})
            else:
                opcode_series.append({})
        else:
            opcode_series.append({})
    
    # Behavioral profiles over time
    profile_series = []
    for c in census:
        prof = c.get("avg_profile", {})
        if isinstance(prof, dict):
            profile_series.append(prof)
        else:
            profile_series.append({})
    
    return {
        "ticks": ticks,
        "population": population,
        "genome_len": genome_len,
        "genome_std": genome_std,
        "energy": energy,
        "age": age,
        "load_move_frac": load_move_frac,
        "load_gene_freq": load_gene_freq,
        "store_gene_freq": store_gene_freq,
        "stigmergy_gene_freq": stigmergy_gene_freq,
        "comm_gene_freq": comm_gene_freq,
        "store_after_compute": store_after_compute,
        "directional_load_div": directional_load_div,
        "trace_util": trace_util,
        "trace_mean": trace_mean,
        "parasite_count": parasite_count,
        "parasite_frac": parasite_frac,
        "mutual_info": mutual_info,
        "mutual_info_raw": mutual_info_raw,
        "tps": tps,
        "opcode_series": opcode_series,
        "profile_series": profile_series,
        "n_census": len(census),
    }

def compute_replicate_metrics(series):
    """Compute derived metrics from a replicate's time series."""
    pop = series["population"]
    ticks = series["ticks"]
    
    # Basic population dynamics
    peak_pop = int(np.max(pop))
    peak_tick = int(ticks[np.argmax(pop)])
    final_pop = int(pop[-1])
    mean_pop = float(np.mean(pop))
    
    # Growth rate: exponential fit to first 20% of census points
    n_early = max(3, len(pop) // 5)
    early_pop = pop[:n_early]
    early_ticks = ticks[:n_early]
    nonzero = early_pop > 0
    growth_rate = 0.0
    if nonzero.sum() >= 2:
        try:
            log_pop = np.log(early_pop[nonzero])
            coeffs = np.polyfit(early_ticks[nonzero], log_pop, 1)
            growth_rate = float(coeffs[0])
        except:
            growth_rate = 0.0
    
    # Extinction check
    extinct = final_pop == 0
    extinction_tick = None
    if extinct:
        zero_mask = pop == 0
        if np.any(zero_mask):
            extinction_tick = int(ticks[np.argmax(zero_mask)])
    
    # ── Decoupling detection ──
    # LOAD gene frequency persists while LOAD execution (load_before_move_frac) drops
    load_gene = series["load_gene_freq"]
    load_exec = series["load_move_frac"]
    
    decoupling_detected = False
    decoupling_tick = None
    if len(load_exec) > 5:
        peak_exec_idx = np.argmax(load_exec)
        peak_exec = load_exec[peak_exec_idx]
        late_start = int(len(load_exec) * 0.6)
        late_exec = load_exec[late_start:]
        late_gene = load_gene[late_start:]
        
        if peak_exec > 0.01:
            exec_ratio = float(np.mean(late_exec) / peak_exec) if peak_exec > 0 else 0
            gene_mean_late = float(np.mean(late_gene))
            gene_mean_early = float(np.mean(load_gene[:late_start]))
            
            if exec_ratio < 0.2 and gene_mean_late > gene_mean_early * 0.5:
                decoupling_detected = True
                decoupling_tick = int(ticks[peak_exec_idx])
    
    # ── Genome complexity trend ──
    genome_trend = "growing" if series["genome_len"][-1] > series["genome_len"][0] * 1.05 else \
                  "shrinking" if series["genome_len"][-1] < series["genome_len"][0] * 0.95 else "stable"
    
    # ── Behavioral profile summaries ──
    # Collapse profile series into early vs late phase means
    profiles = series["profile_series"]
    profile_keys = set()
    for p in profiles:
        profile_keys.update(p.keys())
    profile_keys.discard("length")
    
    n_half = len(profiles) // 2
    early_profile = {k: np.mean([p.get(k, 0) for p in profiles[:n_half]]) for k in profile_keys}
    late_profile = {k: np.mean([p.get(k, 0) for p in profiles[n_half:]]) for k in profile_keys}
    
    # Profile shift magnitude (sum of absolute differences)
    profile_shift = float(np.sum([abs(early_profile.get(k, 0) - late_profile.get(k, 0)) for k in profile_keys]))
    
    # Dominant behavioral class in late phase
    dominant_behavior = max(late_profile, key=late_profile.get) if late_profile else "unknown"
    
    # ── Stigmergy / communication / trace metrics ──
    stigmergy_final = float(series["stigmergy_gene_freq"][-1])
    stigmergy_peak = float(np.max(series["stigmergy_gene_freq"]))
    store_final = float(series["store_gene_freq"][-1])
    comm_final = float(series["comm_gene_freq"][-1])
    
    trace_util_final = float(series["trace_util"][-1])
    trace_util_peak = float(np.max(series["trace_util"]))
    trace_mean_final = float(series["trace_mean"][-1])
    trace_mean_peak = float(np.max(series["trace_mean"]))
    
    # Trace utilization trend
    if series["trace_util"][0] > 0 and series["trace_util"][-1] > series["trace_util"][0] * 1.1:
        trace_trend = "increasing"
    elif series["trace_util"][-1] < series["trace_util"][0] * 0.9:
        trace_trend = "decreasing"
    else:
        trace_trend = "stable"
    
    # ── Mutual information ──
    mi_final = float(series["mutual_info"][-1])
    mi_peak = float(np.max(series["mutual_info"]))
    mi_trend = "increasing" if series["mutual_info"][-1] > series["mutual_info"][0] * 1.1 else \
               "decreasing" if series["mutual_info"][-1] < series["mutual_info"][0] * 0.9 else "stable"
    
    # ── Parasites ──
    parasite_peak = int(np.max(series["parasite_count"]))
    parasite_present = parasite_peak > 0
    
    # ── Directional load diversity ──
    dir_load_final = float(series["directional_load_div"][-1])
    dir_load_peak = float(np.max(series["directional_load_div"]))
    
    # ── Energy dynamics ──
    energy_peak = float(np.max(series["energy"]))
    energy_final = float(series["energy"][-1])
    energy_trend = "declining" if energy_final < energy_peak * 0.5 else "sustained"
    
    return {
        "peak_population": peak_pop,
        "peak_tick": peak_tick,
        "final_population": final_pop,
        "mean_population": mean_pop,
        "growth_rate": growth_rate,
        "extinct": extinct,
        "extinction_tick": extinction_tick,
        "decoupling_detected": decoupling_detected,
        "decoupling_tick": decoupling_tick,
        "genome_trend": genome_trend,
        "genome_len_final": float(series["genome_len"][-1]),
        "genome_len_initial": float(series["genome_len"][0]),
        "profile_shift": profile_shift,
        "dominant_behavior_late": dominant_behavior,
        "early_profile": early_profile,
        "late_profile": late_profile,
        "stigmergy_gene_final": stigmergy_final,
        "stigmergy_gene_peak": stigmergy_peak,
        "store_gene_final": store_final,
        "comm_gene_final": comm_final,
        "trace_util_final": trace_util_final,
        "trace_util_peak": trace_util_peak,
        "trace_mean_final": trace_mean_final,
        "trace_mean_peak": trace_mean_peak,
        "trace_trend": trace_trend,
        "mutual_info_final": mi_final,
        "mutual_info_peak": mi_peak,
        "mutual_info_trend": mi_trend,
        "parasite_peak": parasite_peak,
        "parasite_present": parasite_present,
        "directional_load_final": dir_load_final,
        "directional_load_peak": dir_load_peak,
        "energy_peak": energy_peak,
        "energy_final": energy_final,
        "energy_trend": energy_trend,
        "total_ticks": int(ticks[-1]),
        "n_census": len(pop),
    }

# ─── Cross-Replicate Aggregation ─────────────────────────────────────────

def aggregate_metrics(label, replicates):
    """Compute metrics for each replicate, then aggregate."""
    if replicates is None:
        return None
    
    per_rep = []
    for i, rep in enumerate(replicates):
        try:
            series = extract_series(rep)
            metrics = compute_replicate_metrics(series)
            metrics["seed"] = rep.get("seed", i)
            metrics["env"] = rep.get("env", rep.get("label", "unknown"))
            per_rep.append(metrics)
        except Exception as e:
            print(f"    [WARN] Replicate {i} failed: {e}")
    
    if not per_rep:
        return None
    
    # Aggregate statistics
    agg = {}
    numeric_fields = [
        "peak_population", "final_population", "mean_population", 
        "growth_rate", "genome_len_final", "genome_len_initial",
        "stigmergy_gene_final", "stigmergy_gene_peak",
        "store_gene_final", "comm_gene_final",
        "trace_util_final", "trace_util_peak",
        "trace_mean_final", "trace_mean_peak",
        "mutual_info_final", "mutual_info_peak",
        "directional_load_final", "directional_load_peak",
        "profile_shift", "energy_peak", "energy_final",
    ]
    
    for field in numeric_fields:
        values = [m[field] for m in per_rep if field in m]
        if values:
            agg[field] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    
    # Count occurrences
    agg["n_replicates"] = len(per_rep)
    agg["extinction_count"] = sum(1 for m in per_rep if m.get("extinct", False))
    agg["decoupling_count"] = sum(1 for m in per_rep if m.get("decoupling_detected", False))
    agg["parasite_count"] = sum(1 for m in per_rep if m.get("parasite_present", False))
    
    # Categorical aggregations
    agg["genome_trends"] = defaultdict(int)
    agg["trace_trends"] = defaultdict(int)
    agg["mi_trends"] = defaultdict(int)
    agg["energy_trends"] = defaultdict(int)
    agg["dominant_behaviors"] = defaultdict(int)
    for m in per_rep:
        agg["genome_trends"][m["genome_trend"]] += 1
        agg["trace_trends"][m["trace_trend"]] += 1
        agg["mi_trends"][m["mutual_info_trend"]] += 1
        agg["energy_trends"][m["energy_trend"]] += 1
        agg["dominant_behaviors"][m["dominant_behavior_late"]] += 1
    
    return {"per_replicate": per_rep, "aggregates": agg, "label": label}

# ─── Report Generation ───────────────────────────────────────────────────

def generate_report(results):
    """Generate a readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  GENESIS — COMPREHENSIVE ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")
    
    for label, result in results.items():
        if result is None:
            continue
        
        agg = result["aggregates"]
        per_rep = result["per_replicate"]
        
        lines.append(f"┌{'─' * 70}┐")
        lines.append(f"│  CONDITION: {label:<58} │")
        lines.append(f"│  Replicates: {agg['n_replicates']:<55} │")
        lines.append(f"└{'─' * 70}┘")
        lines.append("")
        
        # Population dynamics
        lines.append("  POPULATION DYNAMICS:")
        for field in ["peak_population", "final_population", "mean_population", "growth_rate"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.4f}  std={v['std']:>10.4f}  range=[{v['min']:.4f}, {v['max']:.4f}]")
        lines.append(f"    {'extinction_count':30s}  {agg['extinction_count']}/{agg['n_replicates']} replicates went extinct")
        lines.append("")
        
        # Decoupling
        lines.append("  DECOUPLING (LOAD gene persists, execution drops):")
        lines.append(f"    Decoupling detected in {agg['decoupling_count']}/{agg['n_replicates']} replicates")
        if agg['decoupling_count'] > 0:
            coupling_ticks = [m["decoupling_tick"] for m in per_rep if m.get("decoupling_detected") and m.get("decoupling_tick")]
            if coupling_ticks:
                lines.append(f"    Decoupling onset ticks: {coupling_ticks}")
        lines.append("")
        
        # Genomic complexity
        lines.append("  GENOMIC COMPLEXITY:")
        for field in ["genome_len_initial", "genome_len_final"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.2f}  std={v['std']:>10.2f}")
        lines.append(f"    Genome length trends: {dict(agg['genome_trends'])}")
        lines.append("")
        
        # Behavioral profiles
        lines.append("  BEHAVIORAL PROFILE SHIFT:")
        if "profile_shift" in agg:
            v = agg["profile_shift"]
            lines.append(f"    {'profile_shift':30s}  mean={v['mean']:>10.4f}  std={v['std']:>10.4f}")
        lines.append(f"    Dominant late behaviors: {dict(agg.get('dominant_behaviors', {}))}")
        lines.append("")
        
        # Stigmergy & communication
        lines.append("  STIGMERGY / COMMUNICATION:")
        for field in ["stigmergy_gene_final", "stigmergy_gene_peak", "store_gene_final", "comm_gene_final"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.6f}  range=[{v['min']:.6f}, {v['max']:.6f}]")
        lines.append("")
        
        # Environment interaction
        lines.append("  ENVIRONMENT INTERACTION (trace info):")
        for field in ["trace_util_final", "trace_util_peak", "trace_mean_final", "trace_mean_peak"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.6f}  range=[{v['min']:.6f}, {v['max']:.6f}]")
        lines.append(f"    Trace utilization trends: {dict(agg.get('trace_trends', {}))}")
        lines.append("")
        
        # Mutual information
        lines.append("  MUTUAL INFORMATION (genome↔behavior coupling):")
        for field in ["mutual_info_final", "mutual_info_peak"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.6f}  range=[{v['min']:.6f}, {v['max']:.6f}]")
        lines.append(f"    MI trends: {dict(agg.get('mi_trends', {}))}")
        lines.append("")
        
        # Directional load diversity
        lines.append("  DIRECTIONAL LOAD DIVERSITY:")
        for field in ["directional_load_final", "directional_load_peak"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.6f}  range=[{v['min']:.6f}, {v['max']:.6f}]")
        lines.append("")
        
        # Parasites
        lines.append(f"  PARASITES: detected in {agg['parasite_count']}/{agg['n_replicates']} replicates")
        lines.append("")
        
        # Energy
        lines.append("  ENERGY:")
        for field in ["energy_peak", "energy_final"]:
            if field in agg:
                v = agg[field]
                lines.append(f"    {field:30s}  mean={v['mean']:>10.2f}  std={v['std']:>10.2f}")
        lines.append(f"    Energy trends: {dict(agg.get('energy_trends', {}))}")
        lines.append("")
        
        # Per-replicate summary table
        lines.append("  PER-REPLICATE SUMMARY:")
        header = f"    {'Seed':>6s}  {'PeakPop':>8s}  {'FinalPop':>9s}  {'GenomeLen':>10s}  {'Stigmergy':>10s}  {'TraceUtil':>10s}  {'Decoupled':>10s}"
        lines.append(header)
        sep = "    " + "─" * (len(header) - 4)
        lines.append(sep)
        for m in per_rep:
            lines.append(f"    {m.get('seed', '?'):>6}  {m['peak_population']:>8d}  {m['final_population']:>9d}  "
                        f"{m['genome_len_final']:>10.1f}  {m.get('stigmergy_gene_final', 0):>10.4f}  "
                        f"{m.get('trace_util_final', 0):>10.4f}  {str(m['decoupling_detected']):>10s}")
        lines.append("")
        lines.append("─" * 72)
        lines.append("")
    
    # Cross-condition comparison
    lines.append("")
    lines.append("┌" + "─" * 70 + "┐")
    lines.append("│  CROSS-CONDITION COMPARISON" + " " * 44 + "│")
    lines.append("└" + "─" * 70 + "┘")
    lines.append("")
    
    for field in ["peak_population", "final_population", "growth_rate", "genome_len_final",
                  "stigmergy_gene_final", "trace_util_final", "mutual_info_final",
                  "profile_shift", "energy_final", "directional_load_final"]:
        lines.append(f"  {field}:")
        for label, result in results.items():
            if result and field in result["aggregates"]:
                v = result["aggregates"][field]
                lines.append(f"    {label:20s}  mean={v['mean']:>12.6f}  std={v['std']:>12.6f}")
        lines.append("")
    
    lines.append("  CATEGORICAL COMPARISONS:")
    for cat, cat_name in [("decoupling_count", "Decoupling"), 
                          ("parasite_count", "Parasites detected"),
                          ("extinction_count", "Extinctions")]:
        lines.append(f"    {cat_name}:")
        for label, result in results.items():
            if result:
                agg = result["aggregates"]
                lines.append(f"      {label:20s}  {agg[cat]}/{agg['n_replicates']} replicates")
        lines.append("")
    
    for cat, cat_name in [("genome_trends", "Genome trends"), 
                          ("trace_trends", "Trace trends"),
                          ("dominant_behaviors", "Dominant behaviors"),
                          ("energy_trends", "Energy trends")]:
        lines.append(f"    {cat_name}:")
        for label, result in results.items():
            if result:
                agg = result["aggregates"]
                lines.append(f"      {label:20s}  {dict(agg.get(cat, {}))}")
        lines.append("")
    
    return "\n".join(lines)

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("GENESIS COMPREHENSIVE ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    
    datasets = load_all()
    
    print("\nComputing metrics per condition...")
    results = {}
    for label, replicates in datasets.items():
        if replicates is None:
            continue
        print(f"  Processing {label}...")
        results[label] = aggregate_metrics(label, replicates)
    
    print("\nGenerating report...")
    report = generate_report(results)
    print(report)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save raw metrics as JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    
    print(f"\n✓ Report saved to: {report_path}")
    print(f"✓ Metrics saved to: {metrics_path}")
    print("✓ Analysis complete!")

if __name__ == "__main__":
    main()