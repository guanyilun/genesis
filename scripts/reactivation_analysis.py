#!/usr/bin/env python3
"""
Visualization and Analysis for Latent Reactivation Experiment
================================================================

Reads results JSON and produces:
  1. Summary statistics (printed)
  2. Time-series plots (L→M over time for transplant vs control)
  3. A results HTML page

Usage:
    python reactivation_analysis.py [results.json]
    # If no file given, uses most recent runs/latent_reactivation_*.json
"""

import json
import sys
import os
import glob
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def find_results(path=None):
    if path and os.path.exists(path):
        return path
    files = sorted(glob.glob('runs/latent_reactivation_*.json'))
    if not files:
        return None
    return files[-1]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def find_emergence_tick(census_data, threshold=0.10):
    """First tick where L→M exceeds threshold."""
    for c in census_data:
        if c.get('load_before_move_frac', 0) >= threshold:
            return c['tick']
    return None


def extract_lm_series(census_data):
    ticks = [c['tick'] for c in census_data]
    lms = [c.get('load_before_move_frac', 0) for c in census_data]
    pops = [c.get('population', 0) for c in census_data]
    mis = [c.get('mutual_information', 0) for c in census_data]
    return ticks, lms, pops, mis


def analyze(results, verbose=True):
    donor = results.get('donor', {})
    transplants = results.get('transplants', [])
    controls = results.get('controls', [])
    params = results.get('params', {})

    if verbose:
        print("=" * 70)
        print("  LATENT REACTIVATION — ANALYSIS")
        print("=" * 70)
        print(f"\n  Parameters: env={params.get('env')}, donor_ticks={params.get('ticks_donor')}, "
              f"transplant_ticks={params.get('ticks_transplant')}, n={params.get('n_replicates')}")
        print(f"\n  DONOR PHASE:")
        print(f"    Completed ticks: {donor.get('ticks_completed', 'N/A')}")
        print(f"    Genomes harvested: {donor.get('genomes_harvested', 'N/A')}")
        print(f"    Genomic L→M at harvest: {donor.get('genomic_lm_fraction', 0):.3f}")

    # Per-replicate analysis
    trans_data = []
    for i, run in enumerate(transplants):
        cd = run.get('census_data', [])
        if not cd:
            continue
        ticks, lms, pops, mis = extract_lm_series(cd)
        emergence = find_emergence_tick(cd)
        peak_lm = max(lms) if lms else 0
        final_lm = lms[-1] if lms else 0
        trans_data.append({
            'idx': i, 'ticks': ticks, 'lms': lms, 'pops': pops, 'mis': mis,
            'emergence': emergence, 'peak_lm': peak_lm, 'final_lm': final_lm,
        })

    ctrl_data = []
    for i, run in enumerate(controls):
        cd = run.get('census_data', [])
        if not cd:
            continue
        ticks, lms, pops, mis = extract_lm_series(cd)
        emergence = find_emergence_tick(cd)
        peak_lm = max(lms) if lms else 0
        final_lm = lms[-1] if lms else 0
        ctrl_data.append({
            'idx': i, 'ticks': ticks, 'lms': lms, 'pops': pops, 'mis': mis,
            'emergence': emergence, 'peak_lm': peak_lm, 'final_lm': final_lm,
        })

    if verbose:
        print(f"\n  TRANSPLANT REPLICATES ({len(trans_data)} valid):")
        for td in trans_data:
            print(f"    Rep {td['idx']}: emergence={td['emergence']}, peak_L→M={td['peak_lm']:.3f}, "
                  f"final_L→M={td['final_lm']:.3f}")
        trans_emergences = [td['emergence'] for td in trans_data if td['emergence'] is not None]
        trans_peaks = [td['peak_lm'] for td in trans_data]
        trans_finals = [td['final_lm'] for td in trans_data]

        print(f"\n  CONTROL REPLICATES ({len(ctrl_data)} valid):")
        for cd in ctrl_data:
            print(f"    Rep {cd['idx']}: emergence={cd['emergence']}, peak_L→M={cd['peak_lm']:.3f}, "
                  f"final_L→M={cd['final_lm']:.3f}")
        ctrl_emergences = [cd['emergence'] for cd in ctrl_data if cd['emergence'] is not None]
        ctrl_peaks = [cd['peak_lm'] for cd in ctrl_data]
        ctrl_finals = [cd['final_lm'] for cd in ctrl_data]

        # Aggregate stats
        if trans_emergences:
            print(f"\n  TRANSPLANT mean emergence: {sum(trans_emergences)/len(trans_emergences):.0f} ticks "
                  f"(n={len(trans_emergences)})")
        else:
            print(f"\n  TRANSPLANT: no replicates reached L→M > 0.10")

        if ctrl_emergences:
            print(f"  CONTROL mean emergence: {sum(ctrl_emergences)/len(ctrl_emergences):.0f} ticks "
                  f"(n={len(ctrl_emergences)})")
        else:
            print(f"  CONTROL: no replicates reached L→M > 0.10")

        if trans_peaks:
            print(f"\n  TRANSPLANT mean peak L→M: {sum(trans_peaks)/len(trans_peaks):.3f}")
        if ctrl_peaks:
            print(f"  CONTROL mean peak L→M: {sum(ctrl_peaks)/len(ctrl_peaks):.3f}")

        # Statistical tests
        if HAS_SCIPY and len(trans_peaks) >= 2 and len(ctrl_peaks) >= 2:
            u, p = stats.mannwhitneyu(trans_peaks, ctrl_peaks, alternative='two-sided')
            print(f"\n  Mann-Whitney U (peak L→M): U={u:.1f}, p={p:.4f}")
            if p < 0.05:
                direction = "TRANSPLANT > CONTROL" if sum(trans_peaks)/len(trans_peaks) > sum(ctrl_peaks)/len(ctrl_peaks) else "CONTROL > TRANSPLANT"
                print(f"  *** Significant: {direction} ***")
            else:
                print(f"  No significant difference (p >= 0.05)")

        if HAS_SCIPY and len(trans_emergences) >= 2 and len(ctrl_emergences) >= 2:
            u, p = stats.mannwhitneyu(trans_emergences, ctrl_emergences, alternative='two-sided')
            print(f"\n  Mann-Whitney U (emergence time): U={u:.1f}, p={p:.4f}")

        print("\n" + "=" * 70)

    return {
        'trans_data': trans_data,
        'ctrl_data': ctrl_data,
        'donor': donor,
    }


def plot_results(results, output_dir='analysis_output/reactivation'):
    """Generate time-series plots comparing transplant vs control."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    analyzed = analyze(results, verbose=False)
    trans_data = analyzed['trans_data']
    ctrl_data = analyzed['ctrl_data']
    donor = analyzed['donor']

    # ── Plot 1: L→M time series (donor + transplant + control) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Donor phase
    if donor.get('census_data'):
        d_ticks = [c['tick'] for c in donor['census_data']]
        d_lms = [c.get('load_before_move_frac', 0) for c in donor['census_data']]
        axes[0].plot(d_ticks, d_lms, 'b-', linewidth=1.5)
        axes[0].set_title('Donor Phase')
        axes[0].set_xlabel('Tick')
        axes[0].set_ylabel('L→M fraction')
        axes[0].axhline(y=0.10, color='gray', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3)

    # Transplant replicates
    for td in trans_data:
        axes[1].plot(td['ticks'], td['lms'], alpha=0.6, linewidth=1)
    if trans_data:
        mean_lm = np.mean([td['lms'] for td in trans_data if len(td['lms']) == len(trans_data[0]['lms'])], axis=0) if HAS_NP else None
        if mean_lm is not None:
            axes[1].plot(trans_data[0]['ticks'], mean_lm, 'r-', linewidth=2, label='mean')
    axes[1].set_title('Transplant Replicates')
    axes[1].set_xlabel('Tick')
    axes[1].axhline(y=0.10, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Control replicates
    for cd in ctrl_data:
        axes[2].plot(cd['ticks'], cd['lms'], alpha=0.6, linewidth=1)
    if ctrl_data:
        mean_lm = np.mean([cd['lms'] for cd in ctrl_data if len(cd['lms']) == len(ctrl_data[0]['lms'])], axis=0) if HAS_NP else None
        if mean_lm is not None:
            axes[2].plot(ctrl_data[0]['ticks'], mean_lm, 'r-', linewidth=2, label='mean')
    axes[2].set_title('Control Replicates')
    axes[2].set_xlabel('Tick')
    axes[2].axhline(y=0.10, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'lm_timeseries.png'), dpi=150)
    plt.close(fig)

    # ── Plot 2: Comparison — mean L→M overlay ──
    fig, ax = plt.subplots(figsize=(10, 6))

    if trans_data:
        for td in trans_data:
            ax.plot(td['ticks'], td['lms'], 'r-', alpha=0.2, linewidth=0.8)
        # Mean
        if HAS_NP:
            max_len = min(len(td['lms']) for td in trans_data)
            all_lms = np.array([td['lms'][:max_len] for td in trans_data])
            mean_ticks = trans_data[0]['ticks'][:max_len]
            ax.plot(mean_ticks, all_lms.mean(axis=0), 'r-', linewidth=2, label='Transplant (mean)')

    if ctrl_data:
        for cd in ctrl_data:
            ax.plot(cd['ticks'], cd['lms'], 'b-', alpha=0.2, linewidth=0.8)
        if HAS_NP:
            max_len = min(len(cd['lms']) for cd in ctrl_data)
            all_lms = np.array([cd['lms'][:max_len] for cd in ctrl_data])
            mean_ticks = ctrl_data[0]['ticks'][:max_len]
            ax.plot(mean_ticks, all_lms.mean(axis=0), 'b-', linewidth=2, label='Control (mean)')

    ax.axhline(y=0.10, color='gray', linestyle='--', alpha=0.5, label='Emergence threshold')
    ax.set_xlabel('Tick')
    ax.set_ylabel('L→M fraction')
    ax.set_title('Latent Reactivation: Transplant vs Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'comparison_overlay.png'), dpi=150)
    plt.close(fig)

    # ── Plot 3: Peak L→M bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    trans_peaks = [td['peak_lm'] for td in trans_data]
    ctrl_peaks = [cd['peak_lm'] for cd in ctrl_data]
    labels = [f'T{i}' for i in range(len(trans_peaks))] + [f'C{i}' for i in range(len(ctrl_peaks))]
    values = trans_peaks + ctrl_peaks
    colors = ['salmon'] * len(trans_peaks) + ['steelblue'] * len(ctrl_peaks)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Peak L→M')
    ax.set_title('Peak L→M by Replicate (T=Transplant, C=Control)')
    ax.axhline(y=0.10, color='gray', linestyle='--', alpha=0.5)
    fig.savefig(os.path.join(output_dir, 'peak_lm_bars.png'), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


def generate_report(results, output_dir='analysis_output/reactivation'):
    """Generate an HTML report."""
    os.makedirs(output_dir, exist_ok=True)
    analyzed = analyze(results, verbose=False)
    trans_data = analyzed['trans_data']
    ctrl_data = analyzed['ctrl_data']
    donor = analyzed['donor']
    params = results.get('params', {})

    trans_peaks = [td['peak_lm'] for td in trans_data]
    ctrl_peaks = [cd['peak_lm'] for cd in ctrl_data]
    trans_finals = [td['final_lm'] for td in trans_data]
    ctrl_finals = [cd['final_lm'] for cd in ctrl_data]
    trans_emergence = [td['emergence'] for td in trans_data if td['emergence'] is not None]
    ctrl_emergence = [cd['emergence'] for cd in ctrl_data if cd['emergence'] is not None]

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Latent Reactivation Experiment</title>
<style>
body {{ font-family: 'Helvetica Neue', sans-serif; max-width: 900px; margin: 40px auto; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c3e50; }}
.box {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
.metric .label {{ font-size: 0.9em; color: #7f8c8d; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: center; }}
th {{ background: #3498db; color: white; }}
img {{ max-width: 100%; border-radius: 6px; margin: 10px 0; }}
.verdict {{ font-size: 1.2em; padding: 15px; border-radius: 6px; margin: 20px 0; }}
.supports {{ background: #d5f5e3; border-left: 4px solid #27ae60; }}
.refutes {{ background: #fadbd8; border-left: 4px solid #e74c3c; }}
.inconclusive {{ background: #fef9e7; border-left: 4px solid #f39c12; }}
</style>
</head>
<body>
<h1>🧬 Latent Reactivation Experiment</h1>

<div class="box">
<h2>Hypothesis</h2>
<p>Organisms from late-stage worlds retain LOAD→MOVE genomic patterns even after
execution has ceased (genome-execution decoupling). When transplanted into fresh
environments with high trace information, they should re-activate stigmergic
behavior faster than controls starting from scratch.</p>
</div>

<div class="box">
<h2>Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Environment</td><td>{params.get('env', 'N/A')}</td></tr>
<tr><td>Donor seed</td><td>{params.get('donor_seed', 'N/A')}</td></tr>
<tr><td>Donor ticks</td><td>{params.get('ticks_donor', 'N/A')}</td></tr>
<tr><td>Transplant ticks</td><td>{params.get('ticks_transplant', 'N/A')}</td></tr>
<tr><td>Replicates per condition</td><td>{params.get('n_replicates', 'N/A')}</td></tr>
</table>
</div>

<div class="box">
<h2>Donor Phase</h2>
<div class="metric"><div class="value">{donor.get('genomes_harvested', 0)}</div><div class="label">Genomes harvested</div></div>
<div class="metric"><div class="value">{donor.get('genomic_lm_fraction', 0):.3f}</div><div class="label">Genomic L→M</div></div>
<div class="metric"><div class="value">{donor.get('avg_genome_length', 0):.1f}</div><div class="label">Avg genome length</div></div>
</div>

<div class="box">
<h2>Results</h2>
<div class="metric"><div class="value">{sum(trans_peaks)/len(trans_peaks):.3f}</div><div class="label">Transplant mean peak L→M</div></div>
<div class="metric"><div class="value">{sum(ctrl_peaks)/len(ctrl_peaks):.3f}</div><div class="label">Control mean peak L→M</div></div>
<div class="metric"><div class="value">{len(trans_emergence)}/{len(trans_data)}</div><div class="label">Transplant emergence rate</div></div>
<div class="metric"><div class="value">{len(ctrl_emergence)}/{len(ctrl_data)}</div><div class="label">Control emergence rate</div></div>
</div>

<div class="box">
<h2>Per-Replicate Detail</h2>
<h3>Transplant</h3>
<table>
<tr><th>Rep</th><th>Emergence</th><th>Peak L→M</th><th>Final L→M</th></tr>
"""
    for td in trans_data:
        e = str(td['emergence']) if td['emergence'] is not None else '—'
        html += f"<tr><td>{td['idx']}</td><td>{e}</td><td>{td['peak_lm']:.3f}</td><td>{td['final_lm']:.3f}</td></tr>\n"

    html += """</table>
<h3>Control</h3>
<table>
<tr><th>Rep</th><th>Emergence</th><th>Peak L→M</th><th>Final L→M</th></tr>
"""
    for cd in ctrl_data:
        e = str(cd['emergence']) if cd['emergence'] is not None else '—'
        html += f"<tr><td>{cd['idx']}</td><td>{e}</td><td>{cd['peak_lm']:.3f}</td><td>{cd['final_lm']:.3f}</td></tr>\n"

    html += """</table>
</div>
"""

    # Images
    for img in ['lm_timeseries.png', 'comparison_overlay.png', 'peak_lm_bars.png']:
        path = os.path.join(output_dir, img)
        if os.path.exists(path):
            html += f'<div class="box"><h2>{img.replace("_"," ").replace(".png","").title()}</h2><img src="{img}"></div>\n'

    # Verdict
    trans_mean = sum(trans_peaks)/len(trans_peaks) if trans_peaks else 0
    ctrl_mean = sum(ctrl_peaks)/len(ctrl_peaks) if ctrl_peaks else 0
    if trans_mean > ctrl_mean * 1.5:
        verdict = "SUPPORTS — transplanted genomes re-activate stigmergy faster"
        css_class = "supports"
    elif ctrl_mean > trans_mean * 1.5:
        verdict = "REFUTES — controls develop stigmergy faster than transplants"
        css_class = "refutes"
    else:
        verdict = "INCONCLUSIVE — no clear difference between conditions"
        css_class = "inconclusive"
    html += f'<div class="verdict {css_class}"><strong>Verdict:</strong> {verdict}</div>\n'
    html += "</body></html>"

    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html)
    print(f"  Report saved to {output_dir}/report.html")
    return html


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    results_path = find_results(path)

    if results_path is None:
        print("No results files found")
        sys.exit(1)

    print(f"Loading: {results_path}")
    results = load_results(results_path)

    analyze(results)
    plot_results(results)
    generate_report(results)


if __name__ == '__main__':
    main()