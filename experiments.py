import os
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sm_mcpm import Simulation 

# 1. CONFIGURATION & SETUP

def load_experiment_config(ablation=False):
    """Loads config and sets up nodes dynamically."""
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Dynamic Node Placement (Linear Interpolation)
    core_pos = np.array(config['core']['position'])
    obj_pos = np.array(config['objective']['position'])
    
    nodes = []
    for i in range(1, 5):
        t = i / 5.0
        pos = core_pos + (obj_pos - core_pos) * t
        nodes.append(pos.tolist())
    
    config['experiment_nodes'] = nodes

    if ablation:
        config['energy']['low_energy_threshold'] = -1.0 
        config['objective']['time_at_objective'] = 999999.0

    return config

# 2. ENHANCED PLOTTING FUNCTIONS

def plot_simulation_state(sim_data, config, title, filename, show_deaths=False):
    """Generates a high-quality figure with infrastructure overlays."""
    vein_map = sim_data['final_vein'] if 'final_vein' in sim_data else sim_data
    death_map = sim_data.get('final_death') if isinstance(sim_data, dict) else None
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    
    # A. Plot Veins (Black veins on White background)
    vein_masked = np.ma.masked_where(vein_map < 0.05, vein_map)
    ax.imshow(vein_masked.T, origin='lower', cmap='Greys', vmin=0, vmax=1, alpha=0.8, zorder=1)
    
    # B. Plot Deaths (Optional Heatmap Overlay)
    if show_deaths and death_map is not None:
        death_log = np.log1p(death_map)
        death_masked = np.ma.masked_where(death_log < 0.1, death_log)
        ax.imshow(death_masked.T, origin='lower', cmap='Reds', alpha=0.6, zorder=2)

    # C. Plot Infrastructure
    c_pos = config['core']['position']
    ax.scatter(c_pos[0], c_pos[1], c='cyan', s=300, marker='*', edgecolors='black', linewidth=1.5, label='Core', zorder=10)
    
    o_pos = config['objective']['position']
    ax.scatter(o_pos[0], o_pos[1], c='gold', s=300, marker='*', edgecolors='black', linewidth=1.5, label='Objective', zorder=10)
    
    if 'experiment_nodes' in config:
        node_radius = config['nodes']['radius']
        for i, n_pos in enumerate(config['experiment_nodes']):
            ax.scatter(n_pos[0], n_pos[1], c='#ff4444', s=100, marker='o', edgecolors='black', zorder=10)
            circ = patches.Circle((n_pos[0], n_pos[1]), node_radius, linewidth=1, edgecolor='#ff4444', facecolor='none', linestyle='--', alpha=0.6, zorder=5)
            ax.add_patch(circ)

    # Formatting
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(0, config['simulation']['width'])
    ax.set_ylim(0, config['simulation']['height'])
    ax.set_aspect('equal')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', markeredgecolor='k', markersize=15, label='Core'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='k', markersize=15, label='Objective'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', markeredgecolor='k', markersize=10, label='Recharge Node'),
    ]
    if show_deaths:
        legend_elements.append(patches.Patch(facecolor='red', edgecolor='none', alpha=0.6, label='Death Density'))
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure: {filename}")

# 3. EXPERIMENT RUNNER

def run_single_trial(cfg, duration, use_nodes, snapshots=None):
    """Runs one simulation instance."""
    sim = Simulation(cfg)
    
    if use_nodes:
        for pos in cfg['experiment_nodes']:
            sim.add_node(pos)
            
    dt = cfg['simulation']['dt']
    total_steps = int(duration / dt)
    captured_snapshots = {}
    
    for step in range(total_steps):
        sim.step()
        
        # Snapshot logic
        if snapshots:
            curr_time = step * dt
            for t_snap in snapshots:
                if abs(curr_time - t_snap) < dt/2:
                    v_map = sim.get_vein_map()
                    if isinstance(v_map, np.ndarray):
                        captured_snapshots[t_snap] = v_map.copy()
    
    # Force capture final frame if missed
    if snapshots and snapshots[-1] not in captured_snapshots:
        captured_snapshots[snapshots[-1]] = sim.get_vein_map().copy()

    metrics = sim.get_metrics()
    metrics['snapshots'] = captured_snapshots
    metrics['final_vein'] = sim.get_vein_map()
    metrics['final_death'] = sim.get_death_map()
    return metrics

def run_multi_trial_condition(name, duration, use_nodes=True, ablation=False, trials=10, snapshots=None):
    """Runs N trials and aggregates statistics."""
    print(f"\n>>> Running Multi-Trial: {name} (N={trials})")
    print(f"    Nodes: {use_nodes}, Ablation: {ablation}, Duration: {duration}s")
    
    cfg = load_experiment_config(ablation=ablation)
    
    # Storage for stats
    stats = {
        'arrivals': [],
        'trips': [],
        'deaths': []
    }
    
    last_result = None # Keep the last one for plotting
    
    start_total = time.time()
    
    for i in range(trials):
        print(f"    Trial {i+1}/{trials}...", end="\r")
        res = run_single_trial(cfg, duration, use_nodes, snapshots)
        
        stats['arrivals'].append(res['objective_arrivals'])
        stats['trips'].append(res['round_trips'])
        stats['deaths'].append(res['total_deaths'])
        
        last_result = res
        last_result['config'] = cfg # Attach config for plotting
        
    print(f"     All trials done in {time.time() - start_total:.2f}s")
    
    # Compute Aggregates
    summary = {
        'arrivals_mean': np.mean(stats['arrivals']),
        'arrivals_std': np.std(stats['arrivals']),
        'trips_mean': np.mean(stats['trips']),
        'trips_std': np.std(stats['trips']),
        'deaths_mean': np.mean(stats['deaths']),
        'deaths_std': np.std(stats['deaths']),
        'last_run': last_result # For plotting
    }
    
    return summary

# 4. MAIN EXECUTION

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    TRIALS = 10
    
    print("="*60 + "\nGENERATING  RESULTS (N=10)\n" + "="*60)

    # A. Figure 2: Vein Evolution (Snapshots from Full System)
    print("\n[ 1. Full System Analysis (Table 5 + Figure 2) ]")
    # Changed 500 -> 498 to be safe on timing
    res_full = run_multi_trial_condition(
        "Full System", 500, use_nodes=True, ablation=False, 
        trials=TRIALS, snapshots=[50, 200, 500]
    )
    
    # Plot snapshots from the LAST run (representative)
    print("\n[ Plotting Figure 2 ]")
    snapshots = res_full['last_run']['snapshots']
    config = res_full['last_run']['config']
    
    for t in [50, 200, 500]:
        if t in snapshots:
            plot_simulation_state(
                snapshots[t], config, 
                f"Vein Evolution (t={t}s)", f"outputs/fig2_t{t}.png"
            )

    # Plot Full Summary
    plot_simulation_state(
        res_full['last_run'], config,
        "Full SM-MCPM System: Vein Network & Infrastructure", 
        "outputs/fig_summary_full.png"
    )

    # B. Ablation Study (Table 5)
    print("\n[ 2. Ablation Study (Table 5 Comparison) ]")
    res_ablated = run_multi_trial_condition(
        "Ablated (No State Mod)", 500, use_nodes=True, ablation=True, trials=TRIALS
    )
    
    # Plot Ablated Summary
    plot_simulation_state(
        res_ablated['last_run'], res_ablated['last_run']['config'],
        "Ablation Failure: Death Density (No State Modulation)", 
        "outputs/fig_summary_ablated.png",
        show_deaths=True
    )

    # C. Infrastructure Necessity (Table 4)
    print("\n[ 3. Infrastructure Study (Table 4) ]")
    
    res_no_nodes = run_multi_trial_condition("No Nodes", 300, use_nodes=False, trials=TRIALS)
    res_with_nodes = run_multi_trial_condition("4 Nodes", 300, use_nodes=True, trials=TRIALS)

    # D. FINAL STATISTICAL TABLES
    print(f"FINAL STATISTICAL RESULTS (Mean ± Std Dev over {TRIALS} runs)")
    
    print("\n>>> TABLE 4: Infrastructure Necessity (300s)")
    print(f"{'Condition':<15} | {'Obj Arrivals':<20} | {'Round Trips':<20}")
    print(f"{'No Nodes':<15} | {res_no_nodes['arrivals_mean']:.1f} ± {res_no_nodes['arrivals_std']:.1f} | {res_no_nodes['trips_mean']:.1f} ± {res_no_nodes['trips_std']:.1f}")
    print(f"{'4 Nodes':<15} | {res_with_nodes['arrivals_mean']:.1f} ± {res_with_nodes['arrivals_std']:.1f} | {res_with_nodes['trips_mean']:.1f} ± {res_with_nodes['trips_std']:.1f}")

    print("\n>>> TABLE 5: Ablation Study (500s)")
    print(f"{'Condition':<15} | {'Total Deaths':<20} | {'Round Trips':<20}")
    print(f"{'Full System':<15} | {res_full['deaths_mean']:.1f} ± {res_full['deaths_std']:.1f} | {res_full['trips_mean']:.1f} ± {res_full['trips_std']:.1f}")
    print(f"{'Ablated':<15} | {res_ablated['deaths_mean']:.1f} ± {res_ablated['deaths_std']:.1f} | {res_ablated['trips_mean']:.1f} ± {res_ablated['trips_std']:.1f}")
    
    print("\n Done. Figures saved to 'outputs/'.")