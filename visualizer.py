"""
SM-MCPM Visualizer
Real-time display and export functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os


# HEATMAP VISUALIZATIONS

def plot_death_map(death_map, core_pos=None, objective_pos=None, save_path=None):
    """Plot death density heatmap with log scale"""
    plt.figure(figsize=(12, 7))
    
    if death_map.max() > 0:
        # Log scale for better visibility
        death_vis = np.log1p(death_map)
        im = plt.imshow(death_vis.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(im, label='log(deaths + 1)')
    else:
        plt.imshow(death_map.T, origin='lower', cmap='hot')
        plt.colorbar(label='Deaths')
    
    plt.title(f'Death Density Map (Total: {death_map.sum():.0f} deaths)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Mark core and objective
    if core_pos is not None:
        plt.plot(core_pos[0], core_pos[1], 'c*', markersize=20, label='Core')
    if objective_pos is not None:
        plt.plot(objective_pos[0], objective_pos[1], 'y*', markersize=20, label='Objective')
    
    if core_pos is not None or objective_pos is not None:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved death map to {save_path}")
    
    plt.close()


def plot_trail_map(trail_map, save_path=None):
    """Plot accumulated trail density"""
    plt.figure(figsize=(12, 7))
    
    if trail_map.max() > 0:
        trail_vis = np.log1p(trail_map)
        im = plt.imshow(trail_vis.T, origin='lower', cmap='viridis', interpolation='bilinear')
        plt.colorbar(im, label='log(trail density)')
    else:
        plt.imshow(trail_map.T, origin='lower', cmap='viridis')
        plt.colorbar(label='Trail Density')
    
    plt.title('Trail History Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved trail map to {save_path}")
    
    plt.close()


def plot_vein_network(vein_map, nodes=None, core_pos=None, objective_pos=None, save_path=None):
    """Plot final vein network with nodes"""
    plt.figure(figsize=(12, 7))
    
    # Show veins
    plt.imshow(vein_map.T, origin='lower', cmap='Blues', interpolation='bilinear')
    plt.colorbar(label='Vein Strength')
    
    # Overlay nodes
    if nodes and len(nodes) > 0:
        node_x = [n['position'][0] for n in nodes]
        node_y = [n['position'][1] for n in nodes]
        plt.scatter(node_x, node_y, c='red', s=200, marker='o', 
                   edgecolors='white', linewidths=2, label='Nodes', zorder=5)
    
    # Core
    if core_pos is not None:
        plt.scatter(core_pos[0], core_pos[1], c='cyan', s=300, marker='*',
                   edgecolors='black', linewidths=2, label='Core', zorder=5)
    
    # Objective
    if objective_pos is not None:
        plt.scatter(objective_pos[0], objective_pos[1], c='gold', s=300, marker='*', 
                   edgecolors='black', linewidths=2, label='Objective', zorder=5)
    
    plt.title(f'Vein Network ({len(nodes) if nodes else 0} nodes)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved vein network to {save_path}")
    
    plt.close()


def plot_score_map(score_map, nodes=None, save_path=None):
    """Plot node placement score map"""
    plt.figure(figsize=(12, 7))
    
    im = plt.imshow(score_map.T, origin='lower', cmap='RdYlGn', interpolation='bilinear')
    plt.colorbar(im, label='Placement Score')
    
    # Overlay existing nodes
    if nodes and len(nodes) > 0:
        node_x = [n['position'][0] for n in nodes]
        node_y = [n['position'][1] for n in nodes]
        plt.scatter(node_x, node_y, c='blue', s=200, marker='x', 
                   linewidths=3, label='Existing Nodes', zorder=5)
    
    plt.title('Node Placement Score Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if nodes and len(nodes) > 0:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved score map to {save_path}")
    
    plt.close()


def visualize_environment(barrier_map, hazard_map, save_path=None):
    """Visualize loaded environment (for debugging)"""
    height, width = barrier_map.shape
    vis = np.zeros((height, width, 3))
    
    # Green = open space
    vis[:, :, 1] = 1.0
    
    # Blue = barriers
    vis[barrier_map > 0.5] = [0, 0, 1]
    
    # Red = hazards
    vis[hazard_map > 0.5] = [1, 0, 0]
    
    plt.figure(figsize=(12, 7))
    plt.imshow(vis.T, origin='lower')
    plt.title('Environment Map (Green=Open, Blue=Barrier, Red=Hazard)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved environment visualization to {save_path}")
    
    plt.close()


def plot_combined_summary(death_map, vein_map, nodes, core_pos, objective_pos, save_path=None):
    """Plot 2x2 summary of key results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Death map
    if death_map.max() > 0:
        death_vis = np.log1p(death_map)
        im0 = axes[0, 0].imshow(death_vis.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(im0, ax=axes[0, 0], label='log(deaths)')
    else:
        axes[0, 0].imshow(death_map.T, origin='lower', cmap='hot')
    axes[0, 0].set_title(f'Death Map ({death_map.sum():.0f} total)')
    axes[0, 0].plot(core_pos[0], core_pos[1], 'c*', markersize=15)
    axes[0, 0].plot(objective_pos[0], objective_pos[1], 'y*', markersize=15)
    
    # Vein network
    im1 = axes[0, 1].imshow(vein_map.T, origin='lower', cmap='Blues', interpolation='bilinear')
    plt.colorbar(im1, ax=axes[0, 1], label='Vein Strength')
    if nodes and len(nodes) > 0:
        node_x = [n['position'][0] for n in nodes]
        node_y = [n['position'][1] for n in nodes]
        axes[0, 1].scatter(node_x, node_y, c='red', s=150, marker='o', edgecolors='white', linewidths=2)
    axes[0, 1].plot(core_pos[0], core_pos[1], 'c*', markersize=15)
    axes[0, 1].plot(objective_pos[0], objective_pos[1], 'y*', markersize=15)
    axes[0, 1].set_title(f'Vein Network ({len(nodes) if nodes else 0} nodes)')
    
     #Combined overlay
    overlay = np.zeros((death_map.shape[0], death_map.shape[1], 3))
    if vein_map.max() > 0:
        vein_norm = vein_map / vein_map.max()
        overlay[:, :, 2] = vein_norm  # Blue channel for veins
    if death_map.max() > 0:
        death_norm = death_map / death_map.max()
        overlay[:, :, 0] = death_norm  # Red channel for deaths
    
    # Transpose correctly for matplotlib (height, width, channels)
    axes[1, 0].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[1, 0].set_title('Overlay (Red=Deaths, Blue=Veins)')
    if nodes and len(nodes) > 0:
        axes[1, 0].scatter(node_x, node_y, c='yellow', s=150, marker='o', edgecolors='white', linewidths=2)
    axes[1, 0].plot(core_pos[0], core_pos[1], 'c*', markersize=15)
    axes[1, 0].plot(objective_pos[0], objective_pos[1], 'y*', markersize=15)
    
    # Node progression
    if nodes and len(nodes) > 0:
        # Show node placement order
        axes[1, 1].imshow(vein_map.T, origin='lower', cmap='Greys', alpha=0.3)
        for i, node in enumerate(nodes):
            axes[1, 1].scatter(node['position'][0], node['position'][1], 
                             c='red', s=300, marker='o', edgecolors='white', linewidths=2)
            axes[1, 1].text(node['position'][0], node['position'][1], str(i+1), 
                          ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        axes[1, 1].plot(core_pos[0], core_pos[1], 'c*', markersize=15, label='Core')
        axes[1, 1].plot(objective_pos[0], objective_pos[1], 'y*', markersize=15, label='Objective')
        axes[1, 1].set_title('Node Placement Order')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No nodes placed', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=20)
        axes[1, 1].set_title('Node Placement')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved combined summary to {save_path}")
    
    plt.close()


# DATA EXPORT

def export_metrics_csv(metrics_history, save_path):
    """Export metrics to CSV"""
    import pandas as pd
    
    df = pd.DataFrame(metrics_history)
    df.to_csv(save_path, index=False)
    print(f" Exported metrics to {save_path}")


def export_final_state(results, save_dir):
    """Export all visualizations and data for paper"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nExporting results to {save_dir}/...")
    
    # Extract data
    phase1 = results.get('phase1', {})
    phase3 = results.get('phase3', {})
    nodes = results.get('nodes', [])
    config = results.get('config', {})
    
    core_pos = config.get('core', {}).get('position', [640, 360])
    obj_pos = config.get('objective', {}).get('position', [1000, 360])
    
    # Death map
    if 'death_map' in phase1:
        plot_death_map(phase1['death_map'], core_pos, obj_pos, 
                      f"{save_dir}/death_map.png")
    
    # Trail map
    if 'trail_map' in phase1:
        plot_trail_map(phase1['trail_map'], f"{save_dir}/trail_map.png")
    
    # Vein network
    if 'vein_map' in phase3:
        plot_vein_network(phase3['vein_map'], nodes, core_pos, obj_pos,
                         f"{save_dir}/vein_network.png")
    
    # Combined summary
    if 'death_map' in phase1 and 'vein_map' in phase3:
        plot_combined_summary(phase1['death_map'], phase3['vein_map'], 
                            nodes, core_pos, obj_pos,
                            f"{save_dir}/summary.png")
    
    # Metrics
    if 'metrics' in phase3:
        export_metrics_csv(phase3['metrics'], f"{save_dir}/metrics.csv")
    
    # Save node positions
    if nodes:
        import json
        with open(f"{save_dir}/nodes.json", 'w') as f:
            json.dump(nodes, f, indent=2)
        print(f" Saved node data to {save_dir}/nodes.json")
    
    print(f"\n All results exported to {save_dir}/")


# REAL-TIME DISPLAY (Optional - Taichi GUI)

class RealtimeDisplay:
    """Real-time visualization using Taichi GUI (optional)"""
    def __init__(self, simulation):
        self.sim = simulation
        # TODO: Implement Taichi GUI if needed
        print(" Real-time display not implemented yet")
    
    def update(self):
        pass
    
    def should_close(self):
        return True


if __name__ == "__main__":
    print(" visualizer.py ready")
    
    # Test with dummy data
    test_death = np.random.rand(1280, 720) * 100
    test_vein = np.random.rand(1280, 720)
    
    plot_death_map(test_death, [200, 360], [1000, 360], 'outputs/test_death.png')
    plot_vein_network(test_vein, [], [200, 360], [1000, 360], 'outputs/test_vein.png')
    
    print(" Test visualizations created")