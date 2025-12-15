"""
SM-MCPM Controller
Orchestrates the three-phase algorithm and node placement logic
"""

import numpy as np
from sm_mcpm import Simulation
from scipy.ndimage import gaussian_filter


class Controller:
    def __init__(self, config):
        """Initialize controller with configuration"""
        self.config = config
        self.sim = Simulation(config)
        self.nodes = []
        self.all_metrics = []
        
        # Placement parameters
        self.death_weight = config['placement']['death_weight']
        self.trail_weight = config['placement']['trail_weight']
        self.proximity_penalty = config['placement']['proximity_penalty']
        self.min_score = config['placement']['min_score_threshold']
        self.min_node_distance = config['nodes']['min_node_distance']
        
        print(" Controller initialized")
    
    
    # PHASE 1: MONTE CARLO EXPLORATION
    
    
    def run_phase1(self, duration=None):
        """
        Phase 1: Monte Carlo exploration without nodes
        
        Args:
            duration: Simulation time in seconds (uses config if None)
        
        Returns:
            dict with death_map, trail_map, stats
        """
        if duration is None:
            duration = self.config['phases']['phase1_duration']
        
        print("PHASE 1: Monte Carlo Exploration")
        print(f"Duration: {duration}s simulated time")
        print(f"No nodes - pure exploration with energy constraints")
        print()
        
        # Calculate steps needed
        dt = self.config['simulation']['dt']
        total_steps = int(duration / dt)
        
        # Reset death map for clean start
        self.sim.death_map.fill(0)
        
        # Accumulate trail history
        trail_history = np.zeros((self.sim.width, self.sim.height), dtype=np.float32)
        
        # Run simulation
        print("Running exploration...")
        for step in range(total_steps):
            self.sim.step()
            
            # Accumulate trail history (time-integrated)
            trail_history += self.sim.get_trail_map() * dt
            
            # Progress updates
            if step % 1000 == 0:
                metrics = self.sim.get_metrics()
                print(f"  Step {step}/{total_steps}: "
                      f"{metrics['total_deaths']:.0f} deaths, "
                      f"{metrics['agents_alive']} alive, "
                      f"energy: {metrics['avg_energy']:.2f}, "
                      f"obj: {metrics.get('objective_arrivals', 0)}, "
                      f"trips: {metrics.get('round_trips', 0)}")
        
        # Get final data
        death_map = self.sim.get_death_map()
        final_metrics = self.sim.get_metrics()
        
        # Statistics
        stats = {
            'total_deaths': float(np.sum(death_map)),
            'max_death_density': float(np.max(death_map)),
            'exploration_area': int(np.count_nonzero(trail_history > 0.1)),
            'final_agents_alive': final_metrics['agents_alive'],
            'objective_arrivals': final_metrics.get('objective_arrivals', 0),
            'round_trips': final_metrics.get('round_trips', 0)
        }
        
        print(f" Phase 1 Complete:")
        print(f"  Total deaths: {stats['total_deaths']:.0f}")
        print(f"  Max death density: {stats['max_death_density']:.0f}")
        print(f"  Exploration area: {stats['exploration_area']} pixels")
        print(f"  Objective arrivals: {stats['objective_arrivals']}")
        print(f"  Successful round trips: {stats['round_trips']}")
        states = self.sim.get_agent_states()
        state_counts = {}
        for s in [-1, 0, 1, 2, 3, 4, 5]:
            count = np.sum(states == s)
            state_counts[s] = count
        
        print("\nAgent state distribution:")
        print(f"  DEAD: {state_counts[-1]}")
        print(f"  IDLE: {state_counts[0]}")
        print(f"  SCOUTING: {state_counts[1]}")
        print(f"  AT_OBJECTIVE: {state_counts[2]}")
        print(f"  LOW_ENERGY: {state_counts[3]}")
        print(f"  RECHARGING: {state_counts[4]}")
        print(f"  RETURNING: {state_counts[5]}")
        
        return {
            'death_map': death_map,
            'trail_map': trail_history,
            'stats': stats
        }
    

    # PHASE 2: NODE PLACEMENT

    
    def run_phase2(self, max_nodes=None, initial_death_map=None, initial_trail_map=None):
        """
        Phase 2: Iterative node placement based on death gradient
        
        Args:
            max_nodes: Maximum nodes to place (uses config if None)
            initial_death_map: Death map from Phase 1 (runs Phase 1 if None)
            initial_trail_map: Trail map from Phase 1
        
        Returns:
            list of placed nodes
        """
        if max_nodes is None:
            max_nodes = self.config['nodes']['max_nodes']
        
        
        print("PHASE 2: Node Placement")
        print(f"Max nodes: {max_nodes}")
        print(f"Placement strategy: {self.death_weight:.0%} death + {self.trail_weight:.0%} trail")
        print()
        
        # Get initial data if not provided
        if initial_death_map is None or initial_trail_map is None:
            print(" No Phase 1 data provided, running Phase 1 first...")
            phase1_results = self.run_phase1()
            death_map = phase1_results['death_map'].copy()
            trail_map = phase1_results['trail_map'].copy()
        else:
            death_map = initial_death_map.copy()
            trail_map = initial_trail_map.copy()
        
        # Iterative placement
        iteration_duration = self.config['phases']['phase2_iteration_duration']
        dt = self.config['simulation']['dt']
        iteration_steps = int(iteration_duration / dt)
        
        for node_idx in range(max_nodes):
            print(f"\n--- Iteration {node_idx + 1}/{max_nodes} ---")
            
            # Compute placement score
            score_map = self.compute_placement_score(death_map, trail_map)
            
            # Find best location
            node_pos = self.place_next_node(score_map)
            
            if node_pos is None:
                print(" No good location found (score below threshold)")
                print(f" Stopping after {len(self.nodes)} nodes")
                break
            
            # Add node to simulation
            success = self.sim.add_node(node_pos)
            if not success:
                print(" Failed to add node")
                break
            
            self.nodes.append({
                'position': node_pos,
                'iteration': node_idx,
                'radius': self.config['nodes']['radius']
            })
            
            # Run simulation with new node
            print(f"  Running simulation with {len(self.nodes)} node(s)...")
            new_death_map = np.zeros_like(death_map)
            new_trail_map = np.zeros_like(trail_map)
            
            for step in range(iteration_steps):
                self.sim.step()
                new_trail_map += self.sim.get_trail_map() * dt
                
                if step % 500 == 0 and step > 0:
                    metrics = self.sim.get_metrics()
                    print(f"    Step {step}/{iteration_steps}: "
                          f"{metrics['agents_alive']} alive, "
                          f"energy: {metrics['avg_energy']:.2f}")
            
            # Get updated death data (only new deaths this iteration)
            current_total_deaths = self.sim.get_death_map()
            new_death_map = current_total_deaths - death_map
            
            # Update cumulative maps
            death_map = current_total_deaths.copy()
            trail_map = new_trail_map
            
            # Check stopping criteria
            survival_rate = self._estimate_survival_rate()
            print(f"  Est. survival rate: {survival_rate:.1%}")
            
            if survival_rate > self.config['stopping']['target_survival_rate']:
                print(f" Target survival rate reached!")
                print(f" Stopping after {len(self.nodes)} nodes")
                break
        states = self.sim.get_agent_states()
        state_counts = {}
        for s in [-1, 0, 1, 2, 3, 4, 5]:
            count = np.sum(states == s)
            state_counts[s] = count
        
        print("\nAgent state distribution:")
        print(f"  DEAD: {state_counts[-1]}")
        print(f"  IDLE: {state_counts[0]}")
        print(f"  SCOUTING: {state_counts[1]}")
        print(f"  AT_OBJECTIVE: {state_counts[2]}")
        print(f"  LOW_ENERGY: {state_counts[3]}")
        print(f"  RECHARGING: {state_counts[4]}")
        print(f"  RETURNING: {state_counts[5]}")
        
        print(f"\n Phase 2 Complete: {len(self.nodes)} nodes placed")
        return self.nodes
    
    def compute_placement_score(self, death_map, trail_map):
        """
        Compute weighted score for node placement
        
        Returns:
            score_map: Higher values = better placement locations
        """
        # Use log normalization for death map (handles sparse data better)
        death_log = np.log1p(death_map)  # log(1 + x) to handle zeros
        D = death_log / (death_log.max() + 1e-6)
        
        # Normalize trail map
        T = trail_map / (trail_map.max() + 1e-6)
        
        # Smooth both maps for stability
        D = gaussian_filter(D, sigma=10.0)  # Larger sigma for broader death zones
        T = gaussian_filter(T, sigma=5.0)
        
        # Compute proximity cost (penalize placing near existing infrastructure)
        C = self._compute_proximity_cost()
        
        # Add distance-to-objective weighting: prefer nodes TOWARD objective
        obj_x, obj_y = self.sim.objective_pos[0], self.sim.objective_pos[1]
        core_x, core_y = self.sim.core_pos[0], self.sim.core_pos[1]
        
        x = np.arange(self.sim.width)
        y = np.arange(self.sim.height)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Distance from core (normalized)
        dist_from_core = np.sqrt((xx - core_x)**2 + (yy - core_y)**2)
        max_dist = np.sqrt((obj_x - core_x)**2 + (obj_y - core_y)**2)
        
        # Bonus for being toward objective (but not too far - use bell curve)
        # Peak at ~40% of distance to objective
        optimal_dist = max_dist * 0.4
        dist_bonus = np.exp(-((dist_from_core - optimal_dist) / (max_dist * 0.3))**2)
        
        # Weighted combination
        score = (self.death_weight * D + 
                 self.trail_weight * T + 
                 0.3 * dist_bonus -  # Bonus for being in right direction
                 self.proximity_penalty * C)
        
        # Apply hard constraints
        score = self._apply_constraints(score)
        
        # Debug output
        print(f"  Score components:")
        print(f"    Death (D): min={D.min():.4f}, max={D.max():.4f}")
        print(f"    Trail (T): min={T.min():.4f}, max={T.max():.4f}")
        print(f"    Proximity (C): max penalty={C.max():.4f}")
        print(f"    Distance bonus: max={dist_bonus.max():.4f}")
        
        return score
    
    def _compute_proximity_cost(self):
        """Compute cost map penalizing proximity to existing nodes/core"""
        cost = np.zeros((self.sim.width, self.sim.height), dtype=np.float32)
        
        # Create coordinate grids
        x = np.arange(self.sim.width)
        y = np.arange(self.sim.height)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Cost from core
        core_x, core_y = self.sim.core_pos[0], self.sim.core_pos[1]
        dist_to_core = np.sqrt((xx - core_x)**2 + (yy - core_y)**2)
        cost = np.exp(-dist_to_core / self.min_node_distance)
        
        # Cost from existing nodes
        for node in self.nodes:
            node_x, node_y = node['position']
            dist_to_node = np.sqrt((xx - node_x)**2 + (yy - node_y)**2)
            cost += np.exp(-dist_to_node / self.min_node_distance)
        
        return cost
    
    def _apply_constraints(self, score):
        """Apply hard constraints (barriers, bounds, min distance)"""
        score_constrained = score.copy()
        
        # Zero out barriers
        barrier_map = self.sim.barrier_map.to_numpy()
        score_constrained[barrier_map > 0.5] = 0
        
        # Zero out hazards
        hazard_map = self.sim.hazard_map.to_numpy()
        score_constrained[hazard_map > 0.5] = 0
        
        # Zero out too close to existing nodes
        for node in self.nodes:
            x, y = int(node['position'][0]), int(node['position'][1])
            x_min = max(0, x - int(self.min_node_distance))
            x_max = min(self.sim.width, x + int(self.min_node_distance))
            y_min = max(0, y - int(self.min_node_distance))
            y_max = min(self.sim.height, y + int(self.min_node_distance))
            score_constrained[x_min:x_max, y_min:y_max] = 0
        
        # Zero out too close to core
        core_x, core_y = int(self.sim.core_pos[0]), int(self.sim.core_pos[1])
        x_min = max(0, core_x - int(self.min_node_distance))
        x_max = min(self.sim.width, core_x + int(self.min_node_distance))
        y_min = max(0, core_y - int(self.min_node_distance))
        y_max = min(self.sim.height, core_y + int(self.min_node_distance))
        score_constrained[x_min:x_max, y_min:y_max] = 0
        
        return score_constrained
    
    def place_next_node(self, score_map):
        """
        Find best location and return position
        
        Returns:
            [x, y] position or None if no good location
        """
        # Find argmax
        max_score = np.max(score_map)
        
        # DEBUG OUTPUT
        print(f"  Score map stats:")
        print(f"    Max score: {max_score:.4f}")
        print(f"    Min score: {np.min(score_map):.4f}")
        print(f"    Mean score: {np.mean(score_map):.4f}")
        print(f"    Non-zero pixels: {np.count_nonzero(score_map)}")
        print(f"    Threshold: {self.min_score}")
        
        if max_score < self.min_score:
            print(f"   Max score {max_score:.4f} below threshold {self.min_score}")
            return None
        
        # Get position of max score
        max_idx = np.argmax(score_map)
        x, y = np.unravel_index(max_idx, score_map.shape)
        
        print(f"   Best location: ({x}, {y}) with score {max_score:.3f}")
        
        return [float(x), float(y)]
    
    def _estimate_survival_rate(self):
        """Estimate what % of agents are completing round trips"""
        metrics = self.sim.get_metrics()
        
        # Use actual tracking data
        survival_rate = metrics.get('survival_rate', 0.0)
        
        # Also print detailed stats for debugging
        arrivals = metrics.get('objective_arrivals', 0)
        round_trips = metrics.get('round_trips', 0)
        total_spawns = metrics.get('total_spawns', 0)
        
        if total_spawns > 100:  # Only print once we have meaningful data
            print(f"    → Objective arrivals: {arrivals}, Round trips: {round_trips}, Spawns: {total_spawns}")
        
        return survival_rate
    
    
 
    # PHASE 3: STABILIZATION
   
    
    def run_phase3(self, duration=None):
        """
        Phase 3: Let network stabilize and measure performance
        
        Returns:
            dict with final metrics, vein_map, etc.
        """
        if duration is None:
            duration = self.config['phases']['phase3_duration']
        
        print("PHASE 3: Network Stabilization")
        print(f"Duration: {duration}s simulated time")
        print(f"Nodes in place: {len(self.nodes)}")
        print()
        
        dt = self.config['simulation']['dt']
        total_steps = int(duration / dt)
        
        # Track metrics over time
        metrics_history = []
        
        print("Stabilizing network...")
        for step in range(total_steps):
            self.sim.step()
            
            # Record metrics periodically
            if step % 500 == 0:
                metrics = self.sim.get_metrics()
                metrics_history.append(metrics)
                
                print(f"  Step {step}/{total_steps}: "
                      f"{metrics['agents_alive']} alive, "
                      f"energy: {metrics['avg_energy']:.2f}, "
                      f"vein coverage: {metrics['vein_coverage']:.1%}")
        
        # Final state
        final_metrics = self.sim.get_metrics()
        vein_map = self.sim.get_vein_map()
        
        print(f"\n Phase 3 Complete:")
        print(f"  Final agents alive: {final_metrics['agents_alive']}")
        print(f"  Final avg energy: {final_metrics['avg_energy']:.2f}")
        print(f"  Vein coverage: {final_metrics['vein_coverage']:.1%}")
        states = self.sim.get_agent_states()
        state_counts = {}
        for s in [-1, 0, 1, 2, 3, 4, 5]:
            count = np.sum(states == s)
            state_counts[s] = count
        
        print("\nAgent state distribution:")
        print(f"  DEAD: {state_counts[-1]}")
        print(f"  IDLE: {state_counts[0]}")
        print(f"  SCOUTING: {state_counts[1]}")
        print(f"  AT_OBJECTIVE: {state_counts[2]}")
        print(f"  LOW_ENERGY: {state_counts[3]}")
        print(f"  RECHARGING: {state_counts[4]}")
        print(f"  RETURNING: {state_counts[5]}")
        
        return {
            'metrics': metrics_history,
            'final_metrics': final_metrics,
            'vein_map': vein_map
        }
    
    # FULL PIPELINE
    
    def run_full_algorithm(self):
        """
        Run complete SM-MCPM algorithm (all 3 phases)
        
        Returns:
            complete results dictionary
        """
        print(" "*10 + "SM-MCPM: State-Modulated Monte Carlo Physarum Machine")
        
        # Phase 1: Exploration
        phase1_results = self.run_phase1()
        
        # Phase 2: Node placement
        nodes = self.run_phase2(
            initial_death_map=phase1_results['death_map'],
            initial_trail_map=phase1_results['trail_map']
        )
        
        # Phase 3: Stabilization
        phase3_results = self.run_phase3()
        
        # Combine results
        results = {
            'phase1': phase1_results,
            'nodes': nodes,
            'phase3': phase3_results,
            'config': self.config
        }
        
        print(" ALGORITHM COMPLETE")
        print(f"  Nodes placed: {len(nodes)}")
        print(f"  Total deaths: {phase1_results['stats']['total_deaths']:.0f}")
        print(f"  Final vein coverage: {phase3_results['final_metrics']['vein_coverage']:.1%}")
        
        return results


if __name__ == "__main__":
    from config import load_config
    
    print("Testing controller...")
    config = load_config('configs/default.yaml')
    controller = Controller(config)
    
    # Quick test of Phase 1
    results = controller.run_phase1(duration=50.0)  # Short test
    print(f"\n controller.py ready!")
    print(f"  Test deaths: {results['stats']['total_deaths']:.0f}")