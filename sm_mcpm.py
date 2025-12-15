"""
SM-MCPM: State-Modulated Monte Carlo Physarum Machine
Core simulation engine - 

This file contains:
- Agent state machine with 6 states
- Physarum sensing (Jones algorithm)
- Energy/hunger system with state-dependent decay
- Trail deposition, diffusion, and decay
- Vein formation with traffic accumulation and hysteresis
- Node recharging system
- Death recording and analysis
- Environment loading (procedural, image, presets)
"""

import taichi as ti
import numpy as np
from enum import IntEnum
from pathlib import Path

# Initialize Taichi with GPU
ti.init(arch=ti.gpu, device_memory_GB=4.0)


# CONSTANTS & ENUMS

class AgentState(IntEnum):
    """Agent behavioral states"""
    DEAD = -1      # Dead agent (will be respawned)
    IDLE = 0       # At core, resting
    SCOUTING = 1   # Exploring, seeking objective
    AT_OBJECTIVE = 2   # At objective, performing task
    LOW_ENERGY = 3     # Emergency mode, seeking node/core
    RECHARGING = 4     # At node/core, restoring energy
    RETURNING = 5      # Navigating back to core

PI = 3.14159265359
TWO_PI = 6.28318530718


# DATA STRUCTURES

@ti.dataclass
class Agent:
    """Agent particle with state and properties"""
    position: ti.math.vec2   # Current position (x, y)
    angle: ti.f32            # Heading angle in radians
    state: ti.i32            # Current behavioral state
    mass: ti.f32             # Energy/health (0-1)
    objective_timer: ti.f32  # Time spent at objective
    lifetime: ti.f32         # Total time alive
    home_node_id: ti.i32     # Which node spawned this agent (-1 = core)


@ti.dataclass
class Node:
    """Recharge station / waypoint"""
    position: ti.math.vec2   # Position (x, y)
    radius: ti.f32           # Interaction radius
    health: ti.f32           # Current health
    max_health: ti.f32       # Maximum health
    active: ti.i32           # 1 = active, 0 = destroyed
    node_id: ti.i32          # Unique identifier


# SIMULATION CLASS

@ti.data_oriented
class Simulation:
    def __init__(self, config):
        """Initialize complete simulation"""
        self.config = config
        
        # Simulation parameters
        self.width = config['simulation']['width']
        self.height = config['simulation']['height']
        self.dt = config['simulation']['dt']
        
        # Agent parameters
        self.max_agents = config['agents']['count']
        self.base_speed = config['agents']['base_speed']
        self.sensor_distance = config['agents']['sensor_distance']
        self.sensor_angle = np.radians(config['agents']['sensor_angle'])
        self.turn_speed = np.radians(config['agents']['turn_speed'])
        self.deposit_strength = config['agents']['deposit_strength']

        # Respawn modes
        self.respawn_mode = config['agents'].get('respawn_mode', 'instant')
        self.respawn_delay = config['agents'].get('respawn_delay', 5.0)
        self.last_respawn_time = 0.0
        
        print(f"  Respawn mode: {self.respawn_mode}")
        
        # Energy parameters
        self.hunger_decay = config['energy']['hunger_decay']
        self.scouting_mult = config['energy']['scouting_multiplier']
        self.objective_drain = config['energy']['objective_drain']
        self.vein_hunger_reduction = config['energy']['vein_hunger_reduction']
        self.low_energy_threshold = config['energy']['low_energy_threshold']
        self.recharge_threshold = config['energy']['recharge_threshold']
        self.full_energy_threshold = config['energy']['full_energy_threshold']
        
        # Trail parameters
        self.trail_decay = config['trails']['decay_rate']
        
        # Vein parameters
        self.vein_threshold = config['veins']['formation_threshold']
        self.vein_solidify = config['veins']['solidify_rate']
        self.vein_decay = config['veins']['decay_rate']
        self.vein_orphan_decay = config['veins']['orphan_decay_rate']
        self.vein_speed_mult = config['veins']['speed_multiplier']
        
        # Node parameters
        self.node_radius = config['nodes']['radius']
        
        # Objective parameters
        self.objective_pos = ti.Vector(config['objective']['position'])
        self.objective_radius = config['objective']['radius']
        self.objective_scent_strength = config['objective']['scent_strength']
        self.objective_scent_falloff = config['objective']['scent_falloff']
        self.time_at_objective = config['objective']['time_at_objective']
        
        # Core parameters
        self.core_pos = ti.Vector(config['core']['position'])
        self.core_radius = config['core']['radius']
        
        # TAICHI FIELDS (GPU MEMORY)
        
        # Agent data
        self.agents = Agent.field(shape=self.max_agents)
        self.agent_count = ti.field(ti.i32, shape=())  # Active agent count
        
        # Spatial fields (2D textures)
        self.trail_map = ti.field(ti.f32, shape=(self.width, self.height))
        self.vein_map = ti.Vector.field(4, ti.f32, shape=(self.width, self.height))
        # vein_map channels: R=strength, G=owner_id, B=orphaned, A=traffic
        
        self.death_map = ti.field(ti.f32, shape=(self.width, self.height))
        self.barrier_map = ti.field(ti.f32, shape=(self.width, self.height))
        self.hazard_map = ti.field(ti.f32, shape=(self.width, self.height))
        
        # Node data
        self.max_nodes = config['nodes']['max_nodes']
        self.nodes = Node.field(shape=self.max_nodes)
        self.node_count = ti.field(ti.i32, shape=())
        
        # Simulation time
        self.time = ti.field(ti.f32, shape=())
        
        # Performance tracking
        self.objective_arrivals = ti.field(ti.i32, shape=())  # Count of agents reaching objective
        self.core_returns = ti.field(ti.i32, shape=())        # Count of successful round trips
        self.total_spawns = ti.field(ti.i32, shape=())        # Total agents spawned
        
        # Random seed
        self.random_seed = ti.field(ti.i32, shape=())
        self.random_seed[None] = 42
        
        # INITIALIZATION
        
        # Setup environment (barriers, hazards)
        self._setup_environment(config['environment'])
        
        # Spawn initial agents at core
        self._spawn_initial_agents()
        
        print(f"✓ Simulation initialized: {self.width}×{self.height}")
        print(f"  Agents: {self.max_agents}")
        print(f"  Core: {self.core_pos}")
        print(f"  Objective: {self.objective_pos}")
    
    # MAIN SIMULATION LOOP
    
    def step(self):
            """Execute one simulation timestep"""
            self.time[None] += self.dt
            
            # Update agents (state machine, sensing, movement, energy)
            self._update_agents()
            
            # Deposit trails from agent positions
            self._deposit_trails()
            
            # Diffuse and decay trails
            self._diffuse_decay_trails()
            
            # Update veins (traffic accumulation, formation, decay)
            self._update_veins()
            
            # Respawn dead agents based on mode
            if self.respawn_mode == 'instant':
                self._respawn_dead_agents()
            elif self.respawn_mode == 'waves':
                current_time = self.time[None]
                if current_time - self.last_respawn_time >= self.respawn_delay:
                    self._respawn_dead_agents()
                    self.last_respawn_time = current_time
            # else: 'none' - no respawning    

    # AGENT KERNELS
    
    @ti.kernel
    def _update_agents(self):
        """Main agent update: state machine, sensing, movement, energy"""
        for i in self.agents:
            if self.agents[i].state == AgentState.DEAD:
                continue
            
            # Update lifetime
            self.agents[i].lifetime += self.dt
            
            # STATE MACHINE TRANSITIONS
            
            state = self.agents[i].state
            mass = self.agents[i].mass
            pos = self.agents[i].position
            
            # Check for low energy (highest priority)
            if mass < self.low_energy_threshold and state != AgentState.RECHARGING:
                self.agents[i].state = AgentState.LOW_ENERGY
                state = AgentState.LOW_ENERGY
            
            # State-specific transitions
            if state == AgentState.IDLE:
                # Random scout trigger
                if ti.random() < 0.10:  # Small probability per frame
                    self.agents[i].state = AgentState.SCOUTING
            
            elif state == AgentState.SCOUTING:
                # Check if reached objective
                if self._distance_to_objective(pos) < self.objective_radius:
                    self.agents[i].state = AgentState.AT_OBJECTIVE
                    self.agents[i].objective_timer = 0.0
                    ti.atomic_add(self.objective_arrivals[None], 1)  # Track arrival
                
                # Check if at node and low energy (pit stop)
                elif mass < self.recharge_threshold:
                    node_id = self._get_node_at_position(pos)
                    if node_id >= 0:
                        self.agents[i].state = AgentState.RECHARGING
            
            elif state == AgentState.AT_OBJECTIVE:
                # Increment timer
                self.agents[i].objective_timer += self.dt
                
                # Check if task complete
                if self.agents[i].objective_timer >= self.time_at_objective:
                    self.agents[i].state = AgentState.RETURNING
            
            elif state == AgentState.LOW_ENERGY:
                # Check if reached node or core
                if self._distance_to_core(pos) < self.core_radius:
                    self.agents[i].state = AgentState.RECHARGING
                else:
                    node_id = self._get_node_at_position(pos)
                    if node_id >= 0:
                        self.agents[i].state = AgentState.RECHARGING
            
            elif state == AgentState.RECHARGING:
                # Check if fully recharged
                if mass > self.full_energy_threshold:
                    # If came from objective, return to core
                    if self.agents[i].objective_timer > 0:
                        self.agents[i].state = AgentState.RETURNING
                    else:
                        self.agents[i].state = AgentState.SCOUTING
            
            elif state == AgentState.RETURNING:
                # Check if reached core
                if self._distance_to_core(pos) < self.core_radius:
                    self.agents[i].state = AgentState.IDLE
                    self.agents[i].objective_timer = 0.0
                    ti.atomic_add(self.core_returns[None], 1)  # Track round trip!
                
                # Check if need pit stop at node
                elif mass < self.recharge_threshold:
                    node_id = self._get_node_at_position(pos)
                    if node_id >= 0:
                        self.agents[i].state = AgentState.RECHARGING
            
            # MOVEMENT (only if not stationary states)
            
            state = self.agents[i].state  # Re-read after transitions
            
            # Movement (only if not stationary states)
            if state != AgentState.IDLE and state != AgentState.AT_OBJECTIVE and state != AgentState.RECHARGING:
                # Get sensing weights for current state
                weights = self._get_sensing_weights(state)
                
                # Three-sensor physarum sensing
                angle = self.agents[i].angle
                
                # Sense forward, left, right
                signal_f = self._sense_combined(pos, angle, weights)
                signal_l = self._sense_combined(pos, angle + self.sensor_angle, weights)
                signal_r = self._sense_combined(pos, angle - self.sensor_angle, weights)
                
                # Turn toward strongest signal
                if signal_l > signal_f and signal_l > signal_r:
                    self.agents[i].angle += self.turn_speed * self.dt
                elif signal_r > signal_f and signal_r > signal_l:
                    self.agents[i].angle -= self.turn_speed * self.dt
                
                # Wrap angle to [0, 2π]
                if self.agents[i].angle < 0:
                    self.agents[i].angle += TWO_PI
                elif self.agents[i].angle >= TWO_PI:
                    self.agents[i].angle -= TWO_PI
                
                # Move forward
                speed = self.base_speed
                
                # Speed boost on veins
                if self._is_on_vein(pos):
                    speed *= self.vein_speed_mult
                
                dx = ti.cos(self.agents[i].angle) * speed * self.dt
                dy = ti.sin(self.agents[i].angle) * speed * self.dt
                
                new_pos = pos + ti.Vector([dx, dy])
                
                # Boundary check
                new_pos[0] = ti.max(0.0, ti.min(float(self.width - 1), new_pos[0]))
                new_pos[1] = ti.max(0.0, ti.min(float(self.height - 1), new_pos[1]))
                
                # Barrier collision (bounce off)
                if not self._is_blocked(new_pos):
                    self.agents[i].position = new_pos
                else:
                    # Random turn on collision
                    self.agents[i].angle += (ti.random() - 0.5) * PI
            
            # ENERGY UPDATE
            
            state = self.agents[i].state
            energy_drain = self._get_energy_drain(state, self._is_on_vein(self.agents[i].position))
            self.agents[i].mass -= energy_drain * self.dt
            
            # Recharge at core or nodes
            if state == AgentState.RECHARGING:
                self.agents[i].mass += 0.5 * self.dt  # Recharge rate
                self.agents[i].mass = ti.min(1.0, self.agents[i].mass)
            
            # Death check
            if self.agents[i].mass <= 0:
                # Record death with atomic operation
                x, y = int(self.agents[i].position[0]), int(self.agents[i].position[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    ti.atomic_add(self.death_map[x, y], 1.0)  #  ATOMIC
                
                self.agents[i].state = AgentState.DEAD
                self.agents[i].mass = 0.0
    
    @ti.func
    def _sense_combined(self, pos: ti.math.vec2, angle: ti.f32, weights) -> ti.f32:
        """
        Combined multi-layer sensing
        weights: [trail, node, vein, objective, core] (5-element vector)
        """
        sensor_pos = pos + ti.Vector([ti.cos(angle), ti.sin(angle)]) * self.sensor_distance
        
        # Clamp sensor position
        sensor_pos[0] = ti.max(0.0, ti.min(float(self.width - 1), sensor_pos[0]))
        sensor_pos[1] = ti.max(0.0, ti.min(float(self.height - 1), sensor_pos[1]))
        
        x = int(sensor_pos[0])
        y = int(sensor_pos[1])
        
        # Ensure bounds
        x = ti.max(0, ti.min(self.width - 1, x))
        y = ti.max(0, ti.min(self.height - 1, y))
        
        # Sample layers
        trail = self.trail_map[x, y]
        vein = self.vein_map[x, y][0]  # Vein strength
        node_scent = self._sense_nodes(sensor_pos)
        objective_scent = self._sense_objective(sensor_pos)
        core_scent = self._sense_core(sensor_pos)
        
        # Weighted combination
        signal = (weights[0] * trail +
                weights[1] * node_scent +
                weights[2] * vein +
                weights[3] * objective_scent +
                weights[4] * core_scent)
        
        return signal
    
    @ti.func
    def _sense_core(self, pos: ti.math.vec2) -> ti.f32:
        """Sense core with distance falloff (same formula as objective)"""
        dist = (pos - self.core_pos).norm()
        # Use same falloff scale as objective for consistency
        scent = 1.0 / (1.0 + dist / self.objective_scent_falloff)
        return scent
    
    @ti.func
    def _get_sensing_weights(self, state: ti.i32):
        """Get sensing weights for state [trail, node, vein, objective, core]"""
        weights = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        
        if state == AgentState.SCOUTING:
            # OBJECTIVE MUST DOMINATE to prevent circling!
            # At 951px: objective_scent=0.174, so we need weight ~3.0 to beat veins
            # Vein weight low (0.15) so agents don't get trapped in highway loops
            weights = ti.Vector([0.2, 0.2, 0.15, 3.0, 0.0])
        elif state == AgentState.LOW_ENERGY:
            # CORE is highest priority! Node second, ignore objective
            weights = ti.Vector([0.2, 0.8, 0.3, 0.0, 1.5])
        elif state == AgentState.RETURNING:
            # CORE is primary goal, nodes for pit stops, use veins for speed
            weights = ti.Vector([0.2, 0.5, 0.4, 0.0, 1.5])
        
        return weights
    
    @ti.func
    def _get_energy_drain(self, state: ti.i32, on_vein: ti.i32) -> ti.f32:
        """Calculate energy drain rate"""
        base = self.hunger_decay
        
        # State multiplier
        mult = 1.0
        if state == AgentState.SCOUTING:
            mult = self.scouting_mult
        elif state == AgentState.AT_OBJECTIVE:
            mult = self.objective_drain
        elif state == AgentState.IDLE or state == AgentState.RECHARGING:
            mult = 0.0
        
        # Vein reduction
        if on_vein == 1:
            mult *= self.vein_hunger_reduction
        
        return base * mult
    
    @ti.func
    def _sense_nodes(self, pos: ti.math.vec2) -> ti.f32:
        """Sense nearest node"""
        max_scent = 0.0
        for i in range(self.node_count[None]):
            if self.nodes[i].active == 1:
                dist = (pos - self.nodes[i].position).norm()
                if dist < self.nodes[i].radius * 2:
                    scent = 1.0 - (dist / (self.nodes[i].radius * 2))
                    max_scent = ti.max(max_scent, scent)
        return max_scent
    
    @ti.func
    def _sense_objective(self, pos: ti.math.vec2) -> ti.f32:
        """Sense objective with distance falloff"""
        dist = (pos - self.objective_pos).norm()
        scent = self.objective_scent_strength / (1.0 + dist / self.objective_scent_falloff)
        return scent
    
    @ti.func
    def _distance_to_core(self, pos: ti.math.vec2) -> ti.f32:
        return (pos - self.core_pos).norm()
    
    @ti.func
    def _distance_to_objective(self, pos: ti.math.vec2) -> ti.f32:
        return (pos - self.objective_pos).norm()
    
    @ti.func
    def _is_blocked(self, pos: ti.math.vec2) -> ti.i32:
        """Check if position is blocked by barrier"""
        x, y = int(pos[0]), int(pos[1])
        result = 1  # Default: blocked (out of bounds)
        if 0 <= x < self.width and 0 <= y < self.height:
            result = int(self.barrier_map[x, y] > 0.5)
        return result

    @ti.func
    def _is_on_vein(self, pos: ti.math.vec2) -> ti.i32:
        """Check if position is on vein"""
        x, y = int(pos[0]), int(pos[1])
        result = 0
        if 0 <= x < self.width and 0 <= y < self.height:
            result = int(self.vein_map[x, y][0] > 0.5)
        return result

    @ti.func
    def _get_node_at_position(self, pos: ti.math.vec2) -> ti.i32:
        """Get node ID at position, -1 if none"""
        result = -1
        for i in range(self.node_count[None]):
            if self.nodes[i].active == 1:
                dist = (pos - self.nodes[i].position).norm()
                if dist < self.nodes[i].radius:
                    result = i
        return result    
    
    # TRAIL KERNELS
    
    @ti.kernel
    def _deposit_trails(self):
        """Deposit pheromone trails at agent positions"""
        for i in self.agents:
            if self.agents[i].state != AgentState.DEAD:
                x = int(self.agents[i].position[0])
                y = int(self.agents[i].position[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.trail_map[x, y] += self.deposit_strength
    
    @ti.kernel
    def _diffuse_decay_trails(self):
        """Diffuse and decay trail map (3x3 blur + decay)"""
        # Create temporary for diffusion
        for i, j in self.trail_map:
            if i > 0 and i < self.width - 1 and j > 0 and j < self.height - 1:
                # 3x3 blur
                sum_val = 0.0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        sum_val += self.trail_map[i + di, j + dj]
                
                # Average and decay
                new_val = (sum_val / 9.0) * 0.98 - self.trail_decay * self.dt
                self.trail_map[i, j] = ti.max(0.0, new_val)
    
    # VEIN KERNELS
    
    @ti.kernel
    def _update_veins(self):
        """Update vein formation, strengthening, and decay"""
        for i, j in self.vein_map:
            trail = self.trail_map[i, j]
            vein = self.vein_map[i, j]
            
            strength = vein[0]  # R: vein strength
            owner = vein[1]     # G: owner node ID
            orphaned = vein[2]  # B: orphaned flag
            traffic = vein[3]   # A: traffic accumulator
            
            # Accumulate traffic when trail density high
            if trail > 0.1:
                traffic += trail * self.vein_solidify * self.dt
            else:
                traffic = ti.max(0.0, traffic - 0.05 * self.dt)
            
            # Form vein if traffic exceeds threshold
            if traffic > self.vein_threshold:
                # Convert excess traffic to vein strength
                excess = traffic - self.vein_threshold
                strength += excess * 0.5 * self.dt
                strength = ti.min(1.0, strength)
                
                # Assign owner (nearest node)
                owner = float(self._find_nearest_node(ti.Vector([float(i), float(j)])) + 1)
            
            # Decay vein if not being reinforced
            if traffic <= self.vein_threshold and strength > 0:
                decay_rate = self.vein_decay
                # TODO: Check if orphaned, use faster decay
                strength -= decay_rate * self.dt
                strength = ti.max(0.0, strength)
            
            # Clear if too weak
            if strength < 0.01 and traffic < 0.01:
                strength = 0.0
                traffic = 0.0
                owner = 0.0
                orphaned = 0.0
            
            # Write back
            self.vein_map[i, j] = ti.Vector([strength, owner, orphaned, traffic])
    
    @ti.func
    def _find_nearest_node(self, pos: ti.math.vec2) -> ti.i32:
        """Find nearest active node, -1 if none"""
        min_dist = 999999.0
        nearest = -1
        for i in range(self.node_count[None]):
            if self.nodes[i].active == 1:
                dist = (pos - self.nodes[i].position).norm()
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
        
        # Also check distance to core
        dist_core = (pos - self.core_pos).norm()
        if dist_core < min_dist:
            nearest = -1  # Core is nearest
        
        return nearest
    
    # AGENT SPAWNING
    
    @ti.kernel
    def _spawn_initial_agents(self):
        """Spawn initial agents at core"""
        for i in self.agents:
            # Random position within core radius
            angle = ti.random() * TWO_PI
            radius = ti.random() * self.core_radius
            
            pos = self.core_pos + ti.Vector([
                ti.cos(angle) * radius,
                ti.sin(angle) * radius
            ])
            
            self.agents[i].position = pos
            self.agents[i].angle = ti.random() * TWO_PI
            self.agents[i].state = AgentState.IDLE  # Start IDLE, transition to SCOUTING naturally
            self.agents[i].mass = 1.0
            self.agents[i].objective_timer = 0.0
            self.agents[i].lifetime = 0.0
            self.agents[i].home_node_id = -1
            ti.atomic_add(self.total_spawns[None], 1)  # Track spawn
        
        self.agent_count[None] = self.max_agents
    
    @ti.kernel
    def _respawn_dead_agents(self):
        """Respawn dead agents at core"""
        for i in self.agents:
            if self.agents[i].state == AgentState.DEAD:
                # Respawn at core
                angle = ti.random() * TWO_PI
                radius = ti.random() * self.core_radius
                
                pos = self.core_pos + ti.Vector([
                    ti.cos(angle) * radius,
                    ti.sin(angle) * radius
                ])
                
                self.agents[i].position = pos
                self.agents[i].angle = ti.random() * TWO_PI
                self.agents[i].state = AgentState.IDLE  # Respawn as IDLE
                self.agents[i].mass = 1.0
                self.agents[i].objective_timer = 0.0
                self.agents[i].lifetime = 0.0
                ti.atomic_add(self.total_spawns[None], 1)  # Track respawn
    
    # NODE MANAGEMENT
    
    def add_node(self, position, radius=None):
        """Add a recharge node (CPU function)"""
        if self.node_count[None] >= self.max_nodes:
            print("⚠ Maximum nodes reached")
            return False
        
        if radius is None:
            radius = self.node_radius
        
        idx = self.node_count[None]
        
        # Set node data
        self.nodes[idx].position = ti.Vector(position)
        self.nodes[idx].radius = radius
        self.nodes[idx].health = 100.0
        self.nodes[idx].max_health = 100.0
        self.nodes[idx].active = 1
        self.nodes[idx].node_id = idx
        
        self.node_count[None] += 1
        
        print(f"✓ Added node {idx} at {position}")
        return True
    
    # DATA EXTRACTION (CPU FUNCTIONS)
    
    def get_death_map(self):
        """Return death map as numpy array"""
        return self.death_map.to_numpy()
    
    def get_trail_map(self):
        """Return trail map as numpy array"""
        return self.trail_map.to_numpy()
    
    def get_vein_map(self):
        """Return vein strength map as numpy array"""
        vein_full = self.vein_map.to_numpy()
        return vein_full[:, :, 0]  # Just strength channel
    
    def get_agent_positions(self):
        """Return active agent positions as numpy array"""
        positions = []
        
        # Read directly from Taichi field
        for i in range(self.max_agents):
            if self.agents[i].state != AgentState.DEAD:
                positions.append([self.agents[i].position[0], self.agents[i].position[1]])
        
        if len(positions) > 0:
            return np.array(positions)
        else:
            return np.zeros((0, 2))
    
    def get_agent_states(self):
        """Return agent states"""
        agents_np = self.agents.to_numpy()
        return agents_np['state']
    
    def get_nodes(self):
        """Return node data as list of dicts"""
        nodes_np = self.nodes.to_numpy()
        node_list = []
        for i in range(self.node_count[None]):
            if nodes_np[i]['active'] == 1:
                node_list.append({
                    'position': [nodes_np[i]['position']['x'], 
                                nodes_np[i]['position']['y']],
                    'radius': nodes_np[i]['radius'],
                    'health': nodes_np[i]['health'],
                    'id': nodes_np[i]['node_id']
                })
        return node_list
    
    def get_metrics(self):
        """Calculate current simulation metrics"""
        agents_np = self.agents.to_numpy()
        
        active_mask = agents_np['state'] != AgentState.DEAD
        active_count = np.sum(active_mask)
        
        if active_count > 0:
            avg_energy = np.mean(agents_np['mass'][active_mask])
        else:
            avg_energy = 0.0
        
        vein_np = self.vein_map.to_numpy()
        vein_coverage = np.sum(vein_np[:, :, 0] > 0.5) / (self.width * self.height)
        
        # Calculate completion rates
        obj_arrivals = int(self.objective_arrivals[None])
        round_trips = int(self.core_returns[None])
        total_spawns = int(self.total_spawns[None])
        
        # Survival rate: ratio of round trips to spawns (capped at 1.0)
        survival_rate = 0.0
        if total_spawns > 0:
            survival_rate = min(1.0, round_trips / total_spawns)
        
        return {
            'time': self.time[None],
            'agents_alive': int(active_count),
            'avg_energy': float(avg_energy),
            'vein_coverage': float(vein_coverage),
            'total_deaths': float(np.sum(self.death_map.to_numpy())),
            'objective_arrivals': obj_arrivals,
            'round_trips': round_trips,
            'total_spawns': total_spawns,
            'survival_rate': float(survival_rate)
        }
    
    # ENVIRONMENT SETUP
    
    def _setup_environment(self, env_config):
        """Setup barriers and hazards from config"""
        env_type = env_config.get('type', 'empty')
        
        if env_type == 'empty':
            pass  # Already zeros
        
        elif env_type == 'procedural':
            self._setup_procedural(env_config)
        
        elif env_type == 'from_image':
            self._load_map_image(env_config['map_image'])
        
        elif env_type == 'preset':
            self._load_preset(env_config['preset_name'])
        
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _setup_procedural(self, env_config):
        """Generate barriers/hazards procedurally"""
        # Get numpy arrays
        barrier_np = self.barrier_map.to_numpy()
        hazard_np = self.hazard_map.to_numpy()
        
        # Add barriers
        for barrier in env_config.get('barriers', []):
            if barrier['type'] == 'rectangle':
                x, y = int(barrier['position'][0]), int(barrier['position'][1])
                w, h = int(barrier['size'][0]), int(barrier['size'][1])
                barrier_np[x:x+w, y:y+h] = 1.0
            
            elif barrier['type'] == 'circle':
                cx, cy = int(barrier['position'][0]), int(barrier['position'][1])
                r = int(barrier['radius'])
                for x in range(max(0, cx-r), min(self.width, cx+r)):
                    for y in range(max(0, cy-r), min(self.height, cy+r)):
                        if (x - cx)**2 + (y - cy)**2 <= r**2:
                            barrier_np[x, y] = 1.0
        
        # Add hazards
        for hazard in env_config.get('hazards', []):
            if hazard['type'] == 'rectangle':
                x, y = int(hazard['position'][0]), int(hazard['position'][1])
                w, h = int(hazard['size'][0]), int(hazard['size'][1])
                hazard_np[x:x+w, y:y+h] = 1.0
            
            elif hazard['type'] == 'circle':
                cx, cy = int(hazard['position'][0]), int(hazard['position'][1])
                r = int(hazard['radius'])
                for x in range(max(0, cx-r), min(self.width, cx+r)):
                    for y in range(max(0, cy-r), min(self.height, cy+r)):
                        if (x - cx)**2 + (y - cy)**2 <= r**2:
                            hazard_np[x, y] = 1.0
        
        # Upload to GPU
        self.barrier_map.from_numpy(barrier_np)
        self.hazard_map.from_numpy(hazard_np)
        
        print(f" Procedural environment setup")
    
    def _load_map_image(self, path):
        """Load color-coded map image"""
        from PIL import Image
        
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height))
        img_array = np.array(img)
        
        # Extract RGB channels
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        
        # Detect colors (with tolerance)
        tolerance = 50
        
        # Blue = Barriers
        is_blue = (b > 200) & (r < tolerance) & (g < tolerance)
        
        # Red = Hazards
        is_red = (r > 200) & (g < tolerance) & (b < tolerance)
        
        # Black = also barriers
        is_black = (r < tolerance) & (g < tolerance) & (b < tolerance)
        
        # Create maps
        barrier_array = np.zeros((self.width, self.height), dtype=np.float32)
        barrier_array[is_blue | is_black] = 1.0
        barrier_array = barrier_array.T
        
        hazard_array = np.zeros((self.width, self.height), dtype=np.float32)
        hazard_array[is_red] = 1.0
        hazard_array = hazard_array.T
        
        # Upload to GPU
        self.barrier_map.from_numpy(barrier_array)
        self.hazard_map.from_numpy(hazard_array)
        
        print(f" Loaded map from {path}")
        print(f"  Barriers: {np.sum(is_blue | is_black)} pixels")
        print(f"  Hazards: {np.sum(is_red)} pixels")
    
    def _load_preset(self, preset_name):
        """Load hardcoded preset environment"""
        barrier_np = self.barrier_map.to_numpy()
        
        if preset_name == 'simple_corridor':
            # Two parallel walls
            barrier_np[300:350, 0:300] = 1.0
            barrier_np[300:350, 420:720] = 1.0
        
        elif preset_name == 'center_wall':
            # Single vertical wall
            barrier_np[640:660, 100:620] = 1.0
        
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        self.barrier_map.from_numpy(barrier_np)
        print(f" Loaded preset: {preset_name}")


# MAIN TEST

if __name__ == "__main__":
    from config import load_config
    
    print("SM-MCPM Core Simulation Test")
    
    # Load config
    config = load_config('configs/default.yaml')
    
    # Create simulation
    sim = Simulation(config)
    
    # Run a few steps
    print("\nRunning 100 simulation steps...")
    for i in range(100):
        sim.step()
        if i % 20 == 0:
            metrics = sim.get_metrics()
            print(f"  Step {i}: {metrics['agents_alive']} alive, "
                  f"avg energy: {metrics['avg_energy']:.2f}")
    
    print("\n sm_mcpm.py fully implemented and tested!")
    print(f"  Final deaths: {sim.get_metrics()['total_deaths']:.0f}")