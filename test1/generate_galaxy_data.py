import rebound
import numpy as np
import os

# --- CONFIGURATION ---
NUM_STARS_PER_GALAXY = 100  # Total 200 stars (Safe for 4GB VRAM training)
TOTAL_TIME = 3.0            # Simulation duration
DT = 0.01                   # Time step (how often we save data)
OUTPUT_FILE = "D:\AI_Physics\Dataset\cosmic_collison\galaxy_collision_data.npz"

def create_galaxy(sim, num_stars, center_pos, center_vel, size_scale):
    """
    Creates a mini-galaxy (plummer sphere) at a specific location.
    """
    # 1. Add a heavy "Black Hole" center to keep the galaxy together
    sim.add(m=10.0, x=center_pos[0], y=center_pos[1], z=center_pos[2],
            vx=center_vel[0], vy=center_vel[1], vz=center_vel[2])

    # 2. Add stars around it
    for _ in range(num_stars):
        # Random position in a sphere
        r = size_scale * np.random.rand()**(1/3) # Uniform sphere
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = 2 * np.pi * np.random.rand()
        
        # Relative coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Orbital velocity (approximate circular orbit around the black hole)
        v_orb = np.sqrt(sim.G * 10.0 / (r + 0.1)) # v = sqrt(GM/r)
        
        # Velocity vectors (perpendicular to radius)
        vx = -v_orb * np.sin(phi)
        vy = v_orb * np.cos(phi)
        vz = (np.random.rand() - 0.5) * 0.1 # Slight vertical jitter

        sim.add(m=0.01, 
                x=center_pos[0]+x, y=center_pos[1]+y, z=center_pos[2]+z,
                vx=center_vel[0]+vx, vy=center_vel[1]+vy, vz=center_vel[2]+vz)

def generate_data():
    print(f"🌌 Initializing Simulation with {NUM_STARS_PER_GALAXY * 2} particles...")
    
    sim = rebound.Simulation()
    sim.G = 1.0 # Use internal units
    
    # --- Create Galaxy 1 (Left, moving Right) ---
    create_galaxy(sim, NUM_STARS_PER_GALAXY, 
                  center_pos=[-3.0, -1.0, 0.0], 
                  center_vel=[0.5, 0.1, 0.0], 
                  size_scale=1.0)

    # --- Create Galaxy 2 (Right, moving Left) ---
    create_galaxy(sim, NUM_STARS_PER_GALAXY, 
                  center_pos=[3.0, 1.0, 0.0], 
                  center_vel=[-0.5, -0.1, 0.0], 
                  size_scale=1.0)

    # --- Run Simulation ---
    times = np.arange(0, TOTAL_TIME, DT)
    N = len(sim.particles)
    
    # Arrays to store data [TimeSteps, NumParticles, 3]
    positions = np.zeros((len(times), N, 3))
    velocities = np.zeros((len(times), N, 3))
    accelerations = np.zeros((len(times), N, 3))

    print("🚀 Running simulation... (This uses CPU, not GPU)")
    
    for i, t in enumerate(times):
        sim.integrate(t)
        
        # Extract data from Rebound
        for j, p in enumerate(sim.particles):
            positions[i, j] = [p.x, p.y, p.z]
            velocities[i, j] = [p.vx, p.vy, p.vz]
            accelerations[i, j] = [p.ax, p.ay, p.az]
            
        if i % 50 == 0:
            print(f"   Step {i}/{len(times)} completed")

    # --- Save to Disk ---
    print(f"💾 Saving data to {OUTPUT_FILE}...")
    np.savez(OUTPUT_FILE, 
             positions=positions, 
             velocities=velocities, 
             accelerations=accelerations, 
             times=times)
    print("✅ Done! Data generated successfully.")

if __name__ == "__main__":
    generate_data()