import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
MODEL_FILE = "galaxy_net_torch.pth"
NUM_STARS = 200    # Total stars to simulate
DURATION = 200     # How many frames to generate
DT = 0.05          # Time step (speed of simulation)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. RE-DEFINE THE MODEL (Must match training exactly)
class LagrangianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state)

# 2. PHYSICS ENGINE (AI Version)
def get_ai_acceleration(model, positions, velocities):
    # Combine into state vector [x, y, z, vx, vy, vz]
    state = np.concatenate([positions, velocities], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
    state_tensor.requires_grad_(True)
    
    # Ask AI for Energy
    L = model(state_tensor).sum()
    
    # Ask AI for Force (Gradient of Energy)
    grads = torch.autograd.grad(L, state_tensor, create_graph=False)[0]
    force = grads[:, :3] # First 3 are dL/dq (Force-like)
    
    return force.detach().cpu().numpy()

# 3. INITIALIZE SIMULATION
def init_simulation():
    # Galaxy 1 (Left, moving right)
    pos1 = np.random.randn(NUM_STARS//2, 3) * 0.5 + np.array([-2, -0.5, 0])
    vel1 = np.random.randn(NUM_STARS//2, 3) * 0.1 + np.array([0.5, 0.1, 0])
    
    # Galaxy 2 (Right, moving left)
    pos2 = np.random.randn(NUM_STARS//2, 3) * 0.5 + np.array([2, 0.5, 0])
    vel2 = np.random.randn(NUM_STARS//2, 3) * 0.1 + np.array([-0.5, -0.1, 0])
    
    pos = np.concatenate([pos1, pos2])
    vel = np.concatenate([vel1, vel2])
    return pos, vel

# 4. RUN & ANIMATE
def main():
    print(f"🚀 Loading AI from {MODEL_FILE}...")
    model = LagrangianNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval() # Set to evaluation mode

    pos, vel = init_simulation()
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Remove axes for space look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')

    # Scatter plot particles
    # Galaxy 1 = Cyan, Galaxy 2 = Magenta
    colors = ['cyan'] * (NUM_STARS//2) + ['magenta'] * (NUM_STARS//2)
    scatter = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, s=10, alpha=0.8)

    print("🎥 Generating Animation (this may take a minute)...")

    def update(frame):
        nonlocal pos, vel
        
        # --- AI INTEGRATOR ---
        # 1. Ask AI for acceleration
        acc = get_ai_acceleration(model, pos, vel)
        
        # 2. Update Velocity (Euler Integration)
        vel += acc * DT
        
        # 3. Update Position
        pos += vel * DT
        
        # Update Plot
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        
        # Rotate camera for cool effect
        ax.view_init(elev=30, azim=frame * 0.5)
        
        if frame % 10 == 0:
            print(f"   Frame {frame}/{DURATION}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=DURATION, interval=30, blit=False)
    
    # Save as GIF
    ani.save('galaxy_ai_simulation.gif', writer='pillow', fps=30)
    print("✅ Saved 'galaxy_ai_simulation.gif'!")
    plt.show()

if __name__ == "__main__":
    main()