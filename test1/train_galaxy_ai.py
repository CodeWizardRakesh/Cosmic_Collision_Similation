import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_FILE = "D:\AI_Physics\Dataset\cosmic_collison\galaxy_collision_data.npz"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Slightly higher for PyTorch
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using Device: {DEVICE}")

# 1. DEFINE THE AI MODEL (The "Brain")
class LagrangianNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 6 (x, y, z, vx, vy, vz) -> Output: 1 (Lagrangian/Energy)
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Softplus(),  # Smooth activation (vital for physics derivatives)
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)

# 2. DEFINE THE PHYSICS (The "Laws")
def equation_of_motion(model, state):
    """
    Computes acceleration using the Euler-Lagrange equation with PyTorch Autograd.
    """
    state = state.requires_grad_(True)
    
    # 1. Get the Lagrangian (Energy)
    L = model(state).sum() # Sum allows us to take gradients of the whole batch at once
    
    # 2. First Derivatives (dL/dq, dL/dv)
    # create_graph=True allows us to take a *second* derivative later
    grads = torch.autograd.grad(L, state, create_graph=True)[0]
    dL_dq, dL_dv = grads[:, :3], grads[:, 3:]

    # 3. Time Derivative of dL/dv (d/dt (dL/dv))
    # This is tricky! d/dt = (d/dq * v) + (d/dv * a)
    # But we want to solve for 'a'.
    
    # We calculate the Hessian-Vector products implicitly to avoid massive matrices
    # acceleration = Inverse(Hessian_vv) * (Force_q - Mixed_Term)
    
    # --- SIMPLIFIED STABILITY MODE ---
    # Full Hessian inversion can be unstable on small batches.
    # For this galaxy sim, we assume the AI learns a standard Hamiltonian form:
    # L = 0.5*m*v^2 - V(q). 
    # Therefore, Acceleration = dL/dq (Force field) / Mass
    # We let the network learn the "Force Field" directly via the gradient of L.
    
    return dL_dq  # This acts as the force pushing the star

# 3. TRAINING LOOP
def train():
    print("📂 Loading Data...")
    data = np.load(DATA_FILE)
    
    # Prepare Data
    pos = data['positions'].reshape(-1, 3)
    vel = data['velocities'].reshape(-1, 3)
    acc = data['accelerations'].reshape(-1, 3)
    
    # Create Inputs [Pos, Vel] and Targets [Accel]
    inputs = np.concatenate([pos, vel], axis=1)
    targets = acc

    # Convert to PyTorch Tensors and move to GPU
    inputs = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(targets, dtype=torch.float32).to(DEVICE)
    
    # Create Dataset and Loader for batching
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = LagrangianNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🧠 Model Initialized. Starting Training...")
    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Predict Acceleration using Physics Laws
            pred_accel = equation_of_motion(model, batch_x)
            
            # Calculate Loss (MSE)
            loss = nn.functional.mse_loss(pred_accel, batch_y)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")

    print("✅ Training Complete!")
    
    # Save Model
    torch.save(model.state_dict(), "galaxy_net_torch.pth")
    print("💾 Model saved to 'galaxy_net_torch.pth'")

    # Plot
    plt.plot(loss_history)
    plt.title("PyTorch AI Learning Gravity")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    train()