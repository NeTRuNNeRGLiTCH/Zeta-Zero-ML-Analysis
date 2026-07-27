import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- 1. THE MODELS ---

class DualTensorSniper(nn.Module):
    """ The New 2-Tensor Idea """
    def __init__(self):
        super().__init__()
        # Tensor 1: The Router (100 -> 13)
        self.router = nn.Sequential(
            nn.Linear(100, 256),
            nn.GELU(),
            nn.Linear(256, 100 * 13) # Generates a projection matrix
        )
        # Tensor 2: The Sniper (13 -> 1)
        self.sniper = nn.LSTM(input_size=13, hidden_size=123, num_layers=3, batch_first=True)
        self.head = nn.Linear(123, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        # Tensor 1 creates the 'Filter Matrix'
        mask_matrix = self.router(x).view(batch_size, 100, 13)
        # Apply the filter: (B, 1, 100) @ (B, 100, 13) = (B, 1, 13)
        distilled = torch.bmm(x.unsqueeze(1), mask_matrix)
        # Tensor 2 processes the 13 distilled inputs
        out, _ = self.sniper(distilled)
        return self.head(out[:, -1, :])

class PrototypeA(nn.Module):
    """ The Baseline (100-input LSTM) """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=3, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. DATA PREP (Window Hugging Logic) ---

def prepare_comparison_data():
    print("Loading 2M Zeros and building the 'Hugging Window'...")
    zeros = np.loadtxt("zero", dtype=np.float64)
    gaps = np.diff(zeros)
    avg_gaps = (2 * np.pi) / np.log((zeros[:-1] / (2 * np.pi)) + 1e-12)
    multipliers = (gaps / avg_gaps).astype(np.float32)
    
    # 500k to 2M range
    start_idx = 500000
    end_idx = len(multipliers) - 1
    num_samples = end_idx - start_idx
    
    X = np.zeros((num_samples, 100), dtype=np.float32)
    Y = multipliers[start_idx : end_idx]
    
    for i in range(num_samples):
        n = start_idx + i
        x0 = n / 1000.0
        
        # 1. Simplified Navigator Formula
        dist = int(np.sqrt((x0 + 50) * np.sqrt(np.abs(np.cos(x0 + 45)))))
        
        # 2. Slice 1: Last 33 multipliers
        fixed_bank = multipliers[n-33 : n]
        
        # 3. Slice 2: Resonance Window (66 units wide)
        # We center the 66-window around distance 'dist'
        res_center = n - dist
        res_start = res_center - 33
        res_end = res_center + 33
        
        # Boundary safety: Ensure window doesn't overlap the fixed_bank or go negative
        if res_end > n - 34:
            res_end = n - 34
            res_start = res_end - 66
            
        res_bank = multipliers[res_start : res_end]
        
        # 4. Final Feature: Target Avg Gap
        avg_feat = avg_gaps[n]
        
        # Combine [33, 66, 1] = 100 inputs
        X[i] = np.concatenate([fixed_bank, res_bank, [avg_feat]])
        
    return torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(1)

# --- 3. TRAINING ENGINE ---

def train_model(model, X, Y, name):
    device = "cuda"
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    # Split 1M Train / ~500k Test
    X_train, Y_train = X[:1000000].to(device), Y[:1000000].to(device)
    X_test, Y_test = X[1000000:].to(device), Y[1000000:].to(device)
    
    print(f"\n🚀 Training {name}...")
    start = time.time()
    
    for epoch in range(1, 3001):
        model.train()
        optimizer.zero_grad()
        
        idx = torch.randint(0, len(X_train), (2048,))
        preds = model(X_train[idx])
        loss = criterion(preds, Y_train[idx])
        loss.backward()
        optimizer.step()
        
        if epoch % 250 == 0:
            print(f"  Epoch {epoch:4} | Loss: {loss.item():.6f}")
            
    # --- BATCHED INFERENCE TO PREVENT OOM ---
    model.eval()
    test_preds_list = []
    eval_batch_size = 10000 # Process 10k at a time instead of 500k
    
    print(f"  Evaluating {name} in chunks...")
    with torch.no_grad():
        for i in range(0, len(X_test), eval_batch_size):
            batch_x = X_test[i : i + eval_batch_size]
            batch_preds = model(batch_x)
            test_preds_list.append(batch_preds.cpu()) # Move to CPU to save VRAM
            
        # Reconstruct the full prediction tensor on CPU
        test_preds = torch.cat(test_preds_list, dim=0)
        
        # Calculate final MAE
        mae = torch.abs(test_preds - Y_test.cpu()).mean().item()
        base_mae = torch.abs(1.0 - Y_test.cpu()).mean().item()
        improvement = ((base_mae - mae) / base_mae) * 100
        
    print(f"✅ {name} Finished in {time.time()-start:.1f}s. Accuracy Improvement: {improvement:.2f}%")
    return improvement

if __name__ == "__main__":
    X, Y = prepare_comparison_data()
    
    # Run Prototype A
    imp_a = train_model(PrototypeA(), X, Y, "PROTOTYPE A (Baseline)")
    
    # Run Dual Tensor Sniper
    imp_s = train_model(DualTensorSniper(), X, Y, "DUAL-TENSOR SNIPER (The Idea)")
    
    print("\n" + "="*50)
    print("🏆 FINAL COMPARISON")
    print("="*50)
    print(f"Prototype A: {imp_a:.2f}%")
    print(f"Dual-Tensor Sniper: {imp_s:.2f}%")
    print(f"Net Gain from Selection Logic: {imp_s - imp_a:.2f}%")
    print("="*50)