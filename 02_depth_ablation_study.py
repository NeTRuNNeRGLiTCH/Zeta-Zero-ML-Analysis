import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import csv
import os
from concurrent.futures import ProcessPoolExecutor

# --- 1. THE 2-TENSOR ARCHITECTURE (Prototype B/C) ---
class DualTensorModel(nn.Module):
    def __init__(self, t1_dim, t2_dim):
        super(DualTensorModel, self).__init__()
        # Tensor 1: The Distiller / Router
        self.router = nn.Linear(t1_dim, t2_dim)
        self.gelu = nn.GELU()
        
        # Tensor 2: The Predictor
        # Input = t2_dim (distilled history) + 1 (target average gap)
        self.lstm = nn.LSTM(input_size=t2_dim + 1, hidden_size=64, num_layers=3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_hist, x_avg):
        # x_hist: (Batch, t1_dim)
        distilled = self.gelu(self.router(x_hist)) # Squash to t2_dim
        
        # Concatenate distilled history with the theoretical average gap
        combined = torch.cat([distilled, x_avg], dim=1) # (Batch, t2_dim + 1)
        
        # Feed to LSTM
        combined = combined.unsqueeze(1)
        out, _ = self.lstm(combined)
        return self.fc(out[:, -1, :])

# --- 2. THE WORKER NODE ---
def run_model_grid(start_k, t1_count, t2_count, multipliers, avg_gaps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    w_size = 100000
    start_idx = start_k * 1000
    end_idx = start_idx + w_size
    
    # --- WINDOW CALCULATION ---
    L1 = int(0.2 * t1_count) # 20% immediate past
    L2 = t1_count - L1       # 80% around K(t)
    
    # Calculate K(t) for the middle of this specific slice
    mid_k = start_k + 50
    k_t = int(np.sqrt((mid_k + 50) * np.sqrt(np.abs(np.cos(mid_k + 45)))))
    
    # Ensure the K(t) window doesn't crash into the L1 window or go negative
    res_center = max(k_t, L2 // 2 + L1 + 1)
    res_start = res_center + (L2 // 2)
    res_end = res_center - (L2 // 2)
    
    # Safe start index to ensure we have enough history
    actual_start = max(start_idx, res_start + 1)
    if actual_start >= end_idx:
        return None # Skip if mathematically impossible
        
    X_hist_list, X_avg_list, Y_list = [], [], []
    
    for i in range(actual_start, end_idx):
        # 80% Deep History
        bank_80 = multipliers[i - res_start : i - res_end]
        # 20% Immediate History
        bank_20 = multipliers[i - L1 : i]
        
        # Handle rounding issues in integer division to guarantee exact t1_count
        combined_hist = np.concatenate([bank_80, bank_20])[-t1_count:]
        
        X_hist_list.append(combined_hist)
        X_avg_list.append([avg_gaps[i]])
        Y_list.append(multipliers[i])
        
    X_h = torch.tensor(np.array(X_hist_list), dtype=torch.float32)
    X_a = torch.tensor(np.array(X_avg_list), dtype=torch.float32)
    Y_t = torch.tensor(np.array(Y_list), dtype=torch.float32).unsqueeze(1)
    
    # Split 90/10
    split = int(len(X_h) * 0.9)
    Xh_train, Xa_train, Y_train = X_h[:split].to(device), X_a[:split].to(device), Y_t[:split].to(device)
    Xh_test, Xa_test, Y_test = X_h[split:].to(device), X_a[split:].to(device), Y_t[split:].to(device)
    
    # Initialize
    torch.manual_seed(42)
    model = DualTensorModel(t1_dim=t1_count, t2_dim=t2_count).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.HuberLoss()
    
    # Train
    model.train()
    for _ in range(1500):
        optimizer.zero_grad()
        idx = torch.randint(0, len(Xh_train), (512,))
        preds = model(Xh_train[idx], Xa_train[idx])
        loss = criterion(preds, Y_train[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(Xh_test, Xa_test)
        ai_error = torch.abs(test_preds - Y_test).mean().item()
        base_error = torch.abs(torch.ones_like(Y_test) - Y_test).mean().item()
        improvement = ((base_error - ai_error) / base_error) * 100
        
    # --- GHOST EXTRACTION (LINEAR SURROGATE) ---
    Xh_np = Xh_test.cpu().numpy()
    Xa_np = Xa_test.cpu().numpy()
    Y_np = test_preds.cpu().numpy()
    
    # Matrix A includes History, Avg Gap, and Bias
    A = np.c_[Xh_np, Xa_np, np.ones(Xh_np.shape[0])]
    weights, _, _, _ = np.linalg.lstsq(A, Y_np, rcond=None)
    w_flat = weights.flatten()
    
    history_weights = np.abs(w_flat[:t1_count])
    top_4_indices = np.argsort(history_weights)[-4:][::-1]
    
    # Map array indices to real historical distances
    ghost_data = []
    for idx in top_4_indices:
        if idx < len(bank_80): # It fell in the 80% deep zone
            dist = res_start - idx
        else:                  # It fell in the 20% local zone
            local_idx = idx - len(bank_80)
            dist = L1 - local_idx
            
        ghost_data.extend([f"M_{dist}", w_flat[idx]])
        
    avg_weight = w_flat[t1_count]

    return [start_k, (end_idx // 1000), t1_count, t2_count, improvement,
            ghost_data[0], ghost_data[1], ghost_data[2], ghost_data[3],
            ghost_data[4], ghost_data[5], ghost_data[6], ghost_data[7],
            avg_weight, ai_error]

# --- 3. ORCHESTRATOR ---
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print(f"[{time.strftime('%H:%M:%S')}] Loading Universe Data...")
    try:
        zeros = np.loadtxt("zero", dtype=np.float64)
    except:
        print("Error: 'zero' file missing.")
        exit()

    gaps = np.diff(zeros)
    avg_gaps = (2 * np.pi) / np.log((zeros[:-1] / (2 * np.pi)) + 1e-12)
    multipliers = (gaps / avg_gaps).astype(np.float32)
    avg_gaps = avg_gaps.astype(np.float32)

    csv_file = "prototype_C_dataset.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Start_K", "End_K", "T1_Input", "T2_Input", "Improvement_Pct",
            "G1_Idx", "G1_Wt", "G2_Idx", "G2_Wt", 
            "G3_Idx", "G3_Wt", "G4_Idx", "G4_Wt", 
            "Avg_Target_Wt", "AI_Error"
        ])

    # --- GRID SEARCH SPECS ---
    starts = range(0, 1950, 50)               # 0k to 1.95M (40 slices)
    t1_inputs = [100, 200, 300, 400]          # The "Eyes" (4 options)
    t2_inputs = range(10, 110, 10)            # The "Bottleneck" (10 options)
    
    tasks = []
    for s_k in starts:
        for t1 in t1_inputs:
            for t2 in t2_inputs:
                tasks.append((s_k, t1, t2))

    total_tasks = len(tasks)
    print(f"\n🚀 LAUNCHING 2-TENSOR ABLATION GRID: {total_tasks:,} Models")

    # 4 Workers maxes out the CPU efficiently while keeping 8GB VRAM stable
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_model_grid, s, t1, t2, multipliers, avg_gaps) for s, t1, t2 in tasks]
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(result)
                    
                    if (i + 1) % 50 == 0 or i == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{total_tasks} | Slice: {result[0]}k | T1: {result[2]} -> T2: {result[3]} | Impr: {result[4]:.2f}%")
            except Exception as e:
                print(f"Task {i} failed: {e}")

    print(f"\n✅ DATASET COMPLETE. Saved to {csv_file}.")