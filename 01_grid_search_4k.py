import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import csv
import os
from concurrent.futures import ProcessPoolExecutor

# --- 1. MODEL ARCHITECTURE ---
class PrototypeA(nn.Module):
    def __init__(self, input_size):
        super(PrototypeA, self).__init__()
        # 3 Layers, 64 Neurons
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. WORKER FUNCTION ---
def run_model(start_k, input_count, offset, multipliers, avg_gaps):
    """
    Trains and evaluates a single model configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SLICE DATA
    window_size = 100000 
    start_idx = start_k * 1000
    
    # Prevent looking into negative indices
    safe_start = max(start_idx, input_count + offset)
    end_idx = safe_start + window_size
    
    X_list, Y_list = [], []
    for i in range(safe_start, end_idx):
        h_start = i - offset - input_count
        h_end = i - offset
        
        history = multipliers[h_start:h_end]
        target_avg = avg_gaps[i]
        
        # input_count + 1 (the average gap)
        X_list.append(np.append(history, target_avg))
        Y_list.append(multipliers[i])
    
    X_data = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_data = torch.tensor(np.array(Y_list), dtype=torch.float32).unsqueeze(1)
    
    # 90/10 Split
    split = int(len(X_data) * 0.9)
    X_train, Y_train = X_data[:split].to(device), Y_data[:split].to(device)
    X_test, Y_test = X_data[split:].to(device), Y_data[split:].to(device)
    
    # 2. INITIALIZE MODEL (Fixed seed for controlled A/B testing)
    torch.manual_seed(42)
    model = PrototypeA(input_size=(input_count + 1)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.HuberLoss()
    
    # 3. TRAINING LOOP
    EPOCHS = 1500
    BATCH_SIZE = 512
    
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        idx = torch.randint(0, len(X_train), (BATCH_SIZE,))
        preds = model(X_train[idx])
        loss = criterion(preds, Y_train[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # 4. EVALUATION
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        ai_error = torch.abs(test_preds - Y_test).mean().item()
        base_error = torch.abs(torch.ones_like(Y_test) - Y_test).mean().item()
        improvement = ((base_error - ai_error) / base_error) * 100

    # 5. WEIGHT EXTRACTION (Least Squares Surrogate)
    X_np = X_test.cpu().numpy()
    Y_np = test_preds.cpu().numpy()
    
    A = np.c_[X_np, np.ones(X_np.shape[0])]
    weights, _, _, _ = np.linalg.lstsq(A, Y_np, rcond=None)
    w_flat = weights.flatten()
    
    # Extract historical weights
    history_weights_abs = np.abs(w_flat[:input_count])
    top_indices = np.argsort(history_weights_abs)[-3:][::-1]
    
    # Map array index to absolute historical distance
    # index 0 = oldest (offset + input_count)
    # index (input_count - 1) = newest (offset + 1)
    ghosts = []
    for idx in top_indices:
        dist = offset + input_count - idx
        weight = w_flat[idx]
        ghosts.append((dist, weight))
        
    avg_weight = w_flat[input_count]
    bias_weight = w_flat[input_count + 1]

    # Return formatted row
    return [
        start_k, (end_idx // 1000), input_count, offset,
        f"M_{ghosts[0][0]}", ghosts[0][1],
        f"M_{ghosts[1][0]}", ghosts[1][1],
        f"M_{ghosts[2][0]}", ghosts[2][1],
        avg_weight, bias_weight, ai_error, improvement
    ]

# --- 3. ORCHESTRATOR ---
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print(f"[{time.strftime('%H:%M:%S')}] Loading Base Data...")
    try:
        zeros = np.loadtxt("zero", dtype=np.float64)
    except Exception as e:
        print(f"Failed to load 'zero' file. Ensure it exists. Error: {e}")
        exit()

    gaps = np.diff(zeros)
    avg_gaps = (2 * np.pi) / np.log((zeros[:-1] / (2 * np.pi)) + 1e-12)
    multipliers = (gaps / avg_gaps).astype(np.float32)
    avg_gaps = avg_gaps.astype(np.float32)

    csv_file = "final_dataset.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Start_K", "End_K", "Input_Count", "Offset", 
            "Ghost_1_Idx", "Ghost_1_Wt", 
            "Ghost_2_Idx", "Ghost_2_Wt", 
            "Ghost_3_Idx", "Ghost_3_Wt", 
            "Avg_Weight", "Bias_Weight", 
            "AI_Error", "Improvement_Pct"
        ])

    # Generate task combinations
    tasks = []
    # 20 Data slices (0k to 950k, step 50k)
    for s_k in range(0, 1000, 50):
        # Input counts (10 to 190)
        for i_c in range(10, 200, 10):
            # Offsets (0 to 190)
            for off in range(0, 200, 10):
                if i_c + off <= 200:
                    tasks.append((s_k, i_c, off))

    total_tasks = len(tasks)
    print(f"Total Models to Train: {total_tasks:,}")
    print(f"[{time.strftime('%H:%M:%S')}] Commencing Grid Search...")

    # Set to 3 workers to maintain VRAM balance
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_model, s_k, i_c, off, multipliers, avg_gaps) for s_k, i_c, off in tasks]
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Progress: {i+1}/{total_tasks} models | Last Impr: {result[-1]:.2f}%")
            except Exception as e:
                print(f"Error on model {i}: {e}")

    print(f"\n✅ GRID SEARCH COMPLETE. Results saved to '{csv_file}'.")