import pandas as pd
import numpy as np
from pysr import PySRRegressor

def run_pysr_discovery():
    # 1. LOAD AND CLEAN THE DATA
    print("Loading data and untangling the Ghost Braid...")
    try:
        df = pd.read_csv("final_dataset.csv")
    except FileNotFoundError:
        print("final_dataset.csv not found!")
        return

    # Filter out the 150k outlier as requested
    df = df[df['Start_K'] != 150]

    # Get the best performing model for each Start_K to ensure high-quality data
    best_models = df.sort_values('Improvement_Pct', ascending=False).groupby('Start_K').head(1)

    X = best_models['Start_K'].values.reshape(-1, 1)

    # Clean the "M_" strings and convert to integers
    g1 = best_models['Ghost_1_Idx'].str.replace('M_', '').astype(float).values
    g2 = best_models['Ghost_2_Idx'].str.replace('M_', '').astype(float).values
    g3 = best_models['Ghost_3_Idx'].str.replace('M_', '').astype(float).values

    # 2. UNTANGLE THE RANKS
    # We sort them horizontally so Y_mid is always the center of the resonance band
    ghost_matrix = np.column_stack((g1, g2, g3))
    ghost_sorted = np.sort(ghost_matrix, axis=1)

    Y_floor = ghost_sorted[:, 0]
    Y_core  = ghost_sorted[:, 1]
    Y_ceil  = ghost_sorted[:, 2]

    # 3. CONFIGURE THE ALIEN MATHEMATICIAN (PySR)
    # We give it all the tools, but heavily penalize complexity so it doesn't give us a 5-page formula
    model = PySRRegressor(
        niterations=100,  # High iteration count for deep search
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
        complexity_of_operators={"^": 2, "sin": 2, "cos": 2, "exp": 2, "log": 2},
        parsimony=1e-3,  # Force it to prefer simple, elegant physics laws
        model_selection="best",
        loss="loss(prediction, target) = (prediction - target)^2" # Standard MSE
    )

    # 4. RUN THE DISCOVERY ON THE CORE RESONANCE
    print("\n" + "="*60)
    print("🧬 IGNITING JULIA ENGINE: SEARCHING FOR CORE RESONANCE LAW (K_t)")
    print("="*60)
    
    # We fit X (Time) against Y_core (The Center of the Ghost Braid)
    model.fit(X, Y_core)

    print("\n" + "="*60)
    print("🏆 PYSR HAS FINISHED.")
    print("="*60)
    
    # Print the absolute best formula it found
    best_formula = model.get_best()
    print(f"\nBest Core Equation: K(t) = {best_formula.equation}")
    print(f"Loss (MSE): {best_formula.loss:.4f}")
    print(f"Complexity Score: {best_formula.complexity}")

if __name__ == "__main__":
    run_pysr_discovery()