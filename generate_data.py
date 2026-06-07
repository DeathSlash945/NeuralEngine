import numpy as np
import torch

class World:
    def __init__(self):
        # Base matrices
        self.W_e = np.array([
            [-0.4, -0.2, -0.1,  0.8], # Anger
            [ 0.6, -0.3,  0.7, -0.5], # Joy
            [-0.2,  0.7, -0.3,  0.6]  # Fear
        ])
        
        self.W_p = np.array([
            [ 0.3, -0.1,  0.2, -0.7], # Trust
            [ 0.1,  0.4,  0.5, -0.3], # Confidence
            [-0.2,  0.3, -0.1,  0.6], # Aggression
            [ 0.0, -0.2,  0.4, -0.1], # Curiosity
            [ 0.4, -0.3,  0.5, -0.4]  # Optimism
        ])
        
        self.gamma = 0.85
        self.alpha = 0.96

    def rule_based_action(self, traits):
        trust, _, aggression, _, _ = traits
        if aggression > 0.6: return 2   # Attack
        if trust < 0.3: return 0        # Withdraw
        return 1                        # Cooperate

    def simulate_trajectory(self, num_steps, archetype_kind):
        # archetypes, idk I made them up
        if archetype_kind == "warrior": p_curr = np.array([0.2, 0.8, 0.7, 0.3, 0.4])
        elif archetype_kind == "scholar": p_curr = np.array([0.7, 0.5, 0.1, 0.9, 0.6])
        else: p_curr = np.array([0.5, 0.5, 0.3, 0.5, 0.5])
            
        e_curr = np.array([0.0, 0.0, 0.0])
        prev_action = 1
        x_history, p_history, action_history = [], [], []
        
        for t in range(num_steps):
            # action feedback modifiers
            social_mod = 0.4 if prev_action == 1 else (-0.5 if prev_action == 2 else -0.2)
            stress_mod = 0.5 if prev_action == 2 else (0.2 if prev_action == 0 else 0.0)
            malice_mod = 0.6 if prev_action == 2 else 0.0
            
            # non linear env math and stochastic tipping points
            # If stress peaks past 0.7, introduce an exponential burnout modifier
            burnout = 1.5 if (len(x_history) > 0 and x_history[-1][1] > 0.7) else 1.0 #thought of this last second as a what-if :P
            
            social_val = np.random.uniform(-0.3, 0.6) + social_mod
            stress = (np.random.uniform(-0.1, 0.5) + stress_mod) * burnout
            reward = np.random.uniform(-0.2, 0.7) - (0.2 * malice_mod)
            malice = np.random.choice([0.0, 0.1, 0.8], p=[0.7, 0.2, 0.1]) or malice_mod
            x_t = np.clip(np.array([social_val, stress, reward, malice]), -1.0, 1.0)
            
            # sigmoid activation function to simulate non-linear emotional thresholds, state step udates here
            raw_e = self.gamma * e_curr + (1 - self.gamma) * (self.W_e @ x_t)
            e_next = 1 / (1 + np.exp(-5 * (raw_e - 0.5))) # Non-linear mapping
            
            p_next = np.clip(self.alpha * p_curr + (1 - self.alpha) * (p_curr + self.W_p @ x_t), 0.0, 1.0)
            action_t = self.rule_based_action(p_next)
            
            x_history.append(x_t)
            p_history.append(p_next)
            action_history.append(action_t)
            e_curr = e_next
            p_curr = p_next
            prev_action = action_t
        return np.array(x_history), np.array(p_history), np.array(action_history)

def generate_dataset():
    world = World()
    archetypes = ["balanced", "warrior", "scholar"]
    all_X, all_Y_p, all_Y_a = [], [], []
    
    print("Generating dataset...")
    for _ in range(5000):
        arch = np.random.choice(archetypes)
        X_seq, Y_p_seq, Y_a_seq = world.simulate_trajectory(50, arch)
        all_X.append(X_seq)
        all_Y_p.append(Y_p_seq)
        all_Y_a.append(Y_a_seq)
        
    torch.save(torch.tensor(np.array(all_X), dtype=torch.float32), "data/X_events.pt")
    torch.save(torch.tensor(np.array(all_Y_p), dtype=torch.float32), "data/Y_personality.pt")
    torch.save(torch.tensor(np.array(all_Y_a), dtype=torch.long), "data/Y_actions.pt")
    print("Completed lol")

if __name__ == "__main__":
    generate_dataset()