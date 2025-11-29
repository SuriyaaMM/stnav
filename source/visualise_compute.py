import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set the style to match the professional academic look of your PDF
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['lines.linewidth'] = 2.0

def plot_all_metrics(ppo_df, replay_df, save_dir="plots"):
    """
    Generates all 10 plots from the provided PDF using the training dataframes.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Policy Convergence (KL Divergence) [cite: 414]
    # ---------------------------------------------------------
    plt.figure()
    if 'approx_kl' in ppo_df.columns:
        sns.lineplot(data=ppo_df, x=ppo_df.index, y='approx_kl', label='Approx KL', color='tab:blue')
    
    plt.title("Policy Convergence")
    plt.ylabel("Divergence (KL)")
    plt.xlabel("PPO Update Step")
    plt.yscale('log') # Log scale often helps visualize convergence behavior
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/1_policy_convergence.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. Value Function Quality (Explained Variance) [cite: 431]
    # ---------------------------------------------------------
    plt.figure()
    if 'explained_var' in ppo_df.columns:
        sns.lineplot(data=ppo_df, x=ppo_df.index, y='explained_var', color='tab:green')
    
    # Add the reference lines from the paper
    plt.axhline(1.0, color='gray', linestyle=':', label='Perfect Prediction (1.0)')
    plt.axhline(0.0, color='red', linestyle=':', label='Random Guessing (0.0)')
    
    plt.title("Value Function Quality")
    plt.ylabel(r"Explained Variance ($R^2$)")
    plt.xlabel("PPO Update Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/2_value_function_quality.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Parameter Space Trajectory [cite: 453]
    # ---------------------------------------------------------
    plt.figure()
    if 'param_change' in replay_df.columns:
        sns.lineplot(data=replay_df, x='iteration', y='param_change', marker='o', color='rebeccapurple')
    
    plt.title("Parameter Space Trajectory")
    plt.ylabel(r"Parameter Change ($||\theta_{new} - \theta_{old}||_2$)")
    plt.xlabel("Training Iteration")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3_parameter_trajectory.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. Optimization Landscape Roughness (Gradient Norm) [cite: 470]
    # ---------------------------------------------------------
    plt.figure()
    if 'grad_norm' in ppo_df.columns:
        sns.lineplot(data=ppo_df, x=ppo_df.index, y='grad_norm', color='#C0392B') # Dark red
    
    plt.title("Optimization Landscape Roughness")
    plt.ylabel("Gradient L2 Norm")
    plt.xlabel("PPO Update Step")
    plt.yscale('log') 
    plt.tight_layout()
    plt.savefig(f"{save_dir}/4_gradient_norm.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 5. Trust Region Constraint (Clip Fraction) [cite: 486]
    # ---------------------------------------------------------
    plt.figure()
    if 'clip_frac' in ppo_df.columns:
        # Fill area under curve to match paper style
        plt.fill_between(ppo_df.index, ppo_df['clip_frac'] * 100, color='#E6B0AA', alpha=0.4)
        sns.lineplot(data=ppo_df, x=ppo_df.index, y=ppo_df['clip_frac'] * 100, color='#C0392B')
    
    plt.title("Trust Region Constraint Activation")
    plt.ylabel("Clipped Samples (%)")
    plt.xlabel("PPO Update Step")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/5_trust_region.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 6. Objective Function Decomposition [cite: 507]
    # ---------------------------------------------------------
    plt.figure()
    metrics = ['policy_loss', 'value_loss', 'loss']
    colors = ['#4A90E2', '#E67E22', 'black']
    labels = ['Policy Loss', 'Value Loss', 'Total Loss']
    styles = ['-', '-', '--']

    for m, c, l, s in zip(metrics, colors, labels, styles):
        if m in ppo_df.columns:
            sns.lineplot(data=ppo_df, x=ppo_df.index, y=m, label=l, color=c, linestyle=s)

    plt.title("Objective Function Decomposition")
    plt.ylabel("Loss Value")
    plt.xlabel("PPO Update Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/6_loss_decomposition.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 7. Sample Efficiency [cite: 527]
    # ---------------------------------------------------------
    plt.figure()
    if 'return_per_sample' in replay_df.columns:
        sns.lineplot(data=replay_df, x='iteration', y='return_per_sample', marker='s', color='#1ABC9C')
    
    plt.title("Sample Efficiency")
    plt.ylabel("Return per Sample")
    plt.xlabel("Training Iteration")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/7_sample_efficiency.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 8. Cumulative Reward [cite: 549]
    # ---------------------------------------------------------
    plt.figure()
    if 'mean_reward' in replay_df.columns:
        sns.lineplot(data=replay_df, x='iteration', y='mean_reward', marker='^', color='#9B59B6') # Purple/Gold
    
    plt.title("Cumulative Reward Progression")
    plt.ylabel("Total Reward")
    plt.xlabel("Training Iteration")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/8_cumulative_reward.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 9. Optimization Phase Portrait [cite: 568]
    # ---------------------------------------------------------
    plt.figure()
    # Need both loss and param change aligned. Assuming replay_df has aggregated loss or we map it.
    # For this plot, we often average loss per iteration to match the 'iteration' scale of param change
    if 'param_change' in replay_df.columns and 'loss' in ppo_df.columns:
        # Resample ppo loss to match replay iterations (approximate)
        avg_loss_per_iter = ppo_df['loss'].groupby(np.arange(len(ppo_df)) // (len(ppo_df)//len(replay_df))).mean()
        
        # Ensure lengths match
        min_len = min(len(replay_df), len(avg_loss_per_iter))
        
        sc = plt.scatter(
            x=replay_df['param_change'][:min_len], 
            y=avg_loss_per_iter[:min_len], 
            c=replay_df['iteration'][:min_len], 
            cmap='viridis', 
            s=100, 
            edgecolors='k',
            alpha=0.8
        )
        plt.colorbar(sc, label='Training Iteration')
        
    plt.title("Optimization Phase Portrait")
    plt.ylabel("Average Loss")
    plt.xlabel(r"Parameter Change ($||\Delta \theta||$)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/9_phase_portrait.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 10. Critic Accuracy [cite: 589]
    # ---------------------------------------------------------
    plt.figure()
    if 'mean_reward' in replay_df.columns and 'pred_value' in replay_df.columns:
        plt.plot(replay_df['iteration'], replay_df['mean_reward'], 'o-', label='Actual Mean Reward')
        plt.plot(replay_df['iteration'], replay_df['pred_value'], 'x-', label='Predicted Value V(s)')
    
    plt.title("Critic Accuracy: Prediction vs Reality")
    plt.ylabel("Magnitude")
    plt.xlabel("Training Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/10_critic_accuracy.png", dpi=300)
    plt.close()

    print(f"All plots saved to {save_dir}/ directory.")

# ==========================================
# Example Usage with your dataframes
# ==========================================
if __name__ == "__main__":
    # Assuming 'ppo_df' and 'replay_df' are already in memory from your training script
    # If running standalone, uncomment the generation below:
    
    # --- Dummy Data Generation for Testing ---
    # steps = 150
    # iters = 25
    # ppo_df = pd.DataFrame({
    #     'approx_kl': np.linspace(0.01, 0.001, steps) + np.random.normal(0, 0.001, steps),
    #     'explained_var': np.linspace(0.2, 0.7, steps) + np.random.normal(0, 0.05, steps),
    #     'grad_norm': np.exp(np.linspace(2, 0, steps)) + np.random.normal(0, 0.1, steps),
    #     'clip_frac': np.linspace(0.3, 0.01, steps),
    #     'policy_loss': np.zeros(steps) + 0.01,
    #     'value_loss': np.linspace(0.3, 0.15, steps),
    #     'loss': np.linspace(0.2, 0.05, steps)
    # })
    
    # replay_df = pd.DataFrame({
    #     'iteration': range(iters),
    #     'mean_reward': np.linspace(-200, 200, iters),
    #     'param_change': np.linspace(8, 5, iters) + np.random.normal(0, 0.5, iters),
    #     'return_per_sample': np.linspace(-0.03, 0.005, iters),
    #     'pred_value': np.linspace(-200, 200, iters) + np.random.normal(0, 50, iters)
    # })
    # -----------------------------------------

    # Call the plotting function
    plot_all_metrics(ppo_df, replay_df)
    pass