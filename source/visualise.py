import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import seaborn as sns

def setup_publication_style():
    """Sets matplotlib configuration for publication-quality figures."""
    try:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
        sns.set_palette("deep")
    except ImportError:
        plt.style.use('seaborn-whitegrid')

    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'lines.linewidth': 2.0,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def save_figure(filename, output_dir):
    """Saves the current figure in PDF and PNG formats."""
    path = os.path.join(output_dir, filename)
    plt.savefig(f"{path}.pdf", format='pdf')
    plt.savefig(f"{path}.png", format='png', dpi=300)
    print(f"Saved {filename} (.pdf & .png)")
    plt.close()

def plot_optimization_analysis(data_path="./terrain_data", output_dir="./terrain_plots"):
    """Generates individual publication-quality plots."""
    os.makedirs(output_dir, exist_ok=True)
    setup_publication_style()
    
    # Load Data
    try:
        ppo_df = pd.read_csv(f"{data_path}/ppo_epoch_metrics.csv", index_col=0)
        iter_df = pd.read_csv(f"{data_path}/train_iter_metrics.csv", index_col=0)
        print(f"Loaded data: {len(ppo_df)} PPO steps, {len(iter_df)} Iterations")
    except FileNotFoundError:
        print("Error: CSV files not found. Run training first.")
        return

    # --- PLOTS 1-10 (Standard Metrics) ---
    
    # 1. KL Divergence
    plt.figure()
    plt.plot(ppo_df.index, ppo_df['kl_divergence'], label=r'$D_{KL}$', color='#1f77b4')
    plt.xlabel('PPO Update Step'); plt.ylabel('Divergence'); plt.yscale('log')
    plt.title('Policy Convergence'); plt.legend()
    save_figure("1_kl_divergence", output_dir)

    # 2. Explained Variance
    plt.figure()
    plt.plot(ppo_df.index, ppo_df['explained_variance'], color='#2ca02c')
    plt.axhline(0, color='r', linestyle=':', alpha=0.5)
    plt.xlabel('PPO Update Step'); plt.ylabel('Explained Variance')
    plt.title('Value Function Quality'); plt.ylim(bottom=-0.2, top=1.1)
    save_figure("2_explained_variance", output_dir)

    # 3. Parameter Change
    plt.figure()
    plt.plot(iter_df.index, iter_df['param_change'], color='#9467bd', marker='o')
    plt.xlabel('Training Iteration'); plt.ylabel('L2 Norm Change'); plt.yscale('log')
    plt.title('Parameter Space Trajectory')
    save_figure("3_parameter_change", output_dir)

    # 4. Gradient Norm
    plt.figure()
    plt.plot(ppo_df.index, ppo_df['grad_norm'], color='#d62728')
    plt.xlabel('PPO Update Step'); plt.ylabel('Gradient Norm'); plt.yscale('log')
    plt.title('Optimization Landscape Roughness')
    save_figure("4_gradient_norm", output_dir)

    # 5. Clipping Ratio
    plt.figure()
    plt.plot(ppo_df.index, ppo_df['clipped_ratio_pct']*100, color='#e377c2')
    plt.xlabel('PPO Update Step'); plt.ylabel('Clipped %')
    plt.title('Trust Region Activation'); plt.ylim(bottom=0)
    save_figure("5_clipping_ratio", output_dir)

    # 6. Loss Decomposition
    plt.figure()
    plt.plot(ppo_df.index, ppo_df['policy_loss'], label='Policy', alpha=0.7)
    plt.plot(ppo_df.index, ppo_df['value_loss'], label='Value', alpha=0.7)
    plt.plot(ppo_df.index, ppo_df['total_loss'], label='Total', color='k', linestyle='--')
    plt.xlabel('PPO Update Step'); plt.ylabel('Loss')
    plt.title('Objective Decomposition'); plt.legend()
    save_figure("6_loss_decomposition", output_dir)

    # 7. Sample Efficiency
    plt.figure()
    plt.plot(iter_df.index, iter_df['returns_per_sample'], color='#17becf', marker='s')
    plt.xlabel('Training Iteration'); plt.ylabel('Return/Sample')
    plt.title('Sample Efficiency')
    save_figure("7_sample_efficiency", output_dir)

    # 8. Cumulative Reward
    plt.figure()
    plt.plot(iter_df.index, iter_df['cumulative_reward'], color='#bcbd22', marker='^')
    plt.xlabel('Training Iteration'); plt.ylabel('Total Reward')
    plt.title('Cumulative Reward')
    save_figure("8_cumulative_reward", output_dir)

    # 9. Phase Portrait
    plt.figure()
    ppo_per_iter = len(ppo_df) // len(iter_df)
    avg_loss = [ppo_df['total_loss'].iloc[i*ppo_per_iter:(i+1)*ppo_per_iter].mean() for i in range(len(iter_df))]
    plt.scatter(iter_df['param_change'], avg_loss, c=range(len(iter_df)), cmap='viridis', s=100, edgecolor='k')
    plt.xlabel('Param Change'); plt.ylabel('Avg Loss'); plt.xscale('log')
    plt.title('Optimization Phase Portrait'); plt.colorbar(label='Iteration')
    save_figure("9_phase_portrait", output_dir)

    # 10. Value Accuracy
    plt.figure()
    plt.plot(iter_df.index, iter_df['avg_reward'], label='Actual', marker='o')
    plt.plot(iter_df.index, iter_df['avg_state_value'], label='Predicted', marker='x')
    plt.xlabel('Iteration'); plt.title('Critic Accuracy'); plt.legend()
    save_figure("10_value_accuracy", output_dir)

    # ==========================================
    # 11. BEST TRAJECTORY (GRID VIEW + GOALS)
    # ==========================================
    try:
        traj_df = pd.read_csv(f"{data_path}/best_trajectory.csv")
        plt.figure(figsize=(8, 8))
        
        # 1. Setup Grid Dimensions (Infer from data or default to 12x12)
        # Note: We assume coordinate system where (0,0) is top-left, 
        # so X corresponds to Rows and Y corresponds to Columns in Matrix notation
        max_x, max_y = int(traj_df['x'].max()), int(traj_df['y'].max())
        grid_h, grid_w = max(12, max_x + 1), max(12, max_y + 1)
        
        # 2. Create Visit Density Matrix
        density = np.zeros((grid_h, grid_w))
        for _, row in traj_df.iterrows():
            density[int(row['x']), int(row['y'])] += 1
            
        # 3. Plot Heatmap of Visits
        sns.heatmap(density, cmap="Blues", cbar_kws={'label': 'Visit Count'}, 
                    linewidths=1.0, linecolor='lightgray', square=True, alpha=0.5, zorder=1)
        
        # 4. Prepare Coordinates (Center them in the cells by adding 0.5)
        # Heatmap x-axis is columns (y in dataframe), y-axis is rows (x in dataframe)
        plot_x = traj_df['y'] + 0.5
        plot_y = traj_df['x'] + 0.5
        
        # 5. Overlay Path Line
        plt.plot(plot_x, plot_y, color='orange', linewidth=3, 
                 marker='.', markersize=0, label='Path', alpha=1.0, zorder=2)
        
        # 6. Identify & Plot Goals (Where points increased)
        # We calculate delta in points. If delta > 0, a goal was visited.
        points_series = traj_df['points']
        goal_indices = points_series[points_series.diff() > 0].index.tolist()
        
        if goal_indices:
            goal_x = plot_x.iloc[goal_indices]
            goal_y = plot_y.iloc[goal_indices]
            plt.scatter(goal_x, goal_y, c='gold', s=150, marker='D',
                        edgecolor='black', linewidth=1.5, zorder=3, label='Goal Reached')

        # 7. Mark Start and End
        plt.scatter(plot_x.iloc[0], plot_y.iloc[0], c='lime', s=300, 
                    edgecolor='black', linewidth=1.5, zorder=4, label='Start', marker='*')
        plt.scatter(plot_x.iloc[-1], plot_y.iloc[-1], c='red', s=250, 
                    edgecolor='black', linewidth=1.5, zorder=4, label='End', marker='X')

        plt.title('Best Episode Trajectory (Grid View)')
        plt.xlabel('Y Coordinate (Columns)')
        plt.ylabel('X Coordinate (Rows)')
        
        # Custom Legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   fancybox=True, shadow=True, ncol=4)
        
        # Fix axis ticks to show integers
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        save_figure("11_best_trajectory_grid", output_dir)
        
    except FileNotFoundError:
        print("Warning: best_trajectory.csv not found.")
    except Exception as e:
        print(f"Error plotting trajectory: {e}")

    print("\n" + "="*50)
    print(f"All plots saved to {output_dir}/")
    print("="*50)

if __name__ == "__main__":
    plot_optimization_analysis()