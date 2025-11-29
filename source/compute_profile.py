import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import re
from fvcore.nn import FlopCountAnalysis
from network import ActorCriticNetwork
from terrain import Terrain

# --- Configuration ---
sns.set_theme(style="white", context="paper", font_scale=1.2)
# Using a distinct palette that is professional and distinct
colors = sns.color_palette("Paired") 

def measure_latency(model, input_tensor, device, warmup=50, runs=200):
    """
    Measures robust inference latency (time) with CUDA events if available.
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            
    # Timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(input_tensor)
        end.record()
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end) # ms
        avg_time = total_time / runs
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(input_tensor)
        end = time.time()
        avg_time = (end - start) * 1000 / runs # Convert to ms
        
    return avg_time

def get_detailed_complexity(model, input_res=(1, 4, 256, 256)):
    """
    Returns detailed breakdown of FLOPs, Parameters, and Latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_res).to(device)
    
    # 1. Measure FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    
    flops_breakdown = {}
    # Get immediate children breakdown
    for name, _ in model.named_children():
        child_flops = flops.by_module().get(name, 0)
        if child_flops > 0:
            flops_breakdown[name] = child_flops

    # Handle remaining ops not in named children
    tracked_flops = sum(flops_breakdown.values())
    if total_flops > tracked_flops:
        flops_breakdown['Head/Misc'] = total_flops - tracked_flops

    # 2. Measure Parameters
    total_params = sum(p.numel() for p in model.parameters())
    params_breakdown = {}
    for name, child in model.named_children():
        p_count = sum(p.numel() for p in child.parameters())
        if p_count > 0:
            params_breakdown[name] = p_count
            
    # 3. Measure Latency
    latency_ms = measure_latency(model, dummy_input, device)
    
    return {
        "total_flops": total_flops,
        "flops_breakdown": flops_breakdown,
        "total_params": total_params,
        "params_breakdown": params_breakdown,
        "latency_ms": latency_ms,
        "device": device.type.upper()
    }

def plot_detailed_donut(data_dict, total_val, unit_label, center_text, title, save_path):
    """
    Generates a high-detail, uncluttered donut chart with a sidebar data table (legend).
    """
    
    # --- AUTO-SCALE LOGIC ---
    # Automatically switch units (G -> M -> K) to avoid "0.00" labels
    scale_factor = 1.0
    
    if unit_label == "G" and total_val < 0.1 * 1e9:
        unit_label = "M"
        scale_factor = 1e6
        # Fix the center text units dynamically
        total_val_scaled = total_val / scale_factor
        center_text = re.sub(r"[\d\.]+\s?G", f"{total_val_scaled:.2f} M", center_text)
        center_text = center_text.replace("GFLOPs", "MFLOPs")
    
    elif unit_label == "M" and total_val < 0.1 * 1e6:
        unit_label = "K"
        scale_factor = 1e3
        total_val_scaled = total_val / scale_factor
        center_text = re.sub(r"[\d\.]+\s?M", f"{total_val_scaled:.2f} K", center_text)
    
    else:
        # Default scaling based on input label
        if unit_label == "G": scale_factor = 1e9
        elif unit_label == "M": scale_factor = 1e6
        elif unit_label == "K": scale_factor = 1e3
    # ------------------------

    labels = []
    sizes = []
    
    # Sort data: Largest slices first
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    
    for name, value in sorted_items:
        percentage = (value / total_val) * 100
        val_scaled = value / scale_factor
        
        # Format: Name ...... 1.25 M (45.2%)
        # Using simple spacing here; consistent font helps alignment
        label_str = f"{name}: {val_scaled:.2f}{unit_label}  ({percentage:.1f}%)"
        labels.append(label_str)
        sizes.append(value)

    # Plot Setup
    # Wider figure (12x7) to accommodate the sidebar legend without squishing the chart
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # 1. The Donut (No Labels)
    wedges, _ = ax.pie(sizes, labels=None, startangle=90, colors=colors,
                       wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2})
    
    # 2. The Center Text (Summary)
    ax.text(0, 0, center_text, ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
    
    # 3. The Sidebar Legend (The "Heavy Detail" Part)
    # bbox_to_anchor places it outside the chart area
    legend = ax.legend(wedges, labels, 
              title="Module Breakdown", 
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), 
              fontsize=11, 
              frameon=False)
    
    # Bold the legend title
    plt.setp(legend.get_title(), fontsize='12', fontweight='bold')

    ax.set_title(title, fontsize=15, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved uncluttered detailed plot to {save_path}")

if __name__ == "__main__":
    # Initialize
    terrain = Terrain(seed=42)
    model = ActorCriticNetwork(terrain.get_state_shape(), len(terrain.actions))
    
    # 1. Get Metrics
    metrics = get_detailed_complexity(model, input_res=(1, *terrain.get_state_shape()))
    
    print("=== Complexity Report ===")
    print(f"Device: {metrics['device']}")
    print(f"Total FLOPs: {metrics['total_flops']/1e9:.6f} G")
    print(f"Total Params: {metrics['total_params']/1e6:.4f} M")
    print(f"Latency: {metrics['latency_ms']:.4f} ms")

    # 2. Generate Plots
    import os
    os.makedirs("terrain_plots", exist_ok=True)
    
    # --- Plot 1: Compute (FLOPs) ---
    # We pass "G" (Giga) as the intent, but the function will auto-downscale to M (Mega)
    # because the rover model is likely very efficient.
    flops_val_g = metrics['total_flops'] / 1e9
    flops_center = (f"Total Compute\n"
                    f"{flops_val_g:.4f} GFLOPs\n\n"
                    f"Inference Time\n"
                    f"{metrics['latency_ms']:.2f} ms\n"
                    f"({metrics['device']})")
    
    plot_detailed_donut(
        metrics['flops_breakdown'], 
        metrics['total_flops'], 
        "G", 
        flops_center, 
        "Computational Cost (FLOPs Profile)", 
        "terrain_plots/compute_flops_dist.png"
    )
    
    # --- Plot 2: Memory (Parameters) ---
    params_val_m = metrics['total_params'] / 1e6
    params_center = (f"Total Model Size\n"
                     f"{params_val_m:.2f} Million\n"
                     f"Parameters")
    
    plot_detailed_donut(
        metrics['params_breakdown'], 
        metrics['total_params'], 
        "M", 
        params_center, 
        "Memory Footprint (Parameter Profile)", 
        "terrain_plots/compute_params_dist.png"
    )