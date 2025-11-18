#!/usr/bin/env python3
"""
Plot diffusion training metrics from TensorBoard logs.

Usage:
    python scripts/plot_diffusion_metrics.py
    python scripts/plot_diffusion_metrics.py --save-plots
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")
    
    # Find event files
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_path}")
    
    # Use the most recent event file
    event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading data from: {event_file.name}")
    
    # Load data
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    # Extract metrics
    data = {}
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"📈 Available metrics: {scalar_tags}")
    
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def plot_training_metrics(data, save_plots=False, output_dir="outputs/plots"):
    """Create comprehensive training plots."""
    
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Diffusion Model Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Step Loss
    if 'train/step_loss' in data:
        ax = axes[0, 0]
        steps = data['train/step_loss']['steps']
        values = data['train/step_loss']['values']
        ax.plot(steps, values, alpha=0.7, linewidth=0.8, color='blue')
        
        # Add smoothed line
        if len(values) > 10:
            window = max(1, len(values) // 50)  # Smooth over 2% of data
            smoothed = pd.Series(values).rolling(window=window, center=True).mean()
            ax.plot(steps, smoothed, color='red', linewidth=2, label=f'Smoothed (window={window})')
            ax.legend(ax.set_title('Training Loss (per step)'))
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Epoch Loss
    if 'train/epoch_loss' in data:
        ax = axes[0, 1]
        epochs = data['train/epoch_loss']['steps']  # Steps are actually epochs for this metric
        values = data['train/epoch_loss']['values']
        ax.plot(epochs, values, marker='o', linewidth=2, markersize=4, color='green')
        ax.set_title('Average Loss per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    if 'train/learning_rate' in data:
        ax = axes[1, 0]
        steps = data['train/learning_rate']['steps']
        values = data['train/learning_rate']['values']
        ax.plot(steps, values, color='orange', linewidth=2)
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Loss Trend (if available)
    if 'train/loss_trend_5epoch' in data:
        ax = axes[1, 1]
        epochs = data['train/loss_trend_5epoch']['steps']
        values = data['train/loss_trend_5epoch']['values']
        ax.plot(epochs, values, marker='s', linewidth=2, markersize=4, color='purple')
        ax.set_title('5-Epoch Loss Trend')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('5-Epoch Moving Average')
        ax.grid(True, alpha=0.3)
    else:
        # If no trend data, show training progress
        ax = axes[1, 1]
        if 'train/epoch_progress' in data:
            steps = data['train/epoch_progress']['steps']
            progress = data['train/epoch_progress']['values']
            ax.plot(steps, progress, color='brown', linewidth=2)
            ax.set_title('Training Progress')
            ax.set_xlabel('Step')
            ax.set_ylabel('Epoch Progress')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = output_path / "diffusion_training_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Metrics plot saved: {plot_file}")
    
    plt.show()


def print_training_summary(data):
    """Print a summary of training progress."""
    print("\\n" + "="*60)
    print("🎯 DIFFUSION TRAINING SUMMARY")
    print("="*60)
    
    if 'train/step_loss' in data:
        step_losses = data['train/step_loss']['values']
        print(f"📈 Step Loss:")
        print(f"   Initial: {step_losses[0]:.4f}")
        print(f"   Final:   {step_losses[-1]:.4f}")
        print(f"   Best:    {min(step_losses):.4f}")
        print(f"   Reduction: {((step_losses[0] - step_losses[-1]) / step_losses[0] * 100):+.1f}%")
    
    if 'train/epoch_loss' in data:
        epoch_losses = data['train/epoch_loss']['values']
        epochs = data['train/epoch_loss']['steps']
        print(f"\\n📊 Epoch Loss:")
        print(f"   Epochs completed: {len(epochs)}")
        print(f"   Best epoch loss: {min(epoch_losses):.4f} (epoch {epochs[epoch_losses.index(min(epoch_losses))]})") 
        print(f"   Latest epoch loss: {epoch_losses[-1]:.4f}")
    
    if 'train/learning_rate' in data:
        lr_values = data['train/learning_rate']['values']
        print(f"\\n⚙️ Learning Rate:")
        print(f"   Initial: {lr_values[0]:.2e}")
        print(f"   Current: {lr_values[-1]:.2e}")
    
    total_steps = max([max(data[key]['steps']) for key in data.keys() if data[key]['steps']])
    print(f"\\n🏃‍♂️ Training Progress:")
    print(f"   Total steps: {total_steps:,}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Plot diffusion training metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to diffusion config file"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to file instead of just displaying"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        log_dir = config['training']['log_dir']
        
        # Load tensorboard data
        data = load_tensorboard_data(log_dir)
        
        # Print summary
        print_training_summary(data)
        
        # Create plots
        plot_training_metrics(data, save_plots=args.save_plots, output_dir=args.output_dir)
        
        if not args.save_plots:
            print("\\n💡 Tip: Use --save-plots to save plots to file")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\nMake sure training has been started and TensorBoard logs exist.")
        print(f"Expected log directory: {config.get('training', {}).get('log_dir', 'Not found')}")


if __name__ == "__main__":
    main()