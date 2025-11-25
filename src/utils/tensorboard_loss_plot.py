import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss_from_tensorboard(log_dir, window_size=5):
    # Plot last-N-trend (sliding window of last N values at each point)
    def last_n_trend(data, N):
        return [np.mean(data[max(0, i-N+1):i+1]) for i in range(len(data))]
    log_dir = Path(log_dir)
    event_files = list(log_dir.glob('**/events.out.tfevents.*'))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    # Use the latest event file
    event_file = sorted(event_files)[-1]
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags()['scalars']
    # Try to use 'epoch_loss' if available, else fall back to 'loss' (step-wise)
    metric_tag = None
    if 'epoch_loss' in tags:
        loss_events = ea.Scalars('epoch_loss')
        steps = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]
        metric_tag = 'epoch_loss'
        x_label = 'Epoch'
        plot_prefix = 'epoch_loss'
    elif 'loss' in tags:
        print("No 'epoch_loss' found, using step-wise 'loss' instead.")
        loss_events = ea.Scalars('loss')
        steps = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]
        metric_tag = 'loss'
        x_label = 'Step'
        plot_prefix = 'step_loss'
    else:
        print(f"No 'epoch_loss' or 'loss' tag found in {tags}")
        return
    # Plot full loss curve
    plt.figure(figsize=(10,6))
    plt.plot(steps, losses, label=f'{metric_tag.capitalize()}', marker='o', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title(f'{metric_tag.capitalize()} Over Training')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir / f'{plot_prefix}_full.png', dpi=150)
    plt.close()
    # Plot moving average
    ma = moving_average(losses, window_size)
    plt.figure(figsize=(10,6))
    plt.plot(steps[:len(ma)], ma, label=f'Moving Avg ({window_size})', color='orange', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title(f'{metric_tag.capitalize()} (Moving Average, window={window_size})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir / f'{plot_prefix}_ma{window_size}.png', dpi=150)
    plt.close()
    # Plot last N trend
    # Sliding window trend (last N at each point)
    N = window_size
    trend = last_n_trend(losses, N)
    plt.figure(figsize=(10,6))
    plt.plot(steps, trend, label=f'Last-{N}-Trend', color='purple', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title(f'{metric_tag.capitalize()} (Last {N} Trend)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir / f'{plot_prefix}_trend{N}.png', dpi=150)
    plt.close()
    N = window_size
    plt.figure(figsize=(10,6))
    plt.plot(steps[-N:], losses[-N:], label=f'Last {N} {x_label}s', color='green', marker='o', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title(f'{metric_tag.capitalize()} (Last {N} {x_label}s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_dir / f'{plot_prefix}_last{N}.png', dpi=150)
    plt.close()
    print(f"Saved: {plot_prefix}_full.png, {plot_prefix}_ma{window_size}.png, {plot_prefix}_trend{N}.png, {plot_prefix}_last{N}.png in {log_dir} (using '{metric_tag}')")
