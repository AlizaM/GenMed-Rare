import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.tensorboard_loss_plot import plot_loss_from_tensorboard

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_loss.py <tensorboard_log_dir> [window_size]")
        sys.exit(1)
    log_dir = Path(sys.argv[1])
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    plot_loss_from_tensorboard(log_dir, window_size)
    print(f"Loss plots saved in {log_dir}")
