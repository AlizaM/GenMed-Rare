#!/usr/bin/env python3
"""
Launch TensorBoard for diffusion training visualization.

Usage:
    python scripts/view_diffusion_tensorboard.py
    python scripts/view_diffusion_tensorboard.py --port 6007
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for diffusion training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config_diffusion.yaml",
        help="Path to diffusion config file"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="Port for TensorBoard server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for TensorBoard server"
    )
    args = parser.parse_args()
    
    # Load config to get log directory
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        log_dir = config['training']['log_dir']
        log_path = Path(log_dir)
        
        if not log_path.exists():
            print(f"❌ Log directory not found: {log_path}")
            print("Run training first to generate logs:")
            print(f"  python scripts/train_diffusion.py --config {args.config}")
            sys.exit(1)
            
        print("🚀 Launching TensorBoard for diffusion training...")
        print(f"📊 Log directory: {log_path}")
        print(f"🌐 URL: http://{args.host}:{args.port}")
        print("\n📈 Available metrics:")
        print("  - train/step_loss: Step-by-step training loss")
        print("  - train/epoch_loss: Average loss per epoch")
        print("  - train/learning_rate: Learning rate schedule")
        print("  - train/loss_trend_5epoch: 5-epoch moving average")
        print("  - validation/image_*: Generated validation images")
        print("\n💡 Tip: Refresh browser if you don't see metrics immediately")
        print("Press Ctrl+C to stop TensorBoard\n")
        
        # Launch TensorBoard
        cmd = [
            "tensorboard",
            "--logdir", str(log_path),
            "--port", str(args.port),
            "--host", args.host,
            "--reload_interval", "30"  # Refresh every 30 seconds
        ]
        
        subprocess.run(cmd)
        
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Missing key in config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()