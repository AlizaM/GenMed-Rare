#!/usr/bin/env python3
"""
View and organize diffusion validation images generated during training.

Usage:
    python scripts/view_validation_images.py
    python scripts/view_validation_images.py --step 5000
    python scripts/view_validation_images.py --create-gif
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from PIL import Image
import numpy as np


def load_config(config_path):
    """Load config to get validation image directory."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_validation_steps(validation_dir):
    """Get all available validation steps."""
    validation_path = Path(validation_dir)
    if not validation_path.exists():
        return []
    
    steps = []
    for step_dir in validation_path.iterdir():
        if step_dir.is_dir() and step_dir.name.startswith('step_'):
            try:
                step_num = int(step_dir.name.split('_')[1])
                steps.append(step_num)
            except (IndexError, ValueError):
                continue
    
    return sorted(steps)


def view_step_images(validation_dir, step):
    """Display validation images for a specific step."""
    step_dir = Path(validation_dir) / f"step_{step:06d}"
    
    if not step_dir.exists():
        print(f"❌ No validation images found for step {step}")
        print(f"Available steps: {get_validation_steps(validation_dir)}")
        return
    
    # Find all images in the step directory
    image_files = sorted(list(step_dir.glob("*.png")) + list(step_dir.glob("*.jpg")))
    
    if not image_files:
        print(f"❌ No image files found in {step_dir}")
        return
    
    print(f"📸 Viewing {len(image_files)} validation images from step {step}")
    print(f"📁 Directory: {step_dir}")
    
    # Create subplot grid
    n_images = len(image_files)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img_file in enumerate(image_files):
        if i >= len(axes):
            break
            
        # Load and display image
        img = Image.open(img_file)
        axes[i].imshow(img, cmap='gray' if img.mode == 'L' else None)
        axes[i].set_title(f"{img_file.name}\\nStep {step}", fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Validation Images - Step {step:,}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_progress_gif(validation_dir, output_path="validation_progress.gif"):
    """Create a GIF showing validation image progress over training."""
    steps = get_validation_steps(validation_dir)
    
    if len(steps) < 2:
        print(f"❌ Need at least 2 validation steps to create GIF. Found: {len(steps)}")
        return
    
    print(f"🎬 Creating progress GIF from {len(steps)} validation steps...")
    
    frames = []
    for step in steps:
        step_dir = Path(validation_dir) / f"step_{step:06d}"
        
        # Get the first image from each step
        image_files = sorted(list(step_dir.glob("*.png")) + list(step_dir.glob("*.jpg")))
        if image_files:
            img = Image.open(image_files[0])
            
            # Add step label to image
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
            ax.set_title(f"Step {step:,}", fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Convert plot to PIL Image
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            frames.append(frame)
            plt.close(fig)
    
    if frames:
        # Save GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000,  # 1 second per frame
            loop=0
        )
        print(f"✅ Progress GIF saved: {output_path}")
        print(f"   Frames: {len(frames)}")
        print(f"   Duration: {len(frames)} seconds")
    else:
        print("❌ No images found to create GIF")


def view_latest_images(validation_dir):
    """View images from the latest validation step."""
    steps = get_validation_steps(validation_dir)
    
    if not steps:
        print(f"❌ No validation images found in {validation_dir}")
        return
    
    latest_step = max(steps)
    print(f"🔥 Showing latest validation images (step {latest_step})")
    view_step_images(validation_dir, latest_step)


def list_available_steps(validation_dir):
    """List all available validation steps."""
    steps = get_validation_steps(validation_dir)
    
    print(f"📋 Available validation steps in {validation_dir}:")
    if steps:
        for step in steps:
            step_dir = Path(validation_dir) / f"step_{step:06d}"
            image_count = len(list(step_dir.glob("*.png")) + list(step_dir.glob("*.jpg")))
            print(f"   Step {step:6,}: {image_count} images")
        print(f"\\nTotal: {len(steps)} validation checkpoints")
    else:
        print("   No validation steps found")
        print("\\n💡 Make sure training has run with validation enabled")


def main():
    parser = argparse.ArgumentParser(description="View diffusion validation images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to diffusion config file"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Specific step to view (default: latest)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available validation steps"
    )
    parser.add_argument(
        "--create-gif",
        action="store_true",
        help="Create a progress GIF from all validation images"
    )
    parser.add_argument(
        "--gif-output",
        type=str,
        default="outputs/validation_progress.gif",
        help="Output path for progress GIF"
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        validation_dir = config['training'].get(
            'validation_image_dir', 
            f"{config['training']['checkpoint_dir']}/../validation_images"
        )
        
        if args.list:
            list_available_steps(validation_dir)
        elif args.create_gif:
            create_progress_gif(validation_dir, args.gif_output)
        elif args.step is not None:
            view_step_images(validation_dir, args.step)
        else:
            view_latest_images(validation_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()