#!/usr/bin/env python3
"""
Demonstration and testing script for diffusion training resume functionality.

This script demonstrates:
1. How to resume training from a specific checkpoint
2. How to automatically find and resume from the latest checkpoint  
3. How to list available checkpoints
4. How to verify checkpoint integrity

Usage:
    python scripts/test_resume_demo.py --list-checkpoints
    python scripts/test_resume_demo.py --verify-checkpoints
    python scripts/test_resume_demo.py --demo-resume
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_diffusion import (
    find_latest_checkpoint,
    extract_step_from_checkpoint_path,
    load_config
)


def list_checkpoints(checkpoint_dir):
    """List all available checkpoints in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory does not exist: {checkpoint_path}")
        return
    
    print(f"📁 Searching for checkpoints in: {checkpoint_path}")
    print("=" * 60)
    
    step_checkpoints = []
    epoch_checkpoints = []
    other_dirs = []
    
    for path in checkpoint_path.iterdir():
        if path.is_dir():
            if path.name.startswith('checkpoint-step-'):
                try:
                    step_num = int(path.name.split('-')[-1])
                    step_checkpoints.append((step_num, path))
                except ValueError:
                    other_dirs.append(path)
            elif path.name.startswith('checkpoint-epoch-'):
                try:
                    epoch_num = int(path.name.split('-')[-1])
                    epoch_checkpoints.append((epoch_num, path))
                except ValueError:
                    other_dirs.append(path)
            elif path.name == 'best_model':
                print(f"🌟 Best model: {path}")
            else:
                other_dirs.append(path)
    
    # Sort and display step checkpoints
    if step_checkpoints:
        print(f"\\n📊 Step-based checkpoints ({len(step_checkpoints)} found):")
        step_checkpoints.sort()
        for step_num, path in step_checkpoints:
            size = get_checkpoint_size(path)
            files = list(path.glob("*"))
            print(f"  Step {step_num:>8,}: {path.name:<25} ({size}, {len(files)} files)")
    
    # Sort and display epoch checkpoints
    if epoch_checkpoints:
        print(f"\\n📈 Epoch-based checkpoints ({len(epoch_checkpoints)} found):")
        epoch_checkpoints.sort()
        for epoch_num, path in epoch_checkpoints:
            size = get_checkpoint_size(path)
            files = list(path.glob("*"))
            print(f"  Epoch {epoch_num:>7}: {path.name:<25} ({size}, {len(files)} files)")
    
    # Display other directories
    if other_dirs:
        print(f"\\n📂 Other directories ({len(other_dirs)} found):")
        for path in other_dirs:
            size = get_checkpoint_size(path)
            print(f"  {path.name:<30} ({size})")
    
    # Show latest checkpoint
    print("\\n" + "=" * 60)
    latest_checkpoint, latest_step = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"🔥 Latest checkpoint: {Path(latest_checkpoint).name} (step {latest_step:,})")
    else:
        print("🆕 No checkpoints found - would start fresh training")


def get_checkpoint_size(checkpoint_path):
    """Get human-readable size of checkpoint directory."""
    try:
        total_size = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
        return format_bytes(total_size)
    except:
        return "unknown size"


def format_bytes(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def verify_checkpoint(checkpoint_path):
    """Verify that a checkpoint has the expected structure."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return False, f"Directory does not exist: {checkpoint_path}"
    
    required_files = ['adapter_config.json']
    optional_files = ['adapter_model.bin', 'adapter_model.safetensors']
    
    issues = []
    
    # Check for required files
    for required_file in required_files:
        file_path = checkpoint_path / required_file
        if not file_path.exists():
            issues.append(f"Missing required file: {required_file}")
    
    # Check for at least one model file
    model_files = [f for f in optional_files if (checkpoint_path / f).exists()]
    if not model_files:
        issues.append(f"Missing model weights file (expected one of: {optional_files})")
    
    # Try to load and validate adapter config
    config_path = checkpoint_path / 'adapter_config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            
            required_config_fields = ['peft_type', 'r', 'lora_alpha', 'target_modules']
            for field in required_config_fields:
                if field not in adapter_config:
                    issues.append(f"Missing config field: {field}")
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON in adapter_config.json: {e}")
        except Exception as e:
            issues.append(f"Error reading adapter_config.json: {e}")
    
    return len(issues) == 0, issues


def verify_all_checkpoints(checkpoint_dir):
    """Verify all checkpoints in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory does not exist: {checkpoint_path}")
        return
    
    print(f"🔍 Verifying checkpoints in: {checkpoint_path}")
    print("=" * 60)
    
    all_checkpoints = []
    
    for path in checkpoint_path.iterdir():
        if path.is_dir() and (path.name.startswith('checkpoint-') or path.name == 'best_model'):
            all_checkpoints.append(path)
    
    if not all_checkpoints:
        print("📭 No checkpoints found to verify")
        return
    
    valid_count = 0
    invalid_count = 0
    
    for checkpoint in sorted(all_checkpoints):
        is_valid, issues = verify_checkpoint(checkpoint)
        
        if is_valid:
            print(f"✅ {checkpoint.name:<25} - Valid")
            valid_count += 1
        else:
            print(f"❌ {checkpoint.name:<25} - Invalid:")
            for issue in issues:
                print(f"   - {issue}")
            invalid_count += 1
    
    print("\\n" + "=" * 60)
    print(f"📊 Verification Summary:")
    print(f"   ✅ Valid checkpoints:   {valid_count}")
    print(f"   ❌ Invalid checkpoints: {invalid_count}")
    print(f"   📁 Total checked:       {valid_count + invalid_count}")


def demo_resume_commands(config_path):
    """Demonstrate resume command examples."""
    config = load_config(config_path)
    checkpoint_dir = config['training']['checkpoint_dir']
    
    print("🚀 Diffusion Training Resume Command Examples")
    print("=" * 60)
    
    print("\\n1. 🆕 Start fresh training (no resume):")
    print(f"   python scripts/train_diffusion.py --config {config_path}")
    
    print("\\n2. 🔄 Resume from latest checkpoint automatically:")
    print(f"   python scripts/train_diffusion.py --config {config_path} --resume-latest")
    
    print("\\n3. 📁 Resume from specific checkpoint:")
    print(f"   python scripts/train_diffusion.py --config {config_path} \\\\")
    print(f"       --resume {checkpoint_dir}/checkpoint-step-5000")
    
    print("\\n4. 🔍 List available checkpoints:")
    print(f"   python scripts/test_resume_demo.py --list-checkpoints --config {config_path}")
    
    print("\\n5. ✅ Verify checkpoint integrity:")
    print(f"   python scripts/test_resume_demo.py --verify-checkpoints --config {config_path}")
    
    # Check current state
    print("\\n" + "=" * 60)
    latest_checkpoint, latest_step = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"🔥 Current state: Latest checkpoint at step {latest_step:,}")
        print(f"   To continue training: add --resume-latest to your command")
    else:
        print("🆕 Current state: No checkpoints found")
        print("   Training will start from scratch")


def main():
    parser = argparse.ArgumentParser(description="Diffusion training resume demo and testing")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to diffusion config file"
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints"
    )
    parser.add_argument(
        "--verify-checkpoints", 
        action="store_true",
        help="Verify integrity of all checkpoints"
    )
    parser.add_argument(
        "--demo-resume",
        action="store_true", 
        help="Show resume command examples"
    )
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        checkpoint_dir = config['training']['checkpoint_dir']
        
        if args.list_checkpoints:
            list_checkpoints(checkpoint_dir)
        elif args.verify_checkpoints:
            verify_all_checkpoints(checkpoint_dir)
        elif args.demo_resume:
            demo_resume_commands(args.config)
        else:
            # Default: show all information
            print("🎯 Diffusion Training Resume Demo")
            print("=" * 60)
            
            demo_resume_commands(args.config)
            print("\\n")
            list_checkpoints(checkpoint_dir)
            print("\\n")
            verify_all_checkpoints(checkpoint_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()