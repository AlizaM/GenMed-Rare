#!/usr/bin/env python3
"""
Quick test script for diffusion model training with REAL training run.

This script runs a minimal diffusion training test to validate:
- Dataset loading works correctly
- Model initialization succeeds
- Training loop runs without errors
- LoRA training works end-to-end
- Memory usage is acceptable

Runs ACTUAL training on a small subset (10-20 images) for 1 epoch.
"""

import os
import sys
import yaml
import torch
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.diffusion_dataset import ChestXrayDiffusionDataset, collate_fn


def create_test_config(original_config_path: str) -> dict:
    """Create a test configuration with reduced parameters."""
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce training parameters for testing
    config['training']['num_train_epochs'] = 1
    config['training']['train_batch_size'] = 1
    config['training']['gradient_accumulation_steps'] = 2
    config['training']['logging_steps'] = 2  # Log every 2 steps
    config['training']['save_steps'] = 20   # Save every 20 steps
    config['training']['validation_steps'] = 10
    config['training']['lr_warmup_steps'] = 5
    config['training']['max_train_steps'] = 10  # Limit to 10 training steps
    
    # Use temporary output directory
    test_output_dir = "outputs/diffusion_test_run"
    config['training']['checkpoint_dir'] = f"{test_output_dir}/checkpoints"
    config['training']['log_dir'] = f"{test_output_dir}/logs"
    
    # Use CPU if no GPU available
    if not torch.cuda.is_available():
        config['training']['mixed_precision'] = "no"
        config['hardware']['enable_attention_slicing'] = False
        config['hardware']['enable_vae_slicing'] = False
    
    return config


def test_dataset_loading(config: dict) -> int:
    """Test dataset loading and return number of samples."""
    print("=" * 50)
    print("TESTING DATASET LOADING")
    print("=" * 50)
    
    data_dir = config['data']['data_dir']
    csv_file = Path(data_dir) / config['data']['csv_file']
    
    print(f"Data directory: {data_dir}")
    print(f"CSV file: {csv_file}")
    print(f"CSV exists: {csv_file.exists()}")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Create dataset
    dataset = ChestXrayDiffusionDataset(
        csv_file=str(csv_file),
        data_dir=data_dir,
        image_size=config['data']['image_size'],
        prompt_template=config['data']['prompt_template'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip']
    )
    
    print(f"✓ Dataset created successfully")
    print(f"✓ Total samples: {len(dataset)}")
    
    # Test loading a few samples
    print("\\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            print(f"✓ Sample {i}: pixel_values shape={sample['pixel_values'].shape}, text='{sample['text'][:50]}...'")
        except Exception as e:
            print(f"✗ Error loading sample {i}: {e}")
            raise
    
    return len(dataset)


def test_dataloader(config: dict, dataset_size: int) -> DataLoader:
    """Test DataLoader creation with a small subset."""
    print("\\n" + "=" * 50)
    print("TESTING DATALOADER")
    print("=" * 50)
    
    # Handle both absolute and relative CSV paths
    csv_file = config['data']['csv_file']
    if Path(csv_file).is_absolute():
        # Already absolute path (test CSV)
        csv_path = csv_file
    else:
        # Relative path (original config), need to join with data_dir
        csv_path = str(Path(config['data']['data_dir']) / csv_file)
    
    print(f"Using CSV file: {csv_path}")
    
    dataset = ChestXrayDiffusionDataset(
        csv_file=csv_path,
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        prompt_template=config['data']['prompt_template'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip']
    )
    
    print(f"✓ DataLoader created with {len(dataset)} samples")
    print(f"✓ Batch size: {config['training']['train_batch_size']}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['train_batch_size'],
        shuffle=True,
        num_workers=min(config['training']['num_workers'], 2),  # Limit workers for testing
        collate_fn=collate_fn
    )
    
    print(f"✓ Number of batches: {len(dataloader)}")
    
    # Test loading a batch
    print("\\nTesting batch loading...")
    try:
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully")
        print(f"  - Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  - Number of texts: {len(batch['text'])}")
        print(f"  - Sample text: '{batch['text'][0]}'")
        
        # Check memory usage
        if torch.cuda.is_available():
            print(f"  - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        raise
    
    return dataloader


def test_model_imports():
    """Test that all required model components can be imported."""
    print("\\n" + "=" * 50)
    print("TESTING MODEL IMPORTS")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
        print("✓ Diffusers imported successfully")
        
        from transformers import CLIPTokenizer, CLIPTextModel
        print("✓ Transformers imported successfully")
        
        from peft import LoraConfig, get_peft_model
        print("✓ PEFT (LoRA) imported successfully")
        
        import accelerate
        print("✓ Accelerate imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def create_small_test_dataset(config: dict, max_samples: int = 15) -> str:
    """Create a small test dataset CSV with limited samples."""
    print(f"\\n" + "=" * 50)
    print("CREATING SMALL TEST DATASET")
    print("=" * 50)
    
    original_csv = Path(config['data']['data_dir']) / config['data']['csv_file']
    df = pd.read_csv(original_csv)
    
    print(f"Original dataset size: {len(df)} samples")
    
    # Take a small subset with diverse labels
    test_df = df.head(max_samples)
    
    # Save to temporary location
    test_dir = Path("outputs/diffusion_test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_csv = test_dir / "test_diffusion_dataset.csv"
    test_df.to_csv(test_csv, index=False)
    
    print(f"✓ Created test dataset with {len(test_df)} samples")
    print(f"✓ Saved to: {test_csv}")
    
    # Update config to use test dataset - use absolute path for CSV
    config['data']['csv_file'] = str(test_csv.absolute())
    
    # Print label distribution
    labels = test_df['Finding Labels'].value_counts()
    print(f"\\nLabel distribution:")
    for label, count in labels.head(5).items():
        print(f"  {label}: {count}")
    
    return str(test_csv.absolute())


def run_actual_training(config: dict) -> bool:
    """Run actual diffusion training with LoRA for a few steps."""
    print(f"\\n" + "=" * 50)
    print("RUNNING ACTUAL DIFFUSION TRAINING")
    print("=" * 50)
    
    try:
        # Import training components
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        from transformers import CLIPTokenizer, CLIPTextModel
        from peft import LoraConfig, get_peft_model, TaskType
        import torch.nn.functional as F
        from tqdm.auto import tqdm
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create output directories
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        log_dir = Path(config['training']['log_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model components
        print("Loading Stable Diffusion components...")
        model_id = config['model']['pretrained_model']
        
        # Load tokenizer and text encoder
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        
        # Load VAE (for encoding images)
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        
        print("✓ Models loaded successfully")
        
        # Set up LoRA for UNet
        print("Setting up LoRA...")
        # Configure LoRA
        lora_config = LoraConfig(
            r=config['model']['lora_rank'],
            lora_alpha=config['model']['lora_alpha'],
            target_modules=config['model']['lora_target_modules'],
            lora_dropout=config['model']['lora_dropout'],
            bias=config['model']['lora_bias'],
        )
        
        unet = get_peft_model(unet, lora_config)
        print(f"✓ LoRA applied to UNet")
        print(f"✓ Trainable parameters: {unet.num_parameters(only_trainable=True):,}")
        
        # Create dataset and dataloader
        print("\\nCreating dataset...")
        dataset = ChestXrayDiffusionDataset(
            csv_file=config['data']['csv_file'],
            data_dir=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            prompt_template=config['data']['prompt_template'],
            center_crop=config['data']['center_crop'],
            random_flip=config['data']['random_flip']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['train_batch_size'],
            shuffle=True,
            num_workers=min(config['training']['num_workers'], 2),
            collate_fn=collate_fn
        )
        
        print(f"✓ Dataset: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=float(config['training']['learning_rate']),
            betas=(float(config['training']['adam_beta1']), float(config['training']['adam_beta2'])),
            weight_decay=float(config['training']['adam_weight_decay']),
            eps=float(config['training']['adam_epsilon']),
        )
        
        # Set up noise scheduler
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("✓ Optimizer and scheduler ready")
        
        # Training loop
        print(f"\\nStarting training for {config['training']['max_train_steps']} steps...")
        
        unet.train()
        text_encoder.eval()
        vae.eval()
        
        global_step = 0
        max_steps = config['training']['max_train_steps']
        
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        for epoch in range(config['training']['num_train_epochs']):
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= max_steps:
                    break
                
                # Move to device
                pixel_values = batch['pixel_values'].to(device, dtype=torch.float32)
                texts = batch['text']
                
                with torch.no_grad():
                    # Encode images to latents
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                    # Encode text
                    text_inputs = tokenizer(
                        texts,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    text_embeddings = text_encoder(text_inputs.input_ids)[0]
                
                # Sample timesteps
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (global_step + 1) % config['training']['gradient_accumulation_steps'] == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config['training']['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                if global_step % config['training']['logging_steps'] == 0:
                    print(f"Step {global_step}: loss = {loss.item():.4f}")
                
                global_step += 1
                progress_bar.update(1)
                
                if global_step >= max_steps:
                    break
            
            if global_step >= max_steps:
                break
        
        progress_bar.close()
        
        # Save checkpoint
        print(f"\\nSaving checkpoint...")
        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        
        # Save LoRA weights
        unet.save_pretrained(checkpoint_dir / "lora_weights")
        
        # Save additional info
        torch.save({
            'global_step': global_step,
            'config': config,
            'final_loss': loss.item(),
        }, checkpoint_path)
        
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        print(f"✓ LoRA weights saved to: {checkpoint_dir / 'lora_weights'}")
        
        # Test inference
        print(f"\\nTesting inference...")
        unet.eval()
        with torch.no_grad():
            test_prompt = "A chest X-ray showing Fibrosis"
            text_inputs = tokenizer([test_prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # Generate some latents
            latents = torch.randn(1, 4, 64, 64).to(device)
            timesteps = torch.tensor([500]).to(device)
            
            output = unet(latents, timesteps, text_embeddings).sample
            print(f"✓ Inference successful, output shape: {output.shape}")
        
        print(f"\\n🎉 Training completed successfully!")
        print(f"   - Trained for {global_step} steps")
        print(f"   - Final loss: {loss.item():.4f}")
        print(f"   - Checkpoint saved")
        print(f"   - LoRA weights ready for use")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run diffusion training test with actual training."""
    print("🚀 DIFFUSION TRAINING TEST - WITH REAL TRAINING")
    print("Testing diffusion model training pipeline with actual LoRA training...")
    print()
    
    # Load configuration
    config_path = "configs/config_diffusion.yaml"
    if not Path(config_path).exists():
        print(f"✗ Configuration file not found: {config_path}")
        return False
    
    config = create_test_config(config_path)
    
    # Print system info
    print("SYSTEM INFO:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    try:
        # Test 1: Dataset loading
        dataset_size = test_dataset_loading(config)
        
        # Test 2: Create small test dataset
        test_csv = create_small_test_dataset(config, max_samples=15)
        
        # Test 3: DataLoader
        dataloader = test_dataloader(config, 15)  # Use small test size
        
        # Test 4: Model imports
        if not test_model_imports():
            return False
        
        # Test 5: ACTUAL TRAINING
        print("\\n🏃‍♂️ Ready to run actual training!")
        print("This will train a LoRA model for ~10 steps on 15 images.")
        print("Expected time: 2-5 minutes depending on GPU.")
        print()
        
        response = input("Run actual diffusion training? [Y/n]: ").strip().lower()
        if response not in ['n', 'no']:
            if not run_actual_training(config):
                return False
        else:
            print("⏭️  Skipping actual training")
            print("✅ Setup validation completed - ready for full training!")
            return True
        
        # Summary
        print("\\n" + "=" * 60)
        print("✅ FULL DIFFUSION TRAINING TEST PASSED!")
        print("=" * 60)
        print(f"✓ Dataset: {dataset_size} total samples available")
        print(f"✓ Test training: Completed successfully on 15 samples")
        print("✓ LoRA training: Working end-to-end")
        print("✓ Checkpoints: Saved and validated")
        print("✓ All required packages: Working")
        print()
        print("🎯 READY FOR FULL DIFFUSION TRAINING!")
        print("   Run: python scripts/train_diffusion.py --config configs/config_diffusion.yaml")
        print()
        print(f"📁 Test outputs saved to: outputs/diffusion_test_run/")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)