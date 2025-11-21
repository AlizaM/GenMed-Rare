"""
Train Stable Diffusion with LoRA on chest X-ray dataset.

Usage:
    python scripts/train_diffusion.py --config configs/config_diffusion.yaml
"""

import argparse
import yaml
import math
import os
from pathlib import Path
from typing import Optional
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data.diffusion_dataset import ChestXrayDiffusionDataset, collate_fn


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        tuple: (checkpoint_path, step_number) or (None, 0) if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None, 0
    
    # Look for step-based checkpoints (preferred for resuming)
    step_checkpoints = []
    epoch_checkpoints = []
    
    for path in checkpoint_path.iterdir():
        if path.is_dir():
            if path.name.startswith('checkpoint-step-'):
                try:
                    step_num = int(path.name.split('-')[-1])
                    step_checkpoints.append((step_num, path))
                except ValueError:
                    continue
            elif path.name.startswith('checkpoint-epoch-'):
                try:
                    epoch_num = int(path.name.split('-')[-1])
                    epoch_checkpoints.append((epoch_num, path))
                except ValueError:
                    continue
    
    # Prefer step-based checkpoints as they're more precise
    if step_checkpoints:
        step_checkpoints.sort(key=lambda x: x[0])
        latest_step, latest_path = step_checkpoints[-1]
        print(f"Found latest step checkpoint: {latest_path} (step {latest_step})")
        return str(latest_path), latest_step
    
    # Fall back to epoch-based checkpoints
    if epoch_checkpoints:
        epoch_checkpoints.sort(key=lambda x: x[0])
        latest_epoch, latest_path = epoch_checkpoints[-1]
        print(f"Found latest epoch checkpoint: {latest_path} (epoch {latest_epoch})")
        # For epoch checkpoints, we estimate step as 0 since we don't know exact step
        return str(latest_path), 0
    
    print("No checkpoints found")
    return None, 0


def load_checkpoint_for_resume(unet, checkpoint_path, config):
    """Load a checkpoint to resume training.
    
    Args:
        unet: The UNet model (without LoRA applied yet)
        checkpoint_path: Path to the checkpoint directory
        config: Training configuration
        
    Returns:
        tuple: (unet_with_lora, success)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")
    
    try:
        if config['model']['use_lora']:
            # For LoRA models, first set up LoRA structure, then load weights
            print("Setting up LoRA structure...")
            unet_with_lora = setup_lora(unet, config)
            
            # Load the LoRA adapter weights
            print("Loading LoRA weights...")
            from peft import PeftModel
            
            # Try to load the adapter
            try:
                # Method 1: Load via from_pretrained (most reliable)
                unet_with_lora = PeftModel.from_pretrained(unet, checkpoint_path)
                print("✓ LoRA checkpoint loaded via from_pretrained")
            except Exception as e1:
                print(f"from_pretrained failed: {e1}")
                try:
                    # Method 2: Load adapter into existing LoRA model
                    unet_with_lora.load_adapter(checkpoint_path, adapter_name="default")
                    print("✓ LoRA checkpoint loaded via load_adapter")
                except Exception as e2:
                    print(f"load_adapter failed: {e2}")
                    raise Exception(f"Failed to load LoRA checkpoint. Errors: from_pretrained={e1}, load_adapter={e2}")
            
            # CRITICAL: Ensure proper gradient settings after loading checkpoint
            # Freeze base model parameters (should not require gradients)
            for name, param in unet_with_lora.named_parameters():
                if 'lora_' not in name:  # Base model parameters
                    param.requires_grad = False
                else:  # LoRA adapter parameters
                    param.requires_grad = True
            
            # Set to training mode
            unet_with_lora.train()
            
            # Verify gradient setup
            trainable_params = sum(p.numel() for p in unet_with_lora.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet_with_lora.parameters())
            print(f"✓ Gradient setup verified: {trainable_params:,} trainable / {total_params:,} total params")
                    
        else:
            # For full model checkpoints
            model_path = checkpoint_path / "pytorch_model.bin"
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found: {model_path}")
            
            state_dict = torch.load(model_path, map_location='cpu')
            unet.load_state_dict(state_dict)
            unet_with_lora = unet
            print("✓ Full model checkpoint loaded")
        
        return unet_with_lora, True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"Checkpoint structure:")
        try:
            for item in checkpoint_path.iterdir():
                print(f"  - {item.name}")
        except:
            print("  Could not list checkpoint contents")
        return None, False


def extract_step_from_checkpoint_path(checkpoint_path):
    """Extract step number from checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        int: Step number, or 0 if cannot determine
    """
    path_str = str(checkpoint_path)
    
    # Look for step-based checkpoints first (most precise)
    if 'checkpoint-step-' in path_str:
        try:
            step_str = path_str.split('checkpoint-step-')[-1]
            # Remove any trailing path separators or additional text
            step_str = step_str.split('/')[0].split('\\')[0]
            return int(step_str)
        except ValueError:
            pass
    
    # Look for epoch-based checkpoints (less precise)
    if 'checkpoint-epoch-' in path_str:
        print("⚠️ Resuming from epoch checkpoint - step count will restart from 0")
        return 0
    
    # Default
    print("⚠️ Could not determine step from checkpoint path - starting from step 0")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to specific checkpoint directory to resume from"
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Automatically resume from the latest checkpoint in checkpoint_dir"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dirs(config):
    """Create output directories."""
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    log_dir = Path(config['training']['log_dir'])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir, log_dir


def setup_models_and_tokenizer(config):
    """Load pretrained models and tokenizer."""
    pretrained_model = config['model']['pretrained_model']
    
    print(f"Loading models from {pretrained_model}...")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model,
        subfolder="tokenizer"
    )
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model,
        subfolder="text_encoder"
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model,
        subfolder="vae"
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model,
        subfolder="unet"
    )
    
    # Freeze VAE and text encoder (we only train UNet with LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    return tokenizer, text_encoder, vae, unet


def setup_lora(unet, config):
    """Setup LoRA for UNet."""
    if not config['model']['use_lora']:
        return unet
    
    print("Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        target_modules=config['model']['lora_target_modules'],
        lora_dropout=config['model']['lora_dropout'],
        bias=config['model']['lora_bias'],
    )
    
    unet = get_peft_model(unet, lora_config)
    
    # Ensure proper gradient settings
    # Base model parameters should be frozen, LoRA parameters should be trainable
    for name, param in unet.named_parameters():
        if 'lora_' not in name:  # Base model parameters
            param.requires_grad = False
        else:  # LoRA adapter parameters
            param.requires_grad = True
    
    # Set to training mode
    unet.train()
    
    # Print trainable parameters info
    unet.print_trainable_parameters()
    
    # Verify gradient setup
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"✓ LoRA setup complete: {trainable_params:,} trainable / {total_params:,} total params")
    
    return unet


def encode_prompt(tokenizer, text_encoder, prompts, device):
    """Encode text prompts to embeddings."""
    # Tokenize
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids
    
    # Encode
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=None,
        )[0]
    
    return prompt_embeds


def train_loop(
    config,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    train_dataloader,
    accelerator,
    writer,
    checkpoint_dir,
    start_step=0,
):
    """Main training loop with enhanced metrics tracking and resume support."""
    num_epochs = config['training']['num_train_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    logging_steps = config['training']['logging_steps']
    save_steps = config['training']['save_steps']
    validation_steps = config['training']['validation_steps']
    
    global_step = start_step
    best_loss = float('inf')
    epoch_losses = []
    
    # Calculate starting epoch from global step
    # NOTE: We need to account for gradient accumulation since len(train_dataloader) 
    # gives batch steps, not training steps
    steps_per_epoch = len(train_dataloader) // config['training']['gradient_accumulation_steps']
    if len(train_dataloader) % config['training']['gradient_accumulation_steps'] != 0:
        steps_per_epoch += 1
    
    start_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    start_step_in_epoch = global_step % steps_per_epoch if steps_per_epoch > 0 else 0
    
    # Ensure we don't exceed total epochs
    start_epoch = min(start_epoch, num_epochs - 1)
    
    # Calculate total remaining steps for progress bar
    total_steps = num_epochs * steps_per_epoch
    remaining_steps = total_steps - start_step
    
    progress_bar = tqdm(
        range(remaining_steps),
        desc=f"Training (from step {start_step})",
        disable=not accelerator.is_local_main_process,
    )
    
    if start_step > 0:
        print(f"🔄 Resuming from epoch {start_epoch} (epoch {start_epoch + 1} of {num_epochs}), global step {start_step}")
        print(f"   Will skip {start_step_in_epoch} steps in current epoch")
        print(f"   Steps per epoch: {steps_per_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps if resuming within an epoch
            if epoch == start_epoch and step < start_step_in_epoch:
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = encode_prompt(
                    tokenizer,
                    text_encoder,
                    batch['text'],
                    accelerator.device
                )
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), float(config['training']['max_grad_norm']))
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Track epoch loss
                epoch_loss += loss.detach().item()
                num_batches += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    if accelerator.is_main_process and writer:
                        writer.add_scalar('train/step_loss', loss.detach().item(), global_step)
                        writer.add_scalar('train/learning_rate', lr_scheduler.get_last_lr()[0], global_step)
                        writer.add_scalar('train/epoch_progress', epoch + (step / len(train_dataloader)), global_step)
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            unet,
                            checkpoint_dir,
                            global_step,
                            config,
                            epoch=epoch
                        )
                
                # Validation
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        validate(
                            unet,
                            vae,
                            text_encoder,
                            tokenizer,
                            noise_scheduler,
                            config,
                            global_step,
                            writer
                        )
        
        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)
        
        # Log epoch-level metrics
        if accelerator.is_main_process and writer:
            writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
            writer.add_scalar('train/epoch_number', epoch, global_step)
            
            # Log loss trend (improvement over last 5 epochs)
            if len(epoch_losses) >= 5:
                recent_trend = sum(epoch_losses[-5:]) / 5
                writer.add_scalar('train/loss_trend_5epoch', recent_trend, epoch)
        
        # Save epoch checkpoint and check if it's the best
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss
            
        if accelerator.is_main_process:
            save_checkpoint(
                unet,
                checkpoint_dir,
                global_step,
                config,
                epoch=epoch,
                is_best=is_best
            )
        
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Avg Loss: {avg_epoch_loss:.4f}" + 
              (" 🌟 (Best!)" if is_best else ""))
    
    # Save final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(unet, checkpoint_dir, global_step, config, epoch=num_epochs-1, is_best=False)
    
    accelerator.end_training()


def save_checkpoint(unet, checkpoint_dir, step, config, epoch=None, is_best=False):
    """Save model checkpoint with better organization."""
    
    # Create step-based checkpoint (for resuming mid-epoch)
    step_path = checkpoint_dir / f"checkpoint-step-{step}"
    step_path.mkdir(exist_ok=True)
    
    if config['model']['use_lora']:
        unet.save_pretrained(step_path)
    else:
        unet.save_pretrained(step_path)
    
    print(f"Step checkpoint saved to {step_path}")
    
    # If epoch is provided, also save epoch-based checkpoint
    if epoch is not None:
        epoch_path = checkpoint_dir / f"checkpoint-epoch-{epoch}"
        
        # Check if epoch checkpoint already exists (to avoid overwriting when resuming)
        if epoch_path.exists():
            print(f"⚠️  Epoch checkpoint already exists: {epoch_path} - skipping to avoid overwrite")
        else:
            epoch_path.mkdir(exist_ok=True)
            
            if config['model']['use_lora']:
                unet.save_pretrained(epoch_path)
            else:
                unet.save_pretrained(epoch_path)
                
            print(f"Epoch checkpoint saved to {epoch_path}")
            
            # Keep only last 3 epoch checkpoints to save space
            cleanup_old_checkpoints(checkpoint_dir, "checkpoint-epoch-", keep_last=3)
    
    # Save as best model if specified
    if is_best:
        best_path = checkpoint_dir / "best_model"
        best_path.mkdir(exist_ok=True)
        
        if config['model']['use_lora']:
            unet.save_pretrained(best_path)
        else:
            unet.save_pretrained(best_path)
            
        print(f"Best model saved to {best_path}")


def cleanup_old_checkpoints(checkpoint_dir, prefix, keep_last=3):
    """Keep only the last N checkpoints to save disk space."""
    import re
    
    checkpoints = []
    for path in checkpoint_dir.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            # Extract epoch/step number
            match = re.search(rf'{prefix}(\d+)', path.name)
            if match:
                num = int(match.group(1))
                checkpoints.append((num, path))
    
    # Sort by number and keep only the last N
    checkpoints.sort(key=lambda x: x[0])
    
    if len(checkpoints) > keep_last:
        for _, old_path in checkpoints[:-keep_last]:
            import shutil
            shutil.rmtree(old_path)
            print(f"Removed old checkpoint: {old_path}")


@torch.no_grad()
def validate(unet, vae, text_encoder, tokenizer, noise_scheduler, config, step, writer):
    """Generate validation images and save them both to TensorBoard and as files."""
    print(f"\nGenerating validation images at step {step}...")
    
    unet.eval()
    
    validation_prompt = config['training']['validation_prompt']
    num_images = config['training']['num_validation_images']
    
    # Create validation image directory
    validation_dir = Path(config['training'].get('validation_image_dir', 
                                                 f"{config['training']['checkpoint_dir']}/../validation_images"))
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Create step-specific subdirectory
    step_dir = validation_dir / f"step_{step:06d}"
    step_dir.mkdir(exist_ok=True)
    
    # Create pipeline for generation
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline = pipeline.to(unet.device)
    
    # Generate images
    images = pipeline(
        validation_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=config['generation']['num_inference_steps'],
        guidance_scale=config['generation']['guidance_scale'],
    ).images
    
    # Save images and log to tensorboard
    for i, img in enumerate(images):
        # Save as PNG file
        img_filename = step_dir / f"validation_image_{i:02d}.png"
        img.save(img_filename)
        
        # Convert PIL Image to numpy array, then to tensor (fixes RuntimeError)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Log to tensorboard
        if writer:
            writer.add_image(f'validation/image_{i}', img_tensor, step)
    
    print(f"✓ Validation images saved to: {step_dir}")
    print(f"✓ TensorBoard: {num_images} images logged at step {step}")
    
    unet.train()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Create output directories
    checkpoint_dir, log_dir = create_output_dirs(config)
    
    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
    )
    
    # Setup models
    tokenizer, text_encoder, vae, unet = setup_models_and_tokenizer(config)
    
    # Enable memory optimizations and checkpointing BEFORE LoRA setup
    if config['hardware']['enable_attention_slicing']:
        unet.set_attention_slice("auto")
    
    if config['hardware']['enable_vae_slicing']:
        vae.enable_slicing()
    
    if config['training']['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()
    
    # Handle resume functionality
    start_step = 0
    resume_checkpoint = None
    
    if args.resume_latest:
        # Find latest checkpoint automatically
        latest_checkpoint, latest_step = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_checkpoint = latest_checkpoint
            start_step = latest_step
            print(f"🔍 Auto-resume: Found latest checkpoint at step {start_step}")
        else:
            print("🔍 Auto-resume: No checkpoints found, starting fresh training")
    elif args.resume:
        # Use specific checkpoint provided
        if Path(args.resume).exists():
            resume_checkpoint = args.resume
            start_step = extract_step_from_checkpoint_path(args.resume)
            print(f"📁 Manual resume: Using checkpoint from step {start_step}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    
    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"\n🔄 Resuming training from checkpoint: {resume_checkpoint}")
        unet, success = load_checkpoint_for_resume(unet, resume_checkpoint, config)
        if not success:
            raise RuntimeError(f"Failed to load checkpoint: {resume_checkpoint}")
    else:
        # Setup LoRA for fresh training
        print("\n🚀 Starting fresh training...")
        unet = setup_lora(unet, config)
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['model']['pretrained_model'],
        subfolder="scheduler"
    )
    
    # Create dataset
    dataset = ChestXrayDiffusionDataset(
        csv_file=str(Path(config['data']['data_dir']) / config['data']['csv_file']),
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        prompt_template=config['data']['prompt_template'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip'],
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=config['training']['train_batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
    )
    
    # Verify gradient setup before creating optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in the model! Check gradient setup.")
    
    print(f"🔧 Pre-optimizer check: {len(trainable_params)} trainable parameter tensors")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,  # Only pass trainable parameters
        lr=float(config['training']['learning_rate']),
        betas=(float(config['training']['adam_beta1']), float(config['training']['adam_beta2'])),
        weight_decay=float(config['training']['adam_weight_decay']),
        eps=float(config['training']['adam_epsilon']),
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['lr_warmup_steps'],
        num_training_steps=len(train_dataloader) * config['training']['num_train_epochs'],
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Setup tensorboard
    writer = None
    if accelerator.is_main_process and config['training']['use_tensorboard']:
        writer = SummaryWriter(log_dir)
    
    # Train
    print("\n" + "="*60)
    if resume_checkpoint:
        print(f"Resuming Training from Step {start_step:,}")
    else:
        print("Starting Training")
    print("="*60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config['training']['train_batch_size']}")
    print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['training']['train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Number of epochs: {config['training']['num_train_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Device: {accelerator.device}")
    if resume_checkpoint:
        print(f"Resume from: {resume_checkpoint}")
        print(f"Starting step: {start_step:,}")
    print("="*60 + "\n")
    
    train_loop(
        config=config,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        writer=writer,
        checkpoint_dir=checkpoint_dir,
        start_step=start_step,
    )
    
    if writer:
        writer.close()
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
