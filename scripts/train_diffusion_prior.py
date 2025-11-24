#!/usr/bin/env python3
"""
Prior-based Diffusion Training Script

Trains a diffusion model using target pathology images paired with healthy "prior" images.
Each target image is repeated multiple times with different healthy priors.
"""

import argparse
import logging
import math
import os
import sys
import random
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import yaml

from src.data.prior_dataset import PriorBasedDiffusionDataset, collate_fn


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prior-based diffusion training")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint"
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        help="Specific step to resume from"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory."""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for step-based checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
    return checkpoints[-1]


def save_model_card(repo_id, images=None, base_model=None, train_text_encoder=False, prompt=None):
    """Save model card for the trained model."""
    img_str = ""
    for i, image in enumerate(images):
        image.save(f"image_{i}.png")
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml_content = f"""
---
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
"""
    model_card = f"""
# Prior-based Fibrosis LoRA

<Gallery />

## Model description

These are LoRA adaption weights for {base_model}. The model was trained using prior-based learning where each fibrosis image was paired with healthy chest X-ray "priors" to learn the pathology-specific features.

## Training procedure

- **Target pathology**: Fibrosis
- **Training approach**: Prior-based learning
- **Base model**: {base_model}
- **Training prompts**: 
  - Target: "a chest x-ray with fibrosis"
  - Prior: "a chest x-ray"

## Trigger words

You should use "a chest x-ray with fibrosis" to trigger the image generation.

## Download model

[Download the LoRA weights](./pytorch_lora_weights.safetensors)

{img_str}
"""
    with open("README.md", "w") as f:
        f.write(yaml_content + model_card)


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract config sections
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    paths_config = config.get('paths', {})
    experiment_config = config.get('experiment', {})
    logging_config = config.get('logging', {})
    
    # Setup accelerator with project configuration
    accelerator_project_config = ProjectConfiguration(
        project_dir=paths_config.get('output_dir', 'outputs/diffusion_prior'),
        logging_dir=paths_config.get('logging_dir', 'outputs/diffusion_prior/logs')
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        mixed_precision=training_config.get('mixed_precision', 'fp16'),
        log_with=logging_config.get('report_to', 'tensorboard'),
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, logging_config.get('log_level', 'INFO').upper()),
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set random seed
    if config.get('seed') is not None:
        set_seed(config['seed'])
    
    # Create output directories
    output_dir = Path(paths_config.get('output_dir', 'outputs/diffusion_prior'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model components
    logger.info(f"Loading models from {model_config['pretrained_model']}...")
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        model_config['pretrained_model'],
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_config['pretrained_model'],
        subfolder="text_encoder"
    )
    
    # Load feature extractor
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_config['pretrained_model'],
        subfolder="feature_extractor"
    )
    
    # Load VAE
    vae_path = model_config.get('vae', {}).get('path')
    if vae_path:
        logger.info(f"Loading custom VAE from {vae_path}...")
        vae = AutoencoderKL.from_pretrained(vae_path)
    else:
        vae = AutoencoderKL.from_pretrained(
            model_config['pretrained_model'],
            subfolder="vae"
        )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_config['pretrained_model'],
        subfolder="unet"
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=training_config.get('lora', {}).get('rank', 64),
        lora_alpha=training_config.get('lora', {}).get('alpha', 64),
        lora_dropout=training_config.get('lora', {}).get('dropout', 0.1),
        target_modules=training_config.get('lora', {}).get('target_modules', 
                                                          ["to_k", "to_q", "to_v", "to_out.0"]),
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Freeze components that shouldn't be trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing
    if training_config.get('gradient_checkpointing', False):
        unet.enable_gradient_checkpointing()
    
    # Setup noise scheduler  
    noise_scheduler = DDPMScheduler(
        beta_start=training_config.get('beta_start', 0.00085),
        beta_end=training_config.get('beta_end', 0.012),
        beta_schedule=training_config.get('beta_schedule', 'scaled_linear'),
        num_train_timesteps=training_config.get('num_train_timesteps', 1000),
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=training_config.get('learning_rate', 1e-4),
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Create dataset
    logger.info("Creating prior-based dataset...")
    train_dataset = PriorBasedDiffusionDataset(
        target_images_dir=training_config['target_images_dir'],
        target_images_csv=training_config['target_images_csv'],
        prior_images_dir=training_config['prior_images_dir'],
        prior_images_csv=training_config['prior_images_csv'],
        target_prompt=training_config['target_prompt'],
        prior_prompt=training_config['prior_prompt'],
        repeats_per_target=training_config.get('repeats_per_target', 10),
        resolution=training_config.get('resolution', 512),
        center_crop=training_config.get('center_crop', True),
        random_flip=training_config.get('random_flip', False),
        tokenizer=tokenizer,
        seed=config.get('seed', 42)
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 4),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_config.get('dataloader_num_workers', 2),
        pin_memory=training_config.get('pin_memory', True),
    )
    
    # Calculate training parameters
    num_epochs = training_config.get('num_epochs', 20)
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_train_steps,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Resume from checkpoint if requested
    starting_epoch = 0
    global_step = 0
    
    if args.resume_latest:
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            accelerator.load_state(str(checkpoint_path))
            global_step = int(checkpoint_path.name.split('-')[1])
            starting_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("No checkpoint found, starting from scratch")
    elif args.resume_step:
        checkpoint_path = output_dir / f"checkpoint-{args.resume_step}"
        if checkpoint_path.exists():
            logger.info(f"Resuming from step {args.resume_step}")
            accelerator.load_state(str(checkpoint_path))
            global_step = args.resume_step
            starting_epoch = global_step // num_update_steps_per_epoch
    
    # Initialize trackers for logging (required for TensorBoard)
    tracker_config = {
        "experiment_name": experiment_config.get('name', 'fibrosis_prior_lora'),
        "tags": experiment_config.get('tags', []),
    }
    accelerator.init_trackers(
        project_name=experiment_config.get('name', 'fibrosis_prior_lora'),
        config=tracker_config,
        init_kwargs={"tensorboard": {"flush_secs": 30}}
    )
    
    # Log training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_config.get('batch_size', 4)}")
    logger.info(f"  Total train batch size = {training_config.get('batch_size', 4) * accelerator.num_processes * gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Starting from epoch = {starting_epoch}")
    logger.info(f"  Starting from global step = {global_step}")
    
    # Training loop
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(starting_epoch, num_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    # For prior-based training, we use the target images as the main training target
                    target_latents = vae.encode(batch["target_images"]).latent_dist.sample()
                    target_latents = target_latents * vae.config.scaling_factor
                    
                    # Encode text prompts
                    target_prompt_embeds = text_encoder(batch["target_prompt_ids"])[0]
                
                # Sample noise and timesteps
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (target_latents.shape[0],), device=target_latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                
                # Predict noise
                model_pred = unet(noisy_latents, timesteps, target_prompt_embeds).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(training_config.get('batch_size', 4))).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                logs = {
                    "loss": train_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint FIRST (before validation that might crash)
                # Only save LoRA weights (not full model) to save disk space
                if global_step % training_config.get('save_steps', 250) == 0:
                    if accelerator.is_main_process:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save only LoRA adapter weights (~5-50 MB instead of 3+ GB)
                        unet_unwrapped = accelerator.unwrap_model(unet)
                        unet_unwrapped.save_pretrained(checkpoint_dir / "lora_weights")
                        
                        # Optionally save training state (optimizer, scheduler, step)
                        # This is much smaller than full model
                        training_state = {
                            'global_step': global_step,
                            'epoch': epoch,
                            'random_state': torch.get_rng_state(),
                        }
                        torch.save(training_state, checkpoint_dir / "training_state.pt")
                        
                        logger.info(f"✅ Saved LoRA checkpoint at step {global_step}")
                
                # Generate validation images AFTER checkpoint is saved
                if global_step % training_config.get('validation_steps', 1000) == 0:
                    if accelerator.is_main_process:
                        logger.info("🎨 Generating validation images...")
                        
                        try:
                            # Create validation pipeline
                            from diffusers import StableDiffusionPipeline
                            
                            # Get the underlying model for pipeline
                            unet_for_pipeline = accelerator.unwrap_model(unet)
                            
                            pipeline = StableDiffusionPipeline(
                                vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                unet=unet_for_pipeline,
                                scheduler=noise_scheduler,
                                feature_extractor=feature_extractor,
                                safety_checker=None,
                                requires_safety_checker=False,
                            )
                            
                            # Generate images
                            validation_prompt = training_config.get('validation_prompt', 'a chest x-ray with fibrosis')
                            num_validation_images = training_config.get('num_validation_images', 4)
                            
                            generator = torch.Generator(device=accelerator.device).manual_seed(config.get('seed', 42))
                            
                            validation_images = []
                            for i in range(num_validation_images):
                                with torch.autocast("cuda"):
                                    image = pipeline(
                                        validation_prompt,
                                        num_inference_steps=50,
                                        generator=generator,
                                        height=512,
                                        width=512,
                                    ).images[0]
                                validation_images.append(image)
                            
                            # Save validation images
                            validation_dir = output_dir / f"validation_step_{global_step}"
                            validation_dir.mkdir(exist_ok=True)
                            
                            for i, image in enumerate(validation_images):
                                image.save(validation_dir / f"image_{i}.png")
                            
                            # Log validation images
                            for tracker in accelerator.trackers:
                                if tracker.name == "tensorboard":
                                    np_images = np.stack([np.asarray(img) for img in validation_images])
                                    tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
                            
                            logger.info(f"✅ Saved {len(validation_images)} validation images")
                            
                            # Clean up pipeline to free memory
                            del pipeline
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            logger.error(f"❌ Validation image generation failed: {e}")
                            logger.info("🔄 Continuing training without validation images...")
            
            if global_step >= max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the LoRA weights
        unet_unwrapped = accelerator.unwrap_model(unet)
        unet_unwrapped.save_pretrained(output_dir / "lora_weights")
        
        # Save model card
        save_model_card(
            repo_id=output_dir / "model_card",
            base_model=model_config['pretrained_model'],
            prompt=training_config.get('validation_prompt', 'a chest x-ray with fibrosis'),
        )
        
        logger.info(f"Training completed! Model saved to {output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()