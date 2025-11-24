"""
Generate synthetic chest X-rays using fine-tuned Stable Diffusion LoRA model.

Usage:
    # Generate from a specific checkpoint
    python scripts/generate_xrays.py --checkpoint outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000
    
    # Generate with custom prompts
    python scripts/generate_xrays.py --checkpoint <path> --prompt "A chest X-ray with Fibrosis"
    
    # Generate multiple images per prompt
    python scripts/generate_xrays.py --checkpoint <path> --num-images 10
    
    # Generate from all labels in dataset
    python scripts/generate_xrays.py --checkpoint <path> --generate-all-labels
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Optional
import pandas as pd

import torch
from tqdm import tqdm
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.diffusion_utils import load_pipeline, generate_images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic X-rays")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (default: use validation prompt from config)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--generate-all-labels",
        action="store_true",
        help="Generate images for all unique labels in the dataset"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="data/diffusion_data/diffusion_dataset_balanced.csv",
        help="CSV file to extract labels from (for --generate-all-labels)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_unique_labels(csv_file: str, prompt_template: str) -> List[str]:
    """
    Extract unique label combinations from dataset CSV.
    
    Returns list of prompts like:
    - "A chest X-ray with Fibrosis"
    - "A chest X-ray with Fibrosis and Pneumonia"
    - "A chest X-ray with Effusion"
    """
    df = pd.read_csv(csv_file)
    
    prompts = []
    unique_labels = df['Finding Labels'].unique()
    
    for labels in unique_labels:
        # Convert pipe-separated labels to formatted text
        if '|' in labels:
            label_list = labels.split('|')
            labels_formatted = ' and '.join(label_list)
        else:
            labels_formatted = labels
        
        prompt = prompt_template.format(labels=labels_formatted)
        prompts.append(prompt)
    
    return sorted(prompts)


def setup_pipeline(checkpoint_path: str, config: dict):
    """
    Load Stable Diffusion pipeline with LoRA weights.
    
    Uses load_pipeline() from diffusion_utils which handles both:
    - New adapter format (adapter_config.json + adapter_model.safetensors)
    - Old Accelerate format (model.safetensors with PEFT structure)
    """
    pretrained_model = config['model']['pretrained_model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading pipeline from checkpoint: {checkpoint_path}")
    
    pipeline = load_pipeline(
        checkpoint_path=checkpoint_path,
        pretrained_model=pretrained_model,
        enable_attention_slicing=True,  # Always enable for memory efficiency
        enable_vae_slicing=True,  # Always enable for memory efficiency
        device=device
    )
    
    return pipeline


def generate_images_for_prompt(
    pipeline,
    prompt: str,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    seed: Optional[int] = None,
):
    """
    Generate images from a single prompt.
    
    Wrapper around diffusion_utils.generate_images() for backward compatibility.
    """
    print(f"\nGenerating {num_images} images...")
    print(f"Prompt: {prompt}")
    
    images = generate_images(
        pipeline=pipeline,
        prompt=prompt,
        num_images=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
        return_numpy=False  # Return PIL Images
    )
    
    return images


def save_images(images: List[Image.Image], output_dir: Path, prompt: str, start_idx: int = 0):
    """
    Save generated images to disk.
    
    Args:
        images: List of PIL Images to save
        output_dir: Directory to save images
        prompt: Text prompt used for generation (used in filename)
        start_idx: Starting index for numbering files
        
    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename from prompt
    safe_prompt = prompt.replace("A chest X-ray with ", "").replace("a chest x-ray with ", "")
    safe_prompt = safe_prompt.replace(" ", "_").replace("/", "-")
    
    saved_paths = []
    for i, img in enumerate(images):
        filename = f"{safe_prompt}_{start_idx + i:04d}.png"
        filepath = output_dir / filename
        img.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def save_images_with_seeds(images: List[Image.Image], output_dir: Path, base_seed: int):
    """
    Save generated images with seed numbers in filenames.
    
    Args:
        images: List of PIL Images to save
        output_dir: Directory to save images
        base_seed: Base seed used for generation (each image is base_seed + i)
        
    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for i, img in enumerate(images):
        seed = base_seed + i
        filename = f"image_{i:02d}_seed_{seed}.png"
        filepath = output_dir / filename
        img.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Fallback: use config or default
        output_dir = Path(config.get('generation', {}).get('output_dir', 'outputs/generated_images'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Setup pipeline
    pipeline = setup_pipeline(args.checkpoint, config)
    
    # Get negative prompt
    negative_prompt = config.get('generation', {}).get('negative_prompt', 
                                                       'blurry, low quality, distorted, artifacts')
    
    # Determine prompts to generate
    if args.generate_all_labels:
        print("\nExtracting unique labels from dataset...")
        prompt_template = config.get('data', {}).get('prompt_template', 'A chest X-ray with {labels}')
        prompts = extract_unique_labels(args.csv_file, prompt_template)
        print(f"Found {len(prompts)} unique label combinations")
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Fallback: use validation prompt from training config
        prompts = [config.get('training', {}).get('validation_prompt', 'a chest x-ray')]
    
    # Generate images for each prompt
    print("\n" + "="*60)
    print("Starting Generation")
    print("="*60)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Images per prompt: {args.num_images}")
    print(f"Total images: {len(prompts) * args.num_images}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print("="*60 + "\n")
    
    all_results = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        # Generate images
        images = generate_images_for_prompt(
            pipeline=pipeline,
            prompt=prompt,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt,
            seed=args.seed + prompt_idx if args.seed else None,
        )
        
        # Save images
        saved_paths = save_images(images, output_dir, prompt, start_idx=0)
        
        # Track results
        for path in saved_paths:
            all_results.append({
                'prompt': prompt,
                'image_path': str(path),
                'guidance_scale': args.guidance_scale,
                'num_inference_steps': args.num_inference_steps,
            })
        
        print(f"  ✓ Saved {len(saved_paths)} images")
    
    # Save results CSV
    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / "generation_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    print("\n" + "="*60)
    print("✓ Generation Complete!")
    print("="*60)
    print(f"Total images generated: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print(f"Results CSV: {results_csv}")
    
    # Print summary by prompt
    print("\nGeneration Summary:")
    for prompt in prompts:
        count = sum(1 for r in all_results if r['prompt'] == prompt)
        print(f"  {prompt}: {count} images")


if __name__ == "__main__":
    main()
