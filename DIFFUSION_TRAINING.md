# Stable Diffusion Fine-tuning for Medical X-ray Generation

This pipeline fine-tunes Stable Diffusion 1.5 with LoRA on chest X-ray images to generate synthetic medical images conditioned on pathology labels.

## Overview

- **Model**: Stable Diffusion 1.5 + LoRA
- **Method**: Text-to-image generation with medical labels as prompts
- **Dataset**: Balanced chest X-ray dataset (10,541 images)
- **Prompts**: `"A chest X-ray with {labels}"` (e.g., "A chest X-ray with Fibrosis")

## Setup

### 1. Install Dependencies

```bash
# Activate your environment
source .venv/bin/activate

# Install diffusion dependencies
pip install -r requirements_diffusion.txt
```

**Optional optimizations** (GPU only):
```bash
# For 8-bit Adam optimizer (saves memory)
pip install bitsandbytes

# For faster attention (requires compatible PyTorch + CUDA)
pip install xformers
```

### 2. Verify Dataset

Test that the dataset loads correctly:
```bash
python scripts/test_diffusion_dataset.py
```

Expected output:
- Dataset size: 10,541 samples
- Image shape: (3, 512, 512)
- Image range: [-1, 1]
- Sample prompts shown

## Training

### Quick Start

```bash
python scripts/train_diffusion.py --config configs/config_diffusion.yaml
```

### Configuration

Edit `configs/config_diffusion.yaml` to adjust:

**Memory Settings** (if you have OOM errors):
```yaml
training:
  train_batch_size: 1          # Reduce if OOM
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
  gradient_checkpointing: true    # Save memory
  mixed_precision: "fp16"         # Use fp16 or bf16
  use_8bit_adam: true            # Requires bitsandbytes

hardware:
  enable_attention_slicing: true  # Reduce memory
  enable_vae_slicing: true        # Reduce VAE memory
```

**LoRA Settings** (trade-off: model capacity vs memory):
```yaml
model:
  lora_rank: 4        # Lower = less memory, faster (4, 8, 16, 32)
  lora_alpha: 4       # Typically same as rank
```

**Training Duration**:
```yaml
training:
  num_train_epochs: 100        # Number of epochs
  save_steps: 500              # Save checkpoint every N steps
  validation_steps: 500        # Generate validation images every N steps
```

### Resume Training

```bash
python scripts/train_diffusion.py --resume outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir outputs/diffusion_models/sd15_lora_fibrosis/logs

# Open browser at http://localhost:6006
```

**What to monitor**:
- `train/loss`: Should decrease over time
- `train/lr`: Learning rate schedule
- `validation/image_*`: Generated validation images

## Generation

### Generate from Checkpoint

```bash
# Generate 10 images with default prompt
python scripts/generate_xrays.py \
  --checkpoint outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000 \
  --num-images 10
```

### Generate with Custom Prompt

```bash
python scripts/generate_xrays.py \
  --checkpoint outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000 \
  --prompt "A chest X-ray with Fibrosis" \
  --num-images 20
```

### Generate All Label Combinations

Generate images for all unique label combinations in the dataset:

```bash
python scripts/generate_xrays.py \
  --checkpoint outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000 \
  --generate-all-labels \
  --num-images 5
```

### Generation Parameters

```bash
python scripts/generate_xrays.py \
  --checkpoint <path> \
  --prompt "A chest X-ray with Fibrosis and Pneumonia" \
  --num-images 10 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --seed 42
```

**Parameters**:
- `--num-inference-steps`: More steps = better quality but slower (default: 50)
- `--guidance-scale`: Higher = more prompt adherence (default: 7.5)
- `--seed`: For reproducible generation

## Expected Training Time

**On GPU (e.g., RTX 3090, 24GB VRAM)**:
- Batch size 1, gradient accumulation 4
- ~5-10 seconds per step
- ~10,541 steps per epoch
- **~15-30 hours per epoch**

**Recommendations**:
- Start with 10-20 epochs, check validation images
- Best results typically after 50-100 epochs
- Save checkpoints every 500-1000 steps to test intermediate results

## Memory Requirements

**Minimum** (with all optimizations):
- ~8-10 GB VRAM (GPU)
- batch_size=1, gradient_checkpointing, fp16, attention_slicing

**Recommended**:
- ~12-16 GB VRAM
- batch_size=2-4, gradient_checkpointing, fp16

**Ideal**:
- ~24 GB VRAM
- batch_size=4-8, mixed_precision

## Output Structure

```
outputs/diffusion_models/sd15_lora_fibrosis/
├── checkpoints/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-5000/
│   └── checkpoint-final/
├── logs/
│   └── events.out.tfevents.*  # TensorBoard logs
└── generated_samples/
    ├── Fibrosis_0000.png
    ├── Fibrosis_0001.png
    └── generation_results.csv
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `train_batch_size: 1`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Use mixed precision: `mixed_precision: "fp16"`
4. Enable memory optimizations:
   ```yaml
   hardware:
     enable_attention_slicing: true
     enable_vae_slicing: true
   ```
5. Reduce LoRA rank: `lora_rank: 4` (instead of 8 or 16)

### Slow Training

1. Increase batch size (if memory allows)
2. Install xformers: `pip install xformers`
3. Enable xformers: `enable_xformers: true`
4. Use bf16 if supported: `mixed_precision: "bf16"`

### Poor Image Quality

1. Train for more epochs (50-100+)
2. Check validation images during training
3. Increase LoRA rank: `lora_rank: 8` or `16`
4. Adjust learning rate: try `5e-5` or `1e-5`
5. Increase guidance scale during generation: `--guidance-scale 9.0`

### Images Don't Match Prompts

1. Train longer (prompt conditioning improves over time)
2. Increase guidance scale: `--guidance-scale 9.0` to `12.0`
3. Verify prompts in dataset: run `test_diffusion_dataset.py`

## Next Steps

After training:
1. **Evaluate quality**: Generate samples for each label
2. **Classifier evaluation**: Use generated images to augment classifier training
3. **FID score**: Measure distribution similarity to real images
4. **Medical expert review**: Validate clinical realism

## Integration with Classifier Training

Once you have generated synthetic images:

1. **Generate dataset**: Use `--generate-all-labels` to create balanced synthetic set
2. **Mix with real data**: Combine synthetic + real at different ratios (10%, 25%, 50%)
3. **Train classifier**: Use augmented dataset for rare disease classification
4. **Compare performance**: Baseline (0%) vs augmented (10%, 25%, 50%)

## References

- Stable Diffusion: https://github.com/Stability-AI/stablediffusion
- Diffusers: https://huggingface.co/docs/diffusers
- LoRA: https://arxiv.org/abs/2106.09685
- PEFT: https://github.com/huggingface/peft
