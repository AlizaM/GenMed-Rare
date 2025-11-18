# Stable Diffusion Pipeline - Quick Reference

## 📋 Complete Workflow

### 1. Setup (One-time)
```bash
# Install dependencies
pip install -r requirements_diffusion.txt

# Test dataset
python scripts/test_diffusion_dataset.py
```

### 2. Training
```bash
# Start training
python scripts/train_diffusion.py --config configs/config_diffusion.yaml

# Monitor in another terminal
tensorboard --logdir outputs/diffusion_models/sd15_lora_fibrosis/logs
```

### 3. Generation
```bash
# Generate samples
python scripts/generate_xrays.py \
  --checkpoint outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000 \
  --num-images 10
```

## 🎛️ Key Configuration Parameters

### Memory Optimization (if OOM)
```yaml
# configs/config_diffusion.yaml
training:
  train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  mixed_precision: "fp16"

hardware:
  enable_attention_slicing: true
  enable_vae_slicing: true
```

### LoRA Settings
```yaml
model:
  lora_rank: 4        # 4 (fast), 8 (balanced), 16 (better quality)
  lora_alpha: 4       # Same as rank
```

### Training Duration
```yaml
training:
  num_train_epochs: 100
  save_steps: 500
  validation_steps: 500
```

## 🔧 Common Commands

### Resume Training
```bash
python scripts/train_diffusion.py \
  --resume outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-5000
```

### Generate All Labels
```bash
python scripts/generate_xrays.py \
  --checkpoint <path> \
  --generate-all-labels \
  --num-images 5
```

### Custom Generation
```bash
python scripts/generate_xrays.py \
  --checkpoint <path> \
  --prompt "A chest X-ray with Fibrosis and Pneumonia" \
  --num-images 20 \
  --guidance-scale 9.0 \
  --seed 42
```

## 📊 Files Created

### Configuration
- `configs/config_diffusion.yaml` - Training configuration

### Code
- `src/data/diffusion_dataset.py` - Dataset class
- `scripts/train_diffusion.py` - Training script
- `scripts/generate_xrays.py` - Generation script
- `scripts/test_diffusion_dataset.py` - Dataset testing

### Documentation
- `DIFFUSION_TRAINING.md` - Comprehensive guide
- `requirements_diffusion.txt` - Dependencies

### Outputs (after training)
- `outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/` - Model weights
- `outputs/diffusion_models/sd15_lora_fibrosis/logs/` - TensorBoard logs
- `outputs/diffusion_models/sd15_lora_fibrosis/generated_samples/` - Generated images

## 💡 Tips

1. **Start small**: Test with 5-10 epochs first
2. **Monitor validation**: Check generated images during training
3. **Adjust guidance**: Higher guidance scale (9-12) for better prompt adherence
4. **Save checkpoints**: Test intermediate checkpoints for best results
5. **GPU required**: Training on CPU is impractical

## 🎯 Expected Results

- **Training**: 15-30 hours per epoch on RTX 3090
- **Best quality**: After 50-100 epochs
- **Validation images**: Appear in TensorBoard every 500 steps
- **Final checkpoint**: Saved as `checkpoint-final`

## ⚠️ Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| OOM Error | Reduce `train_batch_size: 1`, enable all memory optimizations |
| Slow training | Install xformers, increase batch size |
| Poor quality | Train longer (50+ epochs), increase LoRA rank |
| Wrong prompts | Increase guidance scale (9-12) |
| Can't find checkpoint | Check path, use absolute path if needed |

## 📚 For More Details

See `DIFFUSION_TRAINING.md` for:
- Detailed setup instructions
- Advanced configuration options
- Integration with classifier training
- Evaluation metrics
- References
