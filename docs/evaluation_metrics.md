# Evaluation Metrics for Diffusion-Generated Medical Images

Comprehensive metrics for assessing diffusion model quality on chest X-ray generation tasks.

## Per-Checkpoint Metrics

Computed individually for each checkpoint to enable direct comparison.

### 1. Novelty (SSIM/Correlation)

Measures generated image novelty by computing similarity to nearest training sample.

**Implementation**: For each generated image, compute SSIM (Structural Similarity Index) or pixel correlation against all training images. Report nearest-neighbor statistics.

**Key Statistics**:
- `max_similarity`: Maximum similarity (identifies potential memorization)
- `p99_similarity`: 99th percentile (robust outlier detection)
- `mean_similarity`, `median_similarity`: Central tendency measures

**Thresholds**:
- SSIM > 0.9: Potential training set memorization
- SSIM 0.7-0.9: High similarity, minimal novelty
- SSIM < 0.7: Novel generation

---

### 2. Pathology Confidence

Classification confidence for target pathology using TorchXRayVision DenseNet-121 (trained on NIH ChestX-ray14 and other chest X-ray datasets).

**Implementation**: Pass generated images through pre-trained pathology classifier. Extract logits for target class (e.g., Pneumonia, Fibrosis).

**Key Statistics**:
- `mean_confidence`: Average classification probability
- `median_confidence`: Median classification probability
- `std_confidence`: Probability distribution spread

**Thresholds**:
- Confidence > 0.7: Strong pathology presence
- Confidence 0.5-0.7: Moderate presence
- Confidence < 0.5: Weak or absent pathology

**Examples**:
- Pneumonia generator: Verify images classified as Pneumonia, not generic X-rays
- Fibrosis generator: Confirm presence of fibrotic patterns

---

### 3. BioViL Score

Medical-domain text-image alignment using Microsoft BiomedVLP-CXR-BERT-specialized, a vision-language model pre-trained on MIMIC-CXR radiology reports.

**Implementation**: Compute cosine similarity between image embeddings and text prompt embeddings using BioViL encoder.

**Key Statistics**:
- `mean_score`: Average alignment score
- `median_score`: Median alignment score

**Note**: BioViL is trained on medical terminology and radiology-specific concepts, providing more accurate semantic alignment than general-domain models for chest X-ray evaluation.

---

### 4. Diversity (Standard Deviation)

Quantifies intra-batch diversity using pathology probability distributions from TorchXRayVision.

**Implementation**: 
1. Extract pathology probabilities for all generated images (14-class vector per image)
2. Compute standard deviation across batch for each pathology
3. Average standard deviations to obtain overall diversity score

**Key Statistics**:
- `overall_diversity`: Mean standard deviation across pathologies
- `mean_std`: Average per-pathology standard deviation
- `pathology_stds`: Individual standard deviations for each pathology class

**Thresholds**:
- Diversity > 0.2: Adequate variability
- Diversity 0.1-0.2: Limited variability
- Diversity < 0.1: Potential mode collapse

---

## Distribution-Level Metrics

Computed across entire generated dataset. Higher computational cost; typically reserved for final evaluation.

### 5. FMD (Fréchet MedicalNet Distance)

Fréchet distance between generated and real image distributions in TorchXRayVision feature space. Medical-domain analog of FID (Fréchet Inception Distance).

**Implementation**:
- Extract 1024-dimensional features from penultimate layer of TorchXRayVision DenseNet
- Compute mean vectors (μ) and covariance matrices (Σ) for real and generated sets
- Calculate Fréchet distance:

```
FMD = ||μ_gen - μ_real||² + Tr(Σ_gen + Σ_real - 2√(Σ_gen·Σ_real))
```

**Thresholds**:
- FMD < 10: Excellent distributional match
- FMD 10-50: Acceptable match
- FMD > 50: Poor match

**Advantages**: 
- Domain-specific feature extractor (medical imaging)
- Captures distributional properties beyond individual image metrics
- Established metric for generative model evaluation

**Computational Cost**: O(n) feature extraction + O(d²) covariance computation (d=1024)

---

### 6. t-SNE Overlap

Distribution overlap visualization and quantification using t-SNE projection of TorchXRayVision features.

**Implementation**:
1. Extract TorchXRayVision features for real and generated images
2. Apply t-SNE dimensionality reduction to 2D (perplexity=30, n_iter=1000)
3. Compute overlap score: fraction of generated points within distance threshold of real points
4. Generate scatter plot (real: blue, generated: red)

**Key Statistics**:
- `overlap_score`: Fraction of generated samples near real distribution
- `mean_distance`: Average distance to nearest real sample
- `median_distance`: Median distance to nearest real sample

**Thresholds**:
- Overlap > 0.7: High distributional similarity
- Overlap 0.4-0.7: Moderate similarity
- Overlap < 0.4: Distributional mismatch

**Use Cases**:
- Visual inspection for mode collapse
- Identify clustering in limited feature regions
- Complement FMD with interpretable visualization

**Computational Cost**: O(n² log n) for t-SNE with n samples (very expensive for large datasets)

---

## Configuration

Metric selection via preset flags in evaluation scripts:

**Checkpoint Preset** (default, fast):
```bash
--preset checkpoint  # novelty, pathology, biovil, diversity
```

**Diversity Preset** (mode collapse detection):
```bash
--preset diversity  # diversity, pixel_variance, feature_dispersion, self_similarity
```

**Full Preset** (comprehensive evaluation):
```bash
--preset full  # all 9 metrics including FMD and t-SNE
```

---

## Metric Summary

| Metric | Scope | Complexity | Measures | Optimal Direction |
|--------|-------|------------|----------|-------------------|
| SSIM (Novelty) | Per-checkpoint | O(n·m) | Training memorization | Lower |
| Pathology Confidence | Per-checkpoint | O(n) | Medical accuracy | Higher |
| BioViL Score | Per-checkpoint | O(n) | Semantic alignment | Higher |
| Diversity (Std Dev) | Per-checkpoint | O(n) | Intra-batch variability | Higher |
| FMD | Distribution | O(n) | Distributional match | Lower |
| t-SNE Overlap | Distribution | O(n² log n) | Visual distribution match | Higher |

---

## Recommended Evaluation Protocol

### Stage 1: Checkpoint Selection
```bash
python scripts/evaluate_checkpoints.py \
    --config configs/config_eval_fibrosis.yaml \
    --preset checkpoint \
    --num-images 100
```

Evaluate novelty, pathology confidence, BioViL, and diversity on 100 images/checkpoint. Select best checkpoint based on combined ranking.

### Stage 2: Comprehensive Evaluation
```bash
python scripts/evaluate_diffusion_generation.py \
    --generated-dir outputs/checkpoint-6500_Fibrosis_images \
    --real-dir data/pure_class_folders/fibrosis \
    --label Fibrosis \
    --preset full \
    --num-images 1000
```

Run full metric suite (including FMD and t-SNE) on best checkpoint with 2000-4000 images.

---

## Dependencies

**Core Requirements**:
```bash
pip install torch torchvision transformers
pip install scikit-image scikit-learn scipy
pip install torchxrayvision
```

**Model Downloads** (automatic on first use):
- TorchXRayVision: `densenet121-res224-all` (~30 MB)
- BioViL: `microsoft/BiomedVLP-CXR-BERT-specialized` (~500 MB)

---

## Example Output

```
Checkpoint: checkpoint-6500

Per-Checkpoint Metrics:
  Novelty (P99 SSIM): 0.7234
  Pathology Confidence (Pneumonia): 0.812
  BioViL Score: 0.734
  Diversity: 0.234

Distribution Metrics:
  FMD: 23.4
  t-SNE Overlap: 0.68
```

---

## References

- **SSIM (Novelty)**: Wang et al. "Image quality assessment: from error visibility to structural similarity" IEEE TIP (2004)
- **TorchXRayVision (Pathology, Diversity, FMD features)**: Cohen et al. "TorchXRayVision: A library of chest X-ray datasets and models" arXiv:2111.00595 (2021)
- **BioViL (Medical text-image alignment)**: Boecking et al. "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing" ECCV (2022)
- **FID/FMD (Fréchet distance)**: Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" NeurIPS (2017)
- **t-SNE (Distribution visualization)**: van der Maaten & Hinton "Visualizing Data using t-SNE" JMLR (2008)

## Additional Diversity Metrics

Extended diversity analysis metrics for in-depth mode collapse detection.

### Pixel-Level Variance

Variance computed directly in pixel space across generated batch.

**Implementation**: Stack all generated images, compute variance per pixel, average across spatial dimensions.

**Use Case**: Detect complete mode collapse where all pixels are identical.

---

### Feature Dispersion

Covariance matrix analysis in TorchXRayVision feature space.

**Implementation**: 
- Extract features for all generated images
- Compute covariance matrix
- Report determinant (volume) and trace (spread)

**Use Case**: Quantify feature space coverage. Low determinant indicates collapsed distribution.

---

### Self-Similarity (Pairwise SSIM)

Average SSIM between randomly sampled pairs within generated batch.

**Implementation**: Sample 100 random pairs, compute pairwise SSIM, report mean.

**Thresholds**:
- Mean SSIM > 0.9: High internal similarity (potential collapse)
- Mean SSIM 0.7-0.9: Moderate similarity
- Mean SSIM < 0.7: High diversity

**Use Case**: Direct measure of intra-batch redundancy.
