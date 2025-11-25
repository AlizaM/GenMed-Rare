"""
Evaluation metrics for diffusion model outputs.

This module provides:
- SSIM (Structural Similarity Index)
- Pixel correlation
- CLIP text-image alignment scores
- Novelty metrics (nearest neighbor analysis)
- Pathology confidence (TorchXRayVision)
- BioViL semantic similarity
- Diversity metrics (std dev of probabilities)
- FMD (Fréchet MedicalNet Distance)
- t-SNE overlap analysis
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image as numpy array [0, 255]
        img2: Second image as numpy array [0, 255]
    
    Returns:
        SSIM score in range [0, 1]
    """
    from skimage.transform import resize
    
    # Resize images to same size if different (before grayscale conversion)
    if img1.shape != img2.shape:
        # Resize img2 to match img1
        img2 = resize(img2, img1.shape, anti_aliasing=True, preserve_range=True)
    
    # Convert to grayscale for SSIM
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
    else:
        img1_gray = img1
    
    if len(img2.shape) == 3:
        img2_gray = np.mean(img2, axis=2)
    else:
        img2_gray = img2
    
    # Normalize to [0, 1]
    img1_gray = img1_gray / 255.0
    img2_gray = img2_gray / 255.0
    
    return ssim(img1_gray, img2_gray, data_range=1.0)


def compute_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute pixel-wise Pearson correlation between two images.
    
    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
    
    Returns:
        Correlation coefficient in range [-1, 1]
    """
    # Resize images to same size if different
    if img1.shape != img2.shape:
        from skimage.transform import resize
        # Resize img2 to match img1
        img2 = resize(img2, img1.shape, anti_aliasing=True, preserve_range=True)
    
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr


def find_nearest_neighbor(
    generated_img: np.ndarray,
    training_images: List[np.ndarray],
    metric: str = "ssim"
) -> Tuple[int, float, np.ndarray]:
    """
    Find nearest neighbor in training set.
    
    Args:
        generated_img: Generated image as numpy array
        training_images: List of training images as numpy arrays
        metric: Similarity metric ("ssim" or "correlation")
    
    Returns:
        Tuple of (index, similarity_score, training_image)
    """
    best_idx = -1
    best_score = -np.inf
    
    for idx, train_img in enumerate(training_images):
        if metric == "ssim":
            score = compute_ssim(generated_img, train_img)
        elif metric == "correlation":
            score = compute_correlation(generated_img, train_img)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx, best_score, training_images[best_idx]


def compute_novelty_metrics(
    generated_images: List[np.ndarray],
    training_images: List[np.ndarray],
    metric: str = "ssim",
    show_progress: bool = True
) -> Dict[str, any]:
    """
    Compute novelty metrics for generated images.
    
    Lower similarity scores indicate higher novelty (less copying from training set).
    
    Args:
        generated_images: List of generated images as numpy arrays
        training_images: List of training images as numpy arrays
        metric: Similarity metric ("ssim" or "correlation")
        show_progress: Show progress bar
    
    Returns:
        Dictionary with:
            - max_similarity: Maximum similarity to any training image
            - p99_similarity: 99th percentile similarity
            - p95_similarity: 95th percentile similarity
            - mean_similarity: Mean similarity
            - median_similarity: Median similarity
            - min_similarity: Minimum similarity
            - nn_indices: List of nearest neighbor indices
            - nn_scores: List of similarity scores
    """
    nn_scores = []
    nn_indices = []
    
    iterator = tqdm(generated_images, desc=f"Computing {metric} novelty") if show_progress else generated_images
    
    for gen_img in iterator:
        nn_idx, nn_score, _ = find_nearest_neighbor(gen_img, training_images, metric)
        nn_scores.append(nn_score)
        nn_indices.append(nn_idx)
    
    nn_scores = np.array(nn_scores)
    
    return {
        'max_similarity': float(np.max(nn_scores)),
        'p99_similarity': float(np.percentile(nn_scores, 99)),
        'p95_similarity': float(np.percentile(nn_scores, 95)),
        'mean_similarity': float(np.mean(nn_scores)),
        'median_similarity': float(np.median(nn_scores)),
        'min_similarity': float(np.min(nn_scores)),
        'nn_indices': nn_indices,
        'nn_scores': nn_scores.tolist(),
    }


def load_clip_model(device: str = "cuda"):
    """
    Load CLIP model for text-image alignment.
    
    Args:
        device: Device to load model on
    
    Returns:
        Tuple of (model, processor)
    """
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return model, processor


def compute_clip_scores(
    images: List[np.ndarray],
    prompts: List[str],
    clip_model,
    clip_processor,
    device: str,
    show_progress: bool = True
) -> List[float]:
    """
    Compute CLIP scores between images and their prompts.
    
    Higher scores indicate better text-image alignment.
    
    Args:
        images: List of images as numpy arrays
        prompts: List of text prompts (one per image)
        clip_model: CLIP model
        clip_processor: CLIP processor
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        List of CLIP scores (one per image)
    """
    scores = []
    
    iterator = zip(images, prompts)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing CLIP scores")
    
    for img, prompt in iterator:
        # Convert numpy to PIL
        pil_img = Image.fromarray(img)
        
        # Process
        inputs = clip_processor(
            text=[prompt],
            images=[pil_img],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image[0, 0].item()
        
        scores.append(score)
    
    return scores


def compute_score_metrics(scores: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of scores.
    
    Generic helper function used by BioViL and other text-image alignment metrics.
    
    Args:
        scores: List of scores (e.g., BioViL similarity scores)
    
    Returns:
        Dictionary with mean, median, max, min, std
    """
    scores_array = np.array(scores)
    
    return {
        'mean_score': float(np.mean(scores_array)),
        'median_score': float(np.median(scores_array)),
        'max_score': float(np.max(scores_array)),
        'min_score': float(np.min(scores_array)),
        'std_score': float(np.std(scores_array)),
        'scores': scores_array.tolist(),
    }


def load_torchxrayvision_model(device: str = "cuda"):
    """
    Load TorchXRayVision DenseNet model for pathology classification.
    
    Args:
        device: Device to load model on
    
    Returns:
        TorchXRayVision model
    """
    try:
        import torchxrayvision as xrv
    except ImportError:
        raise ImportError(
            "torchxrayvision is required for pathology confidence metrics. "
            "Install it with: pip install torchxrayvision"
        )
    
    print("Loading TorchXRayVision model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()
    return model


def preprocess_for_torchxrayvision(images: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess images for TorchXRayVision model.
    
    TorchXRayVision expects:
    - Grayscale images
    - Shape: (batch, 1, 224, 224)
    - Normalized to [0, 1]
    
    Args:
        images: List of RGB numpy arrays (H, W, 3) in [0, 255]
    
    Returns:
        Batch tensor ready for TorchXRayVision
    """
    import torchvision.transforms as transforms
    
    # Convert to grayscale and resize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    
    processed = [transform(img) for img in images]
    return torch.stack(processed)


def compute_pathology_confidence(
    images: List[np.ndarray],
    target_pathology: str,
    xrv_model,
    device: str,
    show_progress: bool = True
) -> Dict[str, any]:
    """
    Compute pathology confidence scores using TorchXRayVision.
    
    Measures: "Does the image actually contain the target pathology?"
    
    Args:
        images: List of images as numpy arrays
        target_pathology: Target pathology name (e.g., "Fibrosis", "Pneumonia")
        xrv_model: TorchXRayVision model
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        Dictionary with:
            - mean_confidence: Mean probability for target pathology
            - median_confidence: Median probability
            - std_confidence: Standard deviation
            - confidences: List of all confidence scores
    """
    import torchxrayvision as xrv
    
    # Map common pathology names to TorchXRayVision labels
    pathology_mapping = {
        "Fibrosis": "Fibrosis",
        "Pneumonia": "Pneumonia",
        "Effusion": "Effusion",
        "Hernia": "Hernia",
        "Infiltration": "Infiltration",
        "Mass": "Mass",
        "Nodule": "Nodule",
        "Atelectasis": "Atelectasis",
        "Pneumothorax": "Pneumothorax",
        "Pleural_Thickening": "Pleural_Thickening",
        "Cardiomegaly": "Cardiomegaly",
        "Emphysema": "Emphysema",
        "Edema": "Edema",
        "Consolidation": "Consolidation",
    }
    
    if target_pathology not in pathology_mapping:
        raise ValueError(
            f"Unknown pathology: {target_pathology}. "
            f"Available: {list(pathology_mapping.keys())}"
        )
    
    # Get pathology index
    pathology_idx = xrv_model.pathologies.index(pathology_mapping[target_pathology])
    
    confidences = []
    batch_size = 16
    
    iterator = range(0, len(images), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Computing {target_pathology} confidence")
    
    for i in iterator:
        batch_images = images[i:i + batch_size]
        batch_tensor = preprocess_for_torchxrayvision(batch_images).to(device)
        
        with torch.no_grad():
            outputs = xrv_model(batch_tensor)
            # Get probability for target pathology
            probs = torch.sigmoid(outputs[:, pathology_idx])
            confidences.extend(probs.cpu().numpy().tolist())
    
    confidences = np.array(confidences)
    
    return {
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences)),
        'std_confidence': float(np.std(confidences)),
        'min_confidence': float(np.min(confidences)),
        'max_confidence': float(np.max(confidences)),
        'confidences': confidences.tolist(),
    }


def load_biovil_model(device: str = "cuda"):
    """
    Load BioViL (BiomedVLP-CXR-BERT) model for medical image-text alignment.
    
    Args:
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading BioViL model...")
    model_name = "microsoft/BiomedVLP-CXR-BERT-specialized"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def compute_biovil_scores(
    images: List[np.ndarray],
    prompts: List[str],
    biovil_model,
    biovil_tokenizer,
    device: str,
    show_progress: bool = True
) -> List[float]:
    """
    Compute BioViL semantic similarity scores.
    
    BioViL is specialized for medical imaging and provides better
    medical-specific text-image alignment than general CLIP.
    
    Args:
        images: List of images as numpy arrays
        prompts: List of text prompts (one per image)
        biovil_model: BioViL model
        biovil_tokenizer: BioViL tokenizer
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        List of similarity scores (one per image)
    """
    from torchvision import transforms
    
    # BioViL preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    scores = []
    batch_size = 16
    
    iterator = range(0, len(images), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing BioViL scores")
    
    for i in iterator:
        batch_images = images[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        
        # Process images
        image_tensors = torch.stack([transform(img) for img in batch_images]).to(device)
        
        # Process text
        text_inputs = biovil_tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Get embeddings
            image_features = biovil_model.get_image_features(image_tensors)
            text_features = biovil_model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features * text_features).sum(dim=-1)
            scores.extend(similarity.cpu().numpy().tolist())
    
    return scores


def compute_diversity_metrics(
    images: List[np.ndarray],
    xrv_model,
    device: str,
    show_progress: bool = True
) -> Dict[str, float]:
    """
    Compute diversity metrics using std dev of TorchXRayVision probabilities.
    
    Measures: "Are the images different from each other?"
    Higher std dev across pathologies indicates more diverse images.
    
    Args:
        images: List of images as numpy arrays
        xrv_model: TorchXRayVision model
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        Dictionary with diversity metrics
    """
    all_probs = []
    batch_size = 16
    
    iterator = range(0, len(images), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing diversity metrics")
    
    for i in iterator:
        batch_images = images[i:i + batch_size]
        batch_tensor = preprocess_for_torchxrayvision(batch_images).to(device)
        
        with torch.no_grad():
            outputs = xrv_model(batch_tensor)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
    
    # Shape: (num_images, num_pathologies)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Compute std dev for each pathology across all images
    std_per_pathology = np.std(all_probs, axis=0)
    
    # Overall diversity: mean of std devs
    overall_diversity = np.mean(std_per_pathology)
    
    return {
        'overall_diversity': float(overall_diversity),
        'mean_std': float(np.mean(std_per_pathology)),
        'median_std': float(np.median(std_per_pathology)),
        'max_std': float(np.max(std_per_pathology)),
        'min_std': float(np.min(std_per_pathology)),
        'pathology_stds': std_per_pathology.tolist(),
    }


def compute_fmd(
    generated_images: List[np.ndarray],
    real_images: List[np.ndarray],
    xrv_model,
    device: str,
    show_progress: bool = True
) -> float:
    """
    Compute Fréchet MedicalNet Distance (FMD).
    
    Similar to FID but using TorchXRayVision features instead of Inception.
    Measures the distance between the distributions of generated and real images
    in the feature space of a medical imaging model.
    
    Lower FMD = generated distribution is closer to real distribution.
    
    Args:
        generated_images: List of generated images as numpy arrays
        real_images: List of real images as numpy arrays
        xrv_model: TorchXRayVision model
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        FMD score
    """
    def extract_features(images: List[np.ndarray]) -> np.ndarray:
        """Extract features from images using TorchXRayVision."""
        features = []
        batch_size = 16
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")
        
        for i in iterator:
            batch_images = images[i:i + batch_size]
            batch_tensor = preprocess_for_torchxrayvision(batch_images).to(device)
            
            with torch.no_grad():
                # Get features from penultimate layer
                feat = xrv_model.features(batch_tensor)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # Extract features
    gen_features = extract_features(generated_images)
    real_features = extract_features(real_images)
    
    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    mu_real = np.mean(real_features, axis=0)
    
    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_real = np.cov(real_features, rowvar=False)
    
    # Compute FMD
    diff = mu_gen - mu_real
    
    # Product might be almost singular
    covmean, _ = sqrtm(sigma_gen.dot(sigma_real), disp=False)
    
    if not np.isfinite(covmean).all():
        print("WARNING: FMD calculation resulted in singular product; adding epsilon to diagonal")
        offset = np.eye(sigma_gen.shape[0]) * 1e-6
        covmean = sqrtm((sigma_gen + offset).dot(sigma_real + offset))
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fmd = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)
    
    return float(fmd)


def compute_tsne_overlap(
    generated_images: List[np.ndarray],
    real_images: List[np.ndarray],
    xrv_model,
    device: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    show_progress: bool = True
) -> Dict[str, any]:
    """
    Compute t-SNE overlap between generated and real images.
    
    Measures how well generated images overlap with real images in
    a 2D t-SNE embedding of the feature space.
    
    Args:
        generated_images: List of generated images as numpy arrays
        real_images: List of real images as numpy arrays
        xrv_model: TorchXRayVision model
        device: Device for computation
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        show_progress: Show progress bar
    
    Returns:
        Dictionary with:
            - tsne_embeddings: 2D coordinates (shape: [total_images, 2])
            - labels: 0=real, 1=generated
            - overlap_score: Overlap metric (0-1, higher is better)
    """
    def extract_features(images: List[np.ndarray]) -> np.ndarray:
        """Extract features from images using TorchXRayVision."""
        features = []
        batch_size = 16
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features for t-SNE")
        
        for i in iterator:
            batch_images = images[i:i + batch_size]
            batch_tensor = preprocess_for_torchxrayvision(batch_images).to(device)
            
            with torch.no_grad():
                feat = xrv_model.features(batch_tensor)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # Extract features
    gen_features = extract_features(generated_images)
    real_features = extract_features(real_images)
    
    # Combine features
    all_features = np.concatenate([real_features, gen_features], axis=0)
    labels = np.array([0] * len(real_images) + [1] * len(generated_images))
    
    # Compute t-SNE
    if show_progress:
        print(f"Computing t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(all_features) - 1),
        n_iter=n_iter,
        random_state=42,
        verbose=1 if show_progress else 0
    )
    embeddings = tsne.fit_transform(all_features)
    
    # Compute overlap score (using k-nearest neighbors)
    from scipy.spatial.distance import cdist
    
    real_embeddings = embeddings[labels == 0]
    gen_embeddings = embeddings[labels == 1]
    
    # For each generated point, find distance to nearest real point
    distances = cdist(gen_embeddings, real_embeddings)
    min_distances = np.min(distances, axis=1)
    
    # Overlap score: fraction of generated points within threshold of real points
    # Use median distance as threshold
    threshold = np.median(min_distances)
    overlap_score = np.mean(min_distances <= threshold)
    
    return {
        'tsne_embeddings': embeddings.tolist(),
        'labels': labels.tolist(),
        'overlap_score': float(overlap_score),
        'mean_distance': float(np.mean(min_distances)),
        'median_distance': float(np.median(min_distances)),
    }


# ============================================================================
# DIVERSITY ANALYSIS METRICS
# ============================================================================

def compute_intra_class_variance(images: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute pixel-level variance across images.
    
    Higher variance = more diverse images at pixel level.
    
    Args:
        images: List of images as numpy arrays
    
    Returns:
        Dictionary with variance statistics
    """
    # Stack images: shape (num_images, height, width, channels)
    images_array = np.stack(images, axis=0)
    
    # Compute variance across images for each pixel
    pixel_variance = np.var(images_array, axis=0)
    
    # Average variance across all pixels
    mean_pixel_variance = np.mean(pixel_variance)
    
    # Variance per channel
    if images_array.shape[-1] == 3:
        variance_per_channel = [np.mean(np.var(images_array[..., i], axis=0)) for i in range(3)]
    else:
        variance_per_channel = [mean_pixel_variance]
    
    return {
        'mean_pixel_variance': float(mean_pixel_variance),
        'variance_per_channel': variance_per_channel,
        'min_pixel_variance': float(np.min(pixel_variance)),
        'max_pixel_variance': float(np.max(pixel_variance)),
    }


def compute_feature_dispersion(images: List[np.ndarray], xrv_model, device: str, show_progress: bool = True) -> Dict[str, float]:
    """
    Compute dispersion (spread) in TorchXRayVision feature space.
    
    Uses determinant of covariance matrix as measure of spread.
    Higher dispersion = images are more spread out in feature space.
    
    Args:
        images: List of images as numpy arrays
        xrv_model: TorchXRayVision model
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        Dictionary with dispersion metrics
    """
    features = []
    batch_size = 16
    
    iterator = range(0, len(images), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting features for dispersion")
    
    for i in iterator:
        batch_images = images[i:i + batch_size]
        batch_tensor = preprocess_for_torchxrayvision(batch_images).to(device)
        
        with torch.no_grad():
            feat = xrv_model.features(batch_tensor)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
    
    # Shape: (num_images, num_features)
    features = np.concatenate(features, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(features, rowvar=False)
    
    # Dispersion metrics
    # 1. Determinant of covariance (volume of distribution)
    det_cov = np.linalg.det(cov_matrix)
    
    # 2. Trace of covariance (sum of variances)
    trace_cov = np.trace(cov_matrix)
    
    # 3. Mean pairwise distance
    from scipy.spatial.distance import pdist
    pairwise_distances = pdist(features, metric='euclidean')
    mean_pairwise_dist = np.mean(pairwise_distances)
    
    return {
        'det_covariance': float(det_cov),
        'trace_covariance': float(trace_cov),
        'mean_pairwise_distance': float(mean_pairwise_dist),
        'std_pairwise_distance': float(np.std(pairwise_distances)),
    }


def compute_self_similarity(images: List[np.ndarray], num_samples: int = 100, show_progress: bool = True) -> Dict[str, any]:
    """
    Compute pairwise SSIM within the batch of images.
    
    Lower self-similarity = more diverse images.
    
    Args:
        images: List of images
        num_samples: Number of random pairs to sample (for speed)
        show_progress: Show progress bar
    
    Returns:
        Dictionary with self-similarity statistics
    """
    num_images = len(images)
    
    # Sample random pairs
    if num_images * (num_images - 1) // 2 > num_samples:
        # Sample random pairs
        np.random.seed(42)
        pairs = []
        for _ in range(num_samples):
            i, j = np.random.choice(num_images, size=2, replace=False)
            pairs.append((i, j))
    else:
        # Use all pairs
        pairs = [(i, j) for i in range(num_images) for j in range(i + 1, num_images)]
    
    ssim_scores = []
    iterator = pairs
    if show_progress:
        iterator = tqdm(pairs, desc="Computing pairwise SSIM")
    
    for i, j in iterator:
        score = compute_ssim(images[i], images[j])
        ssim_scores.append(score)
    
    ssim_scores = np.array(ssim_scores)
    
    return {
        'mean_self_ssim': float(np.mean(ssim_scores)),
        'median_self_ssim': float(np.median(ssim_scores)),
        'std_self_ssim': float(np.std(ssim_scores)),
        'min_self_ssim': float(np.min(ssim_scores)),
        'max_self_ssim': float(np.max(ssim_scores)),
        'self_ssim_scores': ssim_scores.tolist(),
    }
