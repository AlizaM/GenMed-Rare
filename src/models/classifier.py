"""Model builders for binary classification."""
import torch
import torch.nn as nn
import timm
from typing import Optional
import logging

from src.config import Config

logger = logging.getLogger(__name__)


class SwinClassifier(nn.Module):
    """
    Swin Transformer for binary classification.
    
    Uses timm library for pretrained Swin models with ImageNet weights.
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Load pretrained Swin Transformer
        logger.info(f"Loading {config.model.variant} (pretrained={config.model.pretrained})")
        
        self.backbone = timm.create_model(
            config.model.variant,
            pretrained=config.model.pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension
        self.num_features = self.backbone.num_features
        
        # Freeze backbone if specified
        if config.model.freeze_backbone:
            logger.info("Freezing backbone weights")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.model.dropout),
            nn.Linear(self.num_features, config.model.num_classes)
        )
        
        logger.info(f"Model created with {self.num_features} features -> {config.model.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Logits [batch, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        logger.info("Unfreezing backbone weights")
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(config: Config) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        PyTorch model
    """
    if config.model.name.lower() == "swin_transformer":
        model = SwinClassifier(config)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    # Move to device
    device = torch.device(config.hardware.device)
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into (if load_optimizer=True)
        
    Returns:
        Dictionary with checkpoint metadata
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if requested
    if load_optimizer and optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning("Optimizer state not found in checkpoint")
    
    logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return checkpoint
