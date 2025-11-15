"""
Model definitions for medical image classification.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
import logging

logger = logging.getLogger(__name__)


class SwinClassifier(nn.Module):
    """
    Swin Transformer based classifier for medical images.
    """
    
    def __init__(self, config):
        """
        Initialize Swin classifier.
        
        Args:
            config: Configuration object
        """
        super(SwinClassifier, self).__init__()
        
        self.config = config
        model_config = config.model
        
        # Get the appropriate Swin model based on variant
        self.backbone = self._get_backbone(
            variant=model_config.variant,
            pretrained=model_config.pretrained
        )
        
        # Get the number of features from the backbone
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            # Replace the classification head
            self.backbone.head = nn.Identity()
        else:
            raise AttributeError("Swin model structure unexpected")
        
        # Create custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=model_config.dropout),
            nn.Linear(in_features, model_config.num_classes)
        )
        
        logger.info(f"Initialized Swin-{model_config.variant} classifier")
        logger.info(f"Pretrained: {model_config.pretrained}")
        logger.info(f"Number of classes: {model_config.num_classes}")
    
    def _get_backbone(self, variant: str, pretrained: bool):
        """
        Get Swin Transformer backbone.
        
        Args:
            variant: Model variant (tiny, small, base, large)
            pretrained: Whether to use pretrained weights
            
        Returns:
            Swin Transformer model
        """
        variant = variant.lower()
        
        if variant == 'tiny' or variant == 't':
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            model = swin_t(weights=weights)
        elif variant == 'small' or variant == 's':
            weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            model = swin_s(weights=weights)
        elif variant == 'base' or variant == 'b':
            weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
            model = swin_b(weights=weights)
        else:
            raise ValueError(f"Unknown Swin variant: {variant}. "
                           f"Choose from: tiny, small, base")
        
        return model
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def create_model(config):
    """
    Factory function to create model based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance
    """
    model_name = config.model.name.lower()
    
    if 'swin' in model_name:
        model = SwinClassifier(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('/home/runner/work/GenMed-Rare/GenMed-Rare')
    from src.utils.config import ConfigManager
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = ConfigManager.from_yaml('configs/train_effusion_fibrosis.yaml')
    
    # Create model
    model = create_model(config)
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"\nModel has {n_params:,} trainable parameters")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
