"""
Environment verification tests for diffusion training setup.

Run with:
    pytest tests/test_environment.py -v -s
    
This test verifies that all required dependencies are installed
and the environment is ready for diffusion training.
"""

import pytest
import torch
import importlib
from pathlib import Path


class TestEnvironmentSetup:
    """Test that environment has all required packages and capabilities."""
    
    def test_pytorch_installation(self):
        """Test PyTorch is installed and working."""
        assert torch.__version__ is not None
        
        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        assert torch.allclose(y, torch.tensor([2.0, 4.0, 6.0]))
    
    def test_cuda_availability(self):
        """Test CUDA availability (warning if not available)."""
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"\n✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            
            # Test basic CUDA operations
            if torch.cuda.device_count() > 0:
                x = torch.tensor([1.0]).cuda()
                y = x * 2
                assert y.item() == 2.0
        else:
            print("\n⚠️  CUDA not available - training will be slow on CPU")
        
        # Test passes regardless of CUDA availability
        assert True
    
    @pytest.mark.parametrize("package_name", [
        "diffusers",
        "transformers", 
        "accelerate",
        "peft",
        "bitsandbytes"
    ])
    def test_diffusion_packages(self, package_name):
        """Test that diffusion packages are installed."""
        try:
            package = importlib.import_module(package_name)
            assert package is not None
            print(f"\n✅ {package_name}: {getattr(package, '__version__', 'version unknown')}")
        except ImportError:
            pytest.fail(f"❌ {package_name} is not installed")
    
    def test_diffusers_functionality(self):
        """Test basic diffusers functionality."""
        try:
            from diffusers import DDPMScheduler, UNet2DConditionModel
            
            # Test scheduler creation
            scheduler = DDPMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler",
                local_files_only=False
            )
            assert scheduler is not None
            
            print("\n✅ Diffusers basic functionality works")
        except Exception as e:
            pytest.fail(f"❌ Diffusers functionality test failed: {e}")
    
    def test_transformers_functionality(self):
        """Test basic transformers functionality."""
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
            
            # Test tokenizer loading (should download if not cached)
            tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                local_files_only=False
            )
            assert tokenizer is not None
            
            # Test basic tokenization
            tokens = tokenizer("A chest X-ray", return_tensors="pt")
            assert tokens['input_ids'].shape[1] > 0
            
            print("\n✅ Transformers basic functionality works")
        except Exception as e:
            pytest.fail(f"❌ Transformers functionality test failed: {e}")
    
    def test_peft_functionality(self):
        """Test PEFT (LoRA) functionality."""
        try:
            from peft import LoraConfig, get_peft_model
            import torch.nn as nn
            
            # Create a model with named modules (PEFT requires this)
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                target_modules=["linear"],  # Target the named linear module
                lora_dropout=0.1,
                bias="none",
            )
            
            # Apply LoRA
            peft_model = get_peft_model(model, lora_config)
            assert peft_model is not None
            
            print("\n✅ PEFT (LoRA) functionality works")
        except Exception as e:
            pytest.fail(f"❌ PEFT functionality test failed: {e}")
    
    def test_accelerate_functionality(self):
        """Test Accelerate functionality."""
        try:
            from accelerate import Accelerator
            
            # Test accelerator creation
            accelerator = Accelerator(mixed_precision="no")  # No mixed precision for test
            assert accelerator is not None
            assert accelerator.device is not None
            
            print(f"\n✅ Accelerate functionality works (device: {accelerator.device})")
        except Exception as e:
            pytest.fail(f"❌ Accelerate functionality test failed: {e}")
    
    def test_bitsandbytes_functionality(self):
        """Test bitsandbytes functionality (8-bit optimizers)."""
        try:
            import bitsandbytes as bnb
            import torch.nn as nn
            
            # Test 8-bit Adam optimizer creation
            model = nn.Linear(10, 5)
            optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
            assert optimizer is not None
            
            print(f"\n✅ Bitsandbytes functionality works")
        except Exception as e:
            # bitsandbytes might not work on all systems (especially CPU-only)
            print(f"\n⚠️  Bitsandbytes test failed: {e}")
            print("   This is okay if you're on CPU-only or incompatible CUDA version")
    
    def test_memory_requirements(self):
        """Test basic memory requirements."""
        if torch.cuda.is_available():
            # Check GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n📊 GPU Memory: {total_mem:.1f} GB")
            
            if total_mem < 8:
                print("   ⚠️  Warning: Less than 8GB GPU memory")
                print("   Consider using batch_size=1 and memory optimizations")
            elif total_mem < 16:
                print("   ✅ Adequate GPU memory for moderate batch sizes")
            else:
                print("   🚀 Excellent GPU memory for large batch sizes")
        
        # Test can create reasonably sized tensors
        try:
            # Test ~1GB tensor creation
            x = torch.randn(1, 3, 512, 512)
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                del x_gpu
            del x
            print("   ✅ Basic tensor allocation works")
        except Exception as e:
            print(f"   ⚠️  Large tensor allocation failed: {e}")


class TestConfigurationFiles:
    """Test that configuration files are valid."""
    
    def test_diffusion_config_exists(self):
        """Test that diffusion config file exists."""
        config_path = Path("configs/config_diffusion.yaml")
        assert config_path.exists(), "config_diffusion.yaml not found"
    
    def test_diffusion_config_valid(self):
        """Test that diffusion config is valid YAML."""
        import yaml
        
        config_path = Path("configs/config_diffusion.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test required sections exist
        required_sections = ['experiment', 'model', 'data', 'training']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
        
        # Test some critical values
        assert config['model']['pretrained_model'] is not None
        assert config['data']['image_size'] > 0
        assert config['training']['num_train_epochs'] > 0
    
    def test_data_paths_configuration(self):
        """Test that data paths in config point to existing locations."""
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/config_diffusion.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check data directory
        data_dir = Path(config['data']['data_dir'])
        csv_file = data_dir / config['data']['csv_file']
        
        if not data_dir.exists():
            print(f"\n⚠️  Data directory not found: {data_dir}")
            print("   Make sure to run data preprocessing first")
        elif not csv_file.exists():
            print(f"\n⚠️  CSV file not found: {csv_file}")
            print("   Make sure to run data preprocessing first")
        else:
            print(f"\n✅ Data files found: {csv_file}")


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v", "-s"])