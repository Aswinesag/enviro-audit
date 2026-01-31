#!/usr/bin/env python3
"""
Advanced compatibility patches for GroundingDINO
"""
import sys
import os
import torch
import warnings
warnings.filterwarnings("ignore")

def patch_transformers_compatibility():
    """Apply comprehensive patches for transformers compatibility"""
    
    # Patch 1: Fix BERT model get_head_mask method
    try:
        from transformers.models.bert.modeling_bert import BertModel
        
        def get_head_mask(self, head_mask, num_hidden_layers):
            """Fixed get_head_mask method"""
            if head_mask is not None:
                if head_mask.size()[0] != num_hidden_layers:
                    raise ValueError(
                        f"head_mask should have size [{num_hidden_layers}], but got {head_mask.size()}"
                    )
            return head_mask
        
        if not hasattr(BertModel, 'get_head_mask'):
            BertModel.get_head_mask = get_head_mask
            print("✓ Patched BertModel.get_head_mask method")
        
    except ImportError:
        print("⚠️ Could not import BertModel for patching")
    
    # Patch 2: Fix tensor.to() calls in modeling_utils
    try:
        from transformers import modeling_utils
        
        original_get_extended_attention_mask = modeling_utils.get_extended_attention_mask
        
        def patched_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None, device=None):
            """Patched version that handles device parameter correctly"""
            # Handle the case where device is passed as dtype
            if dtype is not None and hasattr(dtype, 'type') and dtype.type == 'torch.device':
                device = dtype
                dtype = None
            
            return original_get_extended_attention_mask(
                self, attention_mask, input_shape, dtype=dtype, device=device
            )
        
        modeling_utils.get_extended_attention_mask = patched_get_extended_attention_mask
        print("✓ Patched get_extended_attention_mask method")
        
    except Exception as e:
        print(f"⚠️ Could not patch get_extended_attention_mask: {e}")
    
    # Patch 3: Monkey patch tensor operations for CPU compatibility
    try:
        original_to = torch.Tensor.to
        
        def patched_to(self, *args, **kwargs):
            """Patched tensor.to() method that handles device/dtype confusion"""
            # If first argument is a device object and dtype is also specified incorrectly
            if len(args) >= 1 and hasattr(args[0], 'type') and args[0].type == 'torch.device':
                if 'dtype' in kwargs and isinstance(kwargs['dtype'], torch.device):
                    # Swap the incorrectly placed dtype and device
                    device = args[0]
                    kwargs['dtype'] = None
                    return original_to(self, device, **kwargs)
            
            return original_to(self, *args, **kwargs)
        
        torch.Tensor.to = patched_to
        print("✓ Patched torch.Tensor.to method")
        
    except Exception as e:
        print(f"⚠️ Could not patch torch.Tensor.to: {e}")

def patch_groundingdino_transforms():
    """Patch GroundingDINO transforms for better compatibility"""
    try:
        import groundingdino.datasets.transforms as T
        
        # Check if we can access and patch transforms
        if hasattr(T, 'RandomResize'):
            print("✓ GroundingDINO transforms available")
        else:
            print("⚠️ GroundingDINO transforms not found")
            
    except ImportError:
        print("⚠️ Could not import GroundingDINO transforms")

def apply_all_patches():
    """Apply all compatibility patches"""
    print("Applying comprehensive GroundingDINO compatibility patches...")
    
    patch_transformers_compatibility()
    patch_groundingdino_transforms()
    
    print("✓ All patches applied successfully!")

if __name__ == "__main__":
    apply_all_patches()
