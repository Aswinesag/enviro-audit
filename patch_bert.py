#!/usr/bin/env python3
"""
Patch for GroundingDINO BERT compatibility issue
"""
import sys
import os

def patch_bert_model():
    """Patch the BertModel to add missing get_head_mask method"""
    try:
        from transformers.models.bert.modeling_bert import BertModel
        
        def get_head_mask(self, head_mask, num_hidden_layers):
            """Add missing get_head_mask method to BertModel"""
            if head_mask is not None:
                if head_mask.size()[0] != num_hidden_layers:
                    raise ValueError(
                        f"head_mask should have size [{num_hidden_layers}], but got {head_mask.size()}"
                    )
            return head_mask
        
        # Add the method to BertModel if it doesn't exist
        if not hasattr(BertModel, 'get_head_mask'):
            BertModel.get_head_mask = get_head_mask
            print("✓ Patched BertModel.get_head_mask method")
            return True
        else:
            print("✓ BertModel.get_head_mask already exists")
            return True
            
    except ImportError as e:
        print(f"✗ Failed to patch BertModel: {e}")
        return False

if __name__ == "__main__":
    print("Applying GroundingDINO compatibility patches...")
    patch_bert_model()
    print("Patch application complete.")
