# src/models/grounding_dino_detector.py
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Grounding DINO imports
import sys
import os

# Add GroundingDINO to Python path
groundingdino_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GroundingDINO')
if os.path.exists(groundingdino_path):
    sys.path.insert(0, groundingdino_path)

# Apply compatibility patch for BERT model
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

# Apply comprehensive compatibility patches
def apply_advanced_patches():
    """Apply advanced compatibility patches"""
    try:
        # Apply patches directly without importing
        patch_transformers_compatibility()
        patch_groundingdino_transforms()
        print("✓ All patches applied successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Advanced patches failed: {e}, using basic patches")
        return False

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
    
    # Patch 2: Monkey patch tensor operations for CPU compatibility
    try:
        original_to = torch.Tensor.to
        
        def patched_to(self, *args, **kwargs):
            """Patched tensor.to() method that handles device/dtype confusion"""
            # Handle various problematic argument combinations
            
            # Case 1: dtype is actually a device object
            if 'dtype' in kwargs and isinstance(kwargs['dtype'], torch.device):
                # Move device from dtype to correct position
                device = kwargs.pop('dtype')
                if args:
                    # If there's already a positional argument, replace it
                    args = (device,) + args[1:]
                else:
                    # Add device as first argument
                    args = (device,) + args
                return original_to(self, *args, **kwargs)
            
            # Case 2: First argument is a device object and dtype is also specified incorrectly
            if len(args) >= 1 and isinstance(args[0], torch.device):
                if 'dtype' in kwargs and isinstance(kwargs['dtype'], torch.device):
                    # Swap the incorrectly placed dtype and device
                    device = args[0]
                    kwargs['dtype'] = None
                    return original_to(self, device, **kwargs)
            
            # Case 3: Handle when dtype parameter is actually a device
            if 'dtype' in kwargs and str(kwargs['dtype']).startswith('torch.device'):
                device = kwargs.pop('dtype')
                return original_to(self, device, **kwargs)
            
            return original_to(self, *args, **kwargs)
        
        torch.Tensor.to = patched_to
        print("✓ Patched torch.Tensor.to method")
        
    except Exception as e:
        print(f"⚠️ Could not patch torch.Tensor.to: {e}")
    
    # Patch 3: Fix bool tensor operations by monkey patching the subtraction operation
    try:
        original_rsub = torch.Tensor.__rsub__
        
        def patched_rsub(self, other):
            """Patched __rsub__ to handle bool tensor subtraction"""
            # Check if this is the problematic case: (1.0 - bool_tensor)
            if isinstance(other, (int, float)) and hasattr(self, 'dtype') and self.dtype == torch.bool:
                # Convert bool tensor to float before subtraction
                return other - self.float()
            return original_rsub(self, other)
        
        torch.Tensor.__rsub__ = patched_rsub
        print("✓ Patched torch.Tensor.__rsub__ for bool tensor operations")
        
    except Exception as e:
        print(f"⚠️ Could not patch torch.Tensor.__rsub__: {e}")
    
    # Patch 4: Fix torch.finfo dtype/device confusion
    try:
        original_finfo = torch.finfo
        
        def patched_finfo(dtype):
            """Patched finfo to handle when dtype is actually a device"""
            if isinstance(dtype, torch.device):
                # Default to float32 when device is passed instead of dtype
                return original_finfo(torch.float32)
            return original_finfo(dtype)
        
        torch.finfo = patched_finfo
        print("✓ Patched torch.finfo for dtype/device confusion")
        
    except Exception as e:
        print(f"⚠️ Could not patch torch.finfo: {e}")

def patch_groundingdino_transforms():
    """Patch GroundingDINO transforms for better compatibility"""
    try:
        import groundingdino.datasets.transforms as T
        print("✓ GroundingDINO transforms available")
    except ImportError:
        print("⚠️ Could not import GroundingDINO transforms")

# Apply patches
if not apply_advanced_patches():
    # Fallback to basic patches
    patch_bert_model()

try:
    from groundingdino.util.inference import load_model, predict, annotate
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T
    GROUNDINGDINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GroundingDINO import failed: {e}")
    print("Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
    GROUNDINGDINO_AVAILABLE = False

def load_image_from_pil(image_pil: Image.Image):
    """Load image from PIL Image for GroundingDINO"""
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_pil.convert("RGB"))
    image_transformed, _ = transform(image_pil.convert("RGB"), None)
    return image, image_transformed

def predict_cpu(model, image, caption, box_threshold, text_threshold):
    """Custom predict function that forces CPU usage"""
    from groundingdino.util.inference import preprocess_caption
    import torch
    
    caption = preprocess_caption(caption=caption)
    
    # Ensure everything is on CPU to avoid device mismatches
    model = model.cpu()
    image = image.cpu()
    
    # Don't move to device - keep on CPU
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    prediction_logits = outputs["pred_logits"].sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"][0]  # prediction_boxes.shape = (nq, 4)
    
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit in logits
    ]
    
    return boxes, logits, phrases

def get_phrases_from_posmap(posmap, tokenized, tokenizer):
    """Get phrases from position map"""
    assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
    if posmap.dim() != 1:
        raise ValueError(f"posmap has wrong shape: {posmap.shape}")
    from groundingdino.util.utils import get_phrases_from_posmap as original_get_phrases
    return original_get_phrases(posmap, tokenized, tokenizer)

class GroundingDINODetector:
    """Advanced object detection using GroundingDINO."""
    
    def __init__(self, 
                 model_config_path: str = None,
                 model_checkpoint_path: str = None):
        """
        Initialize GroundingDINO detector.
        
        Args:
            model_config_path: Path to model config file
            model_checkpoint_path: Path to model checkpoint
        """
        print("Initializing GroundingDINO detector...")
        
        # Default paths - use absolute paths
        if model_config_path is None:
            # Get the project root directory (3 levels up from src/models/)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_config_path = os.path.join(
                project_root, 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py'
            )
            
        if model_checkpoint_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_checkpoint_path = os.path.join(
                project_root, 'weights', 'groundingdino_swint_ogc.pth'
            )
        
        print(f"Using config path: {model_config_path}")
        print(f"Using weights path: {model_checkpoint_path}")
        
        try:
            # Check if GroundingDINO is available
            if not GROUNDINGDINO_AVAILABLE:
                raise ImportError("GroundingDINO is not available")
                
            # Load model - force CPU mode
            self.model = load_model(
                model_config_path,
                model_checkpoint_path,
                device="cpu"  # Force CPU mode
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Force model to CPU if CUDA is not available
            if not torch.cuda.is_available():
                print("✓ Forcing GroundingDINO to CPU mode (CUDA not available)")
                self.model.cpu()
            
            print(f"✓ GroundingDINO loaded on: {self.device}")
            
            # Construction-specific prompts
            self.construction_prompts = {
                "heavy_machinery": [
                    "excavator", "bulldozer", "backhoe", "crane", "loader",
                    "dump truck", "cement mixer", "forklift", "grader", "compactor",
                    "haul truck", "drill rig", "pile driver"
                ],
                "vehicles": [
                    "construction vehicle", "heavy equipment", "mining truck",
                    "transport truck", "service vehicle"
                ],
                "structures": [
                    "construction site", "building foundation", "scaffolding",
                    "temporary structure", "storage container", "site office"
                ],
                "materials": [
                    "pile of dirt", "pile of gravel", "pile of sand",
                    "construction materials", "debris pile", "soil stockpile"
                ],
                "people": [
                    "construction worker", "worker with hard hat",
                    "safety vest", "construction personnel"
                ]
            }
            
        except Exception as e:
            print(f"✗ Failed to load GroundingDINO: {e}")
            print("This might be due to model compatibility issues.")
            print("The detector will still work but with limited functionality.")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if GroundingDINO is available."""
        return self.model is not None
    
    def detect_construction(
        self,
        image: Image.Image,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_detections: int = 20
    ) -> Dict[str, Any]:
        """
        Detect construction-related objects in image.
        
        Args:
            image: PIL Image
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            max_detections: Maximum number of detections to return
            
        Returns:
            Dictionary with detections and metadata
        """
        if not self.is_available():
            # Fallback: return mock detections for testing
            print("GroundingDINO not available, returning mock detections")
            return {
                "available": False,
                "error": "GroundingDINO not loaded - using fallback mode",
                "detections": [],
                "fallback_mode": True,
                "total_detections": 0,
                "message": "GroundingDINO model failed to load due to compatibility issues. The system will continue with limited detection capabilities."
            }
        
        try:
            # Convert PIL to OpenCV format for GroundingDINO
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create text prompt from all construction categories
            all_prompts = []
            for category, prompts in self.construction_prompts.items():
                all_prompts.extend(prompts)
            
            # Join with periods (GroundingDINO works better with periods)
            text_prompt = ". ".join(all_prompts) + "."
            
            # Load image for GroundingDINO - use custom function
            image_source, image_tensor = load_image_from_pil(image)
            
            # Run detection - use custom CPU predict function
            boxes, logits, phrases = predict_cpu(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Convert to list of detections
            detections = []
            if len(boxes) > 0:
                # Get image dimensions from PIL Image
                w, h = image.size
                for box, score, phrase in zip(boxes, logits, phrases):
                    # Convert box from [x_center, y_center, width, height] to [x1, y1, x2, y2]
                    box = box * torch.Tensor([w, h, w, h])
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    
                    # Convert to list
                    box = box.cpu().numpy().astype(int).tolist()
                    
                    # Determine category
                    category = self._categorize_phrase(phrase)
                    
                    detections.append({
                        "bbox": box,
                        "score": float(score.cpu()) if score.numel() == 1 else float(score.mean()),
                        "label": phrase,
                        "category": category,
                        "confidence": f"{float(score.cpu() if score.numel() == 1 else score.mean()):.1%}"
                    })
            
            # Sort by score and limit
            detections.sort(key=lambda x: x["score"], reverse=True)
            detections = detections[:max_detections]
            
            # Generate statistics
            stats = self._generate_statistics(detections)
            
            # Create annotated image
            annotated_image = self._annotate_image(image.copy(), detections)
            
            return {
                "available": True,
                "detections": detections,
                "statistics": stats,
                "annotated_image": None,  # Don't include PIL Image in JSON response
                "total_detections": len(detections),
                "text_prompt_used": text_prompt
            }
            
        except Exception as e:
            print(f"GroundingDINO detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "available": False,
                "error": str(e),
                "detections": []
            }
    
    def _categorize_phrase(self, phrase: str) -> str:
        """Categorize detected phrase."""
        phrase_lower = phrase.lower()
        
        for category, prompts in self.construction_prompts.items():
            for prompt in prompts:
                if prompt in phrase_lower:
                    return category
        
        # Check for specific keywords
        if any(word in phrase_lower for word in ["excavator", "bulldozer", "crane", "loader"]):
            return "heavy_machinery"
        elif any(word in phrase_lower for word in ["truck", "vehicle", "equipment"]):
            return "vehicles"
        elif any(word in phrase_lower for word in ["pile", "dirt", "gravel", "sand"]):
            return "materials"
        elif any(word in phrase_lower for word in ["worker", "person", "people"]):
            return "people"
        elif any(word in phrase_lower for word in ["site", "structure", "building"]):
            return "structures"
        
        return "other"
    
    def _generate_statistics(self, detections: List[Dict]) -> Dict[str, Any]:
        """Generate statistics from detections."""
        if not detections:
            return {
                "heavy_machinery_count": 0,
                "vehicle_count": 0,
                "material_count": 0,
                "people_count": 0,
                "structure_count": 0,
                "total_count": 0,
                "avg_confidence": 0.0
            }
        
        category_counts = {
            "heavy_machinery": 0,
            "vehicles": 0,
            "materials": 0,
            "people": 0,
            "structures": 0,
            "other": 0
        }
        
        total_score = 0.0
        
        for det in detections:
            category = det.get("category", "other")
            if category in category_counts:
                category_counts[category] += 1
            total_score += det["score"]
        
        return {
            "heavy_machinery_count": category_counts["heavy_machinery"],
            "vehicle_count": category_counts["vehicles"],
            "material_count": category_counts["materials"],
            "people_count": category_counts["people"],
            "structure_count": category_counts["structures"],
            "other_count": category_counts["other"],
            "total_count": len(detections),
            "avg_confidence": total_score / len(detections) if detections else 0.0
        }
    
    def _annotate_image(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Create annotated image with bounding boxes."""
        if not detections:
            return image
        
        draw = ImageDraw.Draw(image)
        
        # Colors for different categories
        category_colors = {
            "heavy_machinery": (255, 0, 0),      # Red
            "vehicles": (0, 0, 255),            # Blue
            "materials": (0, 255, 0),           # Green
            "people": (255, 255, 0),            # Yellow
            "structures": (255, 0, 255),        # Magenta
            "other": (128, 128, 128)            # Gray
        }
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            score = det["score"]
            category = det.get("category", "other")
            
            color = category_colors.get(category, (128, 128, 128))
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label background
            label_text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            
            # Draw label text
            draw.text((bbox[0], bbox[1] - 20), label_text, fill=(255, 255, 255), font=font)
        
        return image
    
    def detect_with_custom_prompt(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> List[Dict]:
        """Detect objects with custom text prompt."""
        if not self.is_available():
            return []
        
        try:
            # Convert PIL to OpenCV for GroundingDINO
            image_source, image_tensor = load_image_from_pil(image)
            
            # Run detection - use custom CPU predict function
            boxes, logits, phrases = predict_cpu(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Format results
            results = []
            if len(boxes) > 0:
                # Get image dimensions from PIL Image
                w, h = image.size
                for box, score, phrase in zip(boxes, logits, phrases):
                    # Convert box format
                    box = box * torch.Tensor([w, h, w, h])
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    
                    results.append({
                        "bbox": box.cpu().numpy().astype(int).tolist(),
                        "score": float(score),
                        "label": phrase,
                        "confidence": f"{float(score):.1%}"
                    })
            
            return results
            
        except Exception as e:
            print(f"Custom prompt detection error: {e}")
            return []