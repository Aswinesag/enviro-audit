# src/utils/image_utils.py
from PIL import Image, ImageEnhance
from typing import Tuple

class ImagePreprocessor:
    """Preprocess images for better analysis."""
    
    @staticmethod
    def prepare_for_detection(
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 512),
        enhance: bool = True
    ) -> Image.Image:
        """
        Prepare image for analysis.
        
        Args:
            image: Input PIL Image
            target_size: Target size for resizing
            enhance: Whether to enhance contrast/brightness
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Enhance if requested
        if enhance:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
        
        return image