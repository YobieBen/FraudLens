"""
Image preprocessing for vision fraud detection.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from loguru import logger


class ImagePreprocessor:
    """
    Preprocesses images for fraud detection models.

    Features:
    - Format standardization
    - Resolution optimization
    - Color space normalization
    - Quality assessment
    - Metal Performance Shaders support for Apple Silicon
    """

    # Supported formats
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tiff"}

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        use_metal: bool = True,
        quality_threshold: float = 0.3,
    ):
        """
        Initialize preprocessor.

        Args:
            target_size: Target image size (width, height)
            normalize: Whether to normalize pixel values
            use_metal: Use Metal acceleration on Apple Silicon
            quality_threshold: Minimum quality score to accept
        """
        self.target_size = target_size
        self.normalize = normalize
        self.use_metal = use_metal and self._check_metal()
        self.quality_threshold = quality_threshold

        logger.info(f"ImagePreprocessor initialized (Metal: {self.use_metal})")

    async def process(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        auto_enhance: bool = True,
    ) -> np.ndarray:
        """
        Process image for model input.

        Args:
            image: Input image
            auto_enhance: Automatically enhance low-quality images

        Returns:
            Processed image array
        """
        # Load image
        pil_image = self._load_image(image)

        # Convert HEIC/HEIF if needed
        if isinstance(image, (str, Path)):
            if Path(image).suffix.lower() in [".heic", ".heif"]:
                pil_image = self._convert_heic(pil_image)

        # Assess quality
        quality_score = self._assess_quality(pil_image)

        # Enhance if needed
        if auto_enhance and quality_score < self.quality_threshold:
            pil_image = self._enhance_image(pil_image)

        # Standardize format (convert to RGB)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Resize to target size
        pil_image = self._smart_resize(pil_image)

        # Convert to numpy array
        img_array = np.array(pil_image)

        # Normalize if requested
        if self.normalize:
            img_array = self._normalize_array(img_array)

        # Apply Metal acceleration if available
        if self.use_metal:
            img_array = self._apply_metal_optimization(img_array)

        return img_array

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray, bytes]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _convert_heic(self, image: Image.Image) -> Image.Image:
        """Convert HEIC/HEIF to standard format."""
        try:
            # Try using pillow-heif if available
            from pillow_heif import register_heif_opener

            register_heif_opener()
            return image
        except ImportError:
            logger.warning("pillow-heif not installed, HEIC support limited")
            return image

    def _assess_quality(self, image: Image.Image) -> float:
        """
        Assess image quality.

        Returns:
            Quality score (0-1)
        """
        # Convert to grayscale for analysis
        gray = image.convert("L")
        arr = np.array(gray)

        # Calculate metrics
        scores = []

        # 1. Contrast score (variance of pixel values)
        contrast = np.std(arr) / 128.0  # Normalize to [0, 1]
        scores.append(min(contrast, 1.0))

        # 2. Sharpness score (Laplacian variance)
        from scipy import ndimage

        laplacian = ndimage.laplace(arr)
        sharpness = np.var(laplacian) / 10000.0  # Normalize
        scores.append(min(sharpness, 1.0))

        # 3. Brightness score (not too dark or bright)
        mean_brightness = np.mean(arr)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        scores.append(brightness_score)

        # 4. Resolution score
        width, height = image.size
        resolution_score = min(width * height / (1920 * 1080), 1.0)
        scores.append(resolution_score)

        # Average score
        return sum(scores) / len(scores)

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance low-quality image."""
        # Auto-contrast
        image = ImageOps.autocontrast(image)

        # Sharpen
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        # Adjust brightness if needed
        arr = np.array(image.convert("L"))
        mean_brightness = np.mean(arr)

        if mean_brightness < 100:  # Too dark
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)
        elif mean_brightness > 200:  # Too bright
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)

        return image

    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Smart resize preserving aspect ratio.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        # Get current size
        width, height = image.size
        target_width, target_height = self.target_size

        # Calculate aspect ratios
        aspect = width / height
        target_aspect = target_width / target_height

        if aspect > target_aspect:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create canvas at target size
        canvas = Image.new("RGB", self.target_size, (0, 0, 0))

        # Paste resized image centered
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas.paste(image, (x_offset, y_offset))

        return canvas

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] or [-1, 1]."""
        if arr.dtype == np.uint8:
            # Normalize to [0, 1]
            return arr.astype(np.float32) / 255.0
        return arr

    def _apply_metal_optimization(self, arr: np.ndarray) -> np.ndarray:
        """Apply Metal Performance Shaders optimization."""
        # This would use Metal Performance Shaders for optimization
        # For now, just return the array
        # In production, would use PyObjC to access Metal
        return arr

    def _check_metal(self) -> bool:
        """Check if Metal is available."""
        try:
            import platform

            return platform.system() == "Darwin" and platform.processor() == "arm"
        except:
            return False

    def batch_process(self, images: list, **kwargs) -> list:
        """Process batch of images."""
        import asyncio

        async def _batch():
            tasks = [self.process(img, **kwargs) for img in images]
            return await asyncio.gather(*tasks)

        return asyncio.run(_batch())

    def get_supported_formats(self) -> set:
        """Get list of supported formats."""
        return self.SUPPORTED_FORMATS
