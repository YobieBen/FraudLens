"""
Document forgery detection for signatures, stamps, and alterations.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class DocumentForgeryDetector:
    """
    Detects forgery in documents.

    Features:
    - Signature verification
    - Stamp authenticity check
    - Alteration detection
    - Font consistency analysis
    - Paper texture analysis
    """

    def __init__(self, sensitivity: float = 0.8):
        """
        Initialize forgery detector.

        Args:
            sensitivity: Detection sensitivity (0-1)
        """
        self.sensitivity = sensitivity
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize detector."""
        self.initialized = True
        logger.info("DocumentForgeryDetector initialized")

    async def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect forgery in document image.

        Args:
            image: Document image array

        Returns:
            Forgery detection result
        """
        indicators = []
        confidence_scores = []

        # Check for alterations
        alterations = self._detect_alterations(image)
        if alterations:
            indicators.extend(alterations)
            confidence_scores.append(0.8)

        # Check signature authenticity
        signature_issues = self._check_signatures(image)
        if signature_issues:
            indicators.extend(signature_issues)
            confidence_scores.append(0.7)

        # Check stamp authenticity
        stamp_issues = self._check_stamps(image)
        if stamp_issues:
            indicators.extend(stamp_issues)
            confidence_scores.append(0.6)

        # Check font consistency
        font_issues = self._check_font_consistency(image)
        if font_issues:
            indicators.extend(font_issues)
            confidence_scores.append(0.5)

        # Check paper texture
        texture_issues = self._check_paper_texture(image)
        if texture_issues:
            indicators.extend(texture_issues)
            confidence_scores.append(0.4)

        # Calculate overall confidence
        is_forged = len(indicators) > 0 and max(confidence_scores, default=0) > (
            1.0 - self.sensitivity
        )
        confidence = max(confidence_scores) if confidence_scores else 0.0

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "indicators": indicators[:10],  # Top 10 indicators
            "alteration_regions": self._find_alteration_regions(image) if is_forged else [],
            "signature_analysis": self._analyze_signatures(image),
            "stamp_analysis": self._analyze_stamps(image),
        }

    def _detect_alterations(self, image: np.ndarray) -> List[str]:
        """Detect document alterations."""
        alterations = []

        # Error Level Analysis (ELA)
        ela_score = self._perform_ela(image)
        if ela_score > 0.3:
            alterations.append(f"ELA detected alterations (score: {ela_score:.2f})")

        # Check for copy-paste artifacts
        if self._has_copy_paste_artifacts(image):
            alterations.append("Copy-paste artifacts detected")

        # Check for pixel inconsistencies
        if self._has_pixel_inconsistencies(image):
            alterations.append("Pixel-level inconsistencies found")

        # Check for unnatural edges
        if self._has_unnatural_edges(image):
            alterations.append("Unnatural edge patterns detected")

        return alterations

    def _check_signatures(self, image: np.ndarray) -> List[str]:
        """Check signature authenticity."""
        issues = []

        # Detect signature regions
        signature_regions = self._detect_signature_regions(image)

        for i, region in enumerate(signature_regions):
            # Check ink consistency
            if not self._check_ink_consistency(region):
                issues.append(f"Signature {i+1}: Inconsistent ink")

            # Check stroke patterns
            if not self._check_stroke_patterns(region):
                issues.append(f"Signature {i+1}: Unnatural stroke patterns")

            # Check pressure variations
            if not self._check_pressure_variations(region):
                issues.append(f"Signature {i+1}: Missing pressure variations")

        return issues

    def _check_stamps(self, image: np.ndarray) -> List[str]:
        """Check stamp authenticity."""
        issues = []

        # Detect stamp regions
        stamp_regions = self._detect_stamp_regions(image)

        for i, region in enumerate(stamp_regions):
            # Check stamp clarity
            if not self._check_stamp_clarity(region):
                issues.append(f"Stamp {i+1}: Suspiciously perfect clarity")

            # Check ink bleeding
            if not self._check_ink_bleeding(region):
                issues.append(f"Stamp {i+1}: No natural ink bleeding")

            # Check rotation alignment
            if not self._check_stamp_alignment(region):
                issues.append(f"Stamp {i+1}: Perfect alignment (suspicious)")

        return issues

    def _check_font_consistency(self, image: np.ndarray) -> List[str]:
        """Check font consistency across document."""
        issues = []

        # Detect text regions
        text_regions = self._detect_text_regions(image)

        if len(text_regions) > 1:
            # Compare font characteristics
            font_features = [self._extract_font_features(region) for region in text_regions]

            # Check for inconsistencies
            if self._has_font_inconsistencies(font_features):
                issues.append("Inconsistent fonts detected")

            # Check for mixed resolutions
            if self._has_resolution_mismatch(text_regions):
                issues.append("Text resolution mismatch")

        return issues

    def _check_paper_texture(self, image: np.ndarray) -> List[str]:
        """Check paper texture consistency."""
        issues = []

        # Extract texture features
        texture_variance = self._calculate_texture_variance(image)

        # Check for unnatural uniformity
        if texture_variance < 10:
            issues.append("Unnaturally uniform paper texture")

        # Check for digital artifacts
        if self._has_digital_artifacts(image):
            issues.append("Digital printing artifacts detected")

        # Check for scanning artifacts
        if self._has_scanning_artifacts(image):
            issues.append("Multiple scanning generations detected")

        return issues

    def _perform_ela(self, image: np.ndarray) -> float:
        """Perform Error Level Analysis."""
        import io

        from PIL import Image

        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_img = Image.fromarray(image)

        # Save with JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        # Reload and compare
        compressed = Image.open(buffer)
        compressed_arr = np.array(compressed)

        # Calculate difference
        if compressed_arr.shape == image.shape:
            diff = np.abs(image.astype(float) - compressed_arr.astype(float))

            # Amplify differences
            ela = diff * 10
            ela = np.minimum(ela, 255)

            # Calculate score based on variance
            score = np.var(ela) / 1000.0  # Normalize
            return min(score, 1.0)

        return 0.0

    def _has_copy_paste_artifacts(self, image: np.ndarray) -> bool:
        """Check for copy-paste artifacts."""
        # Look for repeated patterns
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Simple check: look for identical regions
        h, w = gray.shape
        patch_size = 32

        if h > patch_size * 2 and w > patch_size * 2:
            # Sample random patches
            for _ in range(10):
                y1 = np.random.randint(0, h - patch_size)
                x1 = np.random.randint(0, w - patch_size)
                patch1 = gray[y1 : y1 + patch_size, x1 : x1 + patch_size]

                # Search for similar patches
                for _ in range(5):
                    y2 = np.random.randint(0, h - patch_size)
                    x2 = np.random.randint(0, w - patch_size)

                    if abs(y1 - y2) > patch_size or abs(x1 - x2) > patch_size:
                        patch2 = gray[y2 : y2 + patch_size, x2 : x2 + patch_size]

                        # Check similarity
                        diff = np.mean(np.abs(patch1 - patch2))
                        if diff < 5:  # Very similar
                            return True

        return False

    def _has_pixel_inconsistencies(self, image: np.ndarray) -> bool:
        """Check for pixel-level inconsistencies."""
        # Check for JPEG block artifacts in specific regions
        if image.ndim == 3:
            # Look for 8x8 block boundaries (JPEG artifacts)
            for c in range(3):
                channel = image[:, :, c]

                # Calculate differences at 8-pixel intervals
                h_diff = np.abs(np.diff(channel[::8, :], axis=0))
                v_diff = np.abs(np.diff(channel[:, ::8], axis=1))

                # High differences at block boundaries indicate tampering
                if np.mean(h_diff) > 20 or np.mean(v_diff) > 20:
                    return True

        return False

    def _has_unnatural_edges(self, image: np.ndarray) -> bool:
        """Check for unnatural edge patterns."""
        from scipy import ndimage

        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Detect edges
        edges = ndimage.sobel(gray)

        # Check for perfectly straight lines (might indicate digital editing)
        from scipy.stats import linregress

        # Sample edge points
        edge_points = np.where(np.abs(edges) > np.mean(np.abs(edges)) + np.std(np.abs(edges)))

        if len(edge_points[0]) > 100:
            # Fit line to sample of points
            sample_idx = np.random.choice(
                len(edge_points[0]), min(100, len(edge_points[0])), replace=False
            )
            x = edge_points[1][sample_idx]
            y = edge_points[0][sample_idx]

            if len(x) > 2:
                _, _, r_value, _, _ = linregress(x, y)

                # Perfect line has r_value close to 1
                if abs(r_value) > 0.95:
                    return True

        return False

    def _detect_signature_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect signature regions in document."""
        regions = []

        # Simple heuristic: look for dark regions in typical signature locations
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        h, w = gray.shape

        # Check bottom third of document
        bottom_region = gray[2 * h // 3 :, :]

        # Find dark regions
        threshold = np.mean(bottom_region) - np.std(bottom_region)
        mask = bottom_region < threshold

        # Find connected components
        from scipy import ndimage

        labeled, num_features = ndimage.label(mask)

        for i in range(1, min(num_features + 1, 4)):  # Max 3 signatures
            region_mask = labeled == i
            if np.sum(region_mask) > 100:  # Minimum size
                y_coords, x_coords = np.where(region_mask)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                min_x, max_x = np.min(x_coords), np.max(x_coords)

                signature = bottom_region[min_y : max_y + 1, min_x : max_x + 1]
                regions.append(signature)

        return regions

    def _detect_stamp_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect stamp regions in document."""
        regions = []

        # Look for circular/rectangular colored regions
        if image.ndim == 3:
            # Check for red/blue stamps
            red_channel = image[:, :, 0]
            blue_channel = image[:, :, 2]

            # Red stamp detection
            red_mask = (red_channel > 150) & (image[:, :, 1] < 100) & (blue_channel < 100)

            # Blue stamp detection
            blue_mask = (blue_channel > 150) & (image[:, :, 1] < 100) & (red_channel < 100)

            # Find regions
            from scipy import ndimage

            for mask in [red_mask, blue_mask]:
                labeled, num_features = ndimage.label(mask)

                for i in range(1, min(num_features + 1, 3)):
                    region_mask = labeled == i
                    if np.sum(region_mask) > 500:  # Minimum size
                        y_coords, x_coords = np.where(region_mask)
                        min_y, max_y = np.min(y_coords), np.max(y_coords)
                        min_x, max_x = np.min(x_coords), np.max(x_coords)

                        stamp = image[min_y : max_y + 1, min_x : max_x + 1]
                        regions.append(stamp)

        return regions

    def _detect_text_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions in document."""
        regions = []

        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Simple text detection using horizontal projection
        h_projection = np.mean(gray < np.mean(gray), axis=1)

        # Find text lines
        text_lines = h_projection > 0.1

        # Group consecutive lines
        from scipy import ndimage

        labeled, num_features = ndimage.label(text_lines)

        for i in range(1, min(num_features + 1, 10)):
            line_indices = np.where(labeled == i)[0]
            if len(line_indices) > 5:
                min_y, max_y = np.min(line_indices), np.max(line_indices)
                text_region = gray[min_y : max_y + 1, :]
                regions.append(text_region)

        return regions

    def _check_ink_consistency(self, signature: np.ndarray) -> bool:
        """Check if ink appears consistent."""
        if signature.size == 0:
            return True

        # Check color variance
        if signature.ndim == 3:
            color_std = np.std(signature.reshape(-1, 3), axis=0)
            # Natural ink should have some variance
            return np.all(color_std > 5) and np.all(color_std < 50)

        return True

    def _check_stroke_patterns(self, signature: np.ndarray) -> bool:
        """Check for natural stroke patterns."""
        if signature.size == 0:
            return True

        # Check for stroke width variations
        from scipy import ndimage

        if signature.ndim == 3:
            signature = np.mean(signature, axis=2)

        # Detect strokes
        binary = signature < np.mean(signature)

        # Calculate stroke widths
        distances = ndimage.distance_transform_edt(binary)

        # Natural signatures have varying stroke widths
        width_variance = np.var(distances[binary])

        return width_variance > 1.0

    def _check_pressure_variations(self, signature: np.ndarray) -> bool:
        """Check for pressure variations in signature."""
        if signature.size == 0:
            return True

        if signature.ndim == 3:
            signature = np.mean(signature, axis=2)

        # Check intensity variations (darker = more pressure)
        intensities = signature[signature < np.mean(signature)]

        if len(intensities) > 0:
            # Natural signatures have pressure variations
            intensity_std = np.std(intensities)
            return intensity_std > 10

        return True

    def _check_stamp_clarity(self, stamp: np.ndarray) -> bool:
        """Check if stamp has natural imperfections."""
        if stamp.size == 0:
            return True

        # Calculate edge sharpness
        from scipy import ndimage

        if stamp.ndim == 3:
            stamp = np.mean(stamp, axis=2)

        edges = ndimage.sobel(stamp)

        # Perfect digital stamps have very sharp edges
        edge_sharpness = np.mean(np.abs(edges))

        # Natural stamps should have some blur/imperfection
        return edge_sharpness < 50

    def _check_ink_bleeding(self, stamp: np.ndarray) -> bool:
        """Check for natural ink bleeding."""
        if stamp.size == 0:
            return True

        # Check edges for gradual transition
        if stamp.ndim == 3:
            # Check color channels for bleeding
            for c in range(3):
                channel = stamp[:, :, c]

                # Calculate gradient at edges
                from scipy import ndimage

                gradient = np.abs(ndimage.sobel(channel))

                # Natural stamps have gradual transitions
                if np.max(gradient) > 100:  # Very sharp transition
                    return False

        return True

    def _check_stamp_alignment(self, stamp: np.ndarray) -> bool:
        """Check if stamp is suspiciously well-aligned."""
        if stamp.size == 0:
            return True

        # Check if stamp edges are perfectly aligned with image axes
        if stamp.ndim == 3:
            stamp = np.mean(stamp, axis=2)

        # Detect primary orientation
        from scipy import ndimage

        # Find angle using moments
        m = ndimage.measurements.center_of_mass(stamp < np.mean(stamp))

        # Natural stamps are rarely perfectly aligned
        # This is simplified - in production would use proper angle detection
        return True

    def _extract_font_features(self, text_region: np.ndarray) -> Dict[str, float]:
        """Extract font characteristics."""
        features = {}

        if text_region.size == 0:
            return features

        # Calculate average stroke width
        from scipy import ndimage

        binary = text_region < np.mean(text_region)
        distances = ndimage.distance_transform_edt(binary)

        features["stroke_width"] = np.mean(distances[binary]) if np.any(binary) else 0
        features["char_density"] = np.mean(binary)
        features["contrast"] = np.std(text_region)

        return features

    def _has_font_inconsistencies(self, font_features: List[Dict]) -> bool:
        """Check for font inconsistencies."""
        if len(font_features) < 2:
            return False

        # Compare stroke widths
        stroke_widths = [f.get("stroke_width", 0) for f in font_features]
        if stroke_widths:
            width_variance = np.var(stroke_widths)
            if width_variance > 2:  # Significant variance
                return True

        return False

    def _has_resolution_mismatch(self, text_regions: List[np.ndarray]) -> bool:
        """Check for resolution mismatches."""
        if len(text_regions) < 2:
            return False

        # Calculate sharpness for each region
        sharpness_values = []

        for region in text_regions:
            from scipy import ndimage

            if region.ndim == 3:
                region = np.mean(region, axis=2)

            # Laplacian for sharpness
            laplacian = ndimage.laplace(region)
            sharpness = np.var(laplacian)
            sharpness_values.append(sharpness)

        # Check for significant differences
        if sharpness_values:
            sharpness_range = np.max(sharpness_values) - np.min(sharpness_values)
            return sharpness_range > 100

        return False

    def _calculate_texture_variance(self, image: np.ndarray) -> float:
        """Calculate texture variance."""
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Calculate local variance
        from scipy import ndimage

        mean = ndimage.uniform_filter(gray, size=5)
        sqr_mean = ndimage.uniform_filter(gray**2, size=5)
        variance = sqr_mean - mean**2

        return np.mean(variance)

    def _has_digital_artifacts(self, image: np.ndarray) -> bool:
        """Check for digital printing artifacts."""
        # Look for halftone patterns
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # FFT to detect regular patterns
        fft = np.fft.fft2(gray)
        fft_mag = np.abs(fft)

        # Look for peaks indicating regular patterns
        mean_mag = np.mean(fft_mag)
        std_mag = np.std(fft_mag)
        peaks = fft_mag > (mean_mag + 3 * std_mag)

        # Many peaks indicate halftone or digital patterns
        return np.sum(peaks) > 100

    def _has_scanning_artifacts(self, image: np.ndarray) -> bool:
        """Check for multiple scanning generation artifacts."""
        # Look for moiré patterns
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Check for regular interference patterns
        fft = np.fft.fft2(gray)
        fft_mag = np.abs(fft)

        # Moiré creates specific frequency patterns
        h, w = fft_mag.shape
        center_h, center_w = h // 2, w // 2

        # Check for rings in frequency domain
        distances = np.sqrt(
            (np.arange(h)[:, None] - center_h) ** 2 + (np.arange(w) - center_w) ** 2
        )

        # Look for regular rings
        for radius in [10, 20, 30, 40]:
            ring_mask = (distances > radius - 2) & (distances < radius + 2)
            ring_energy = np.mean(fft_mag[ring_mask])

            if ring_energy > np.mean(fft_mag) * 2:
                return True

        return False

    def _find_alteration_regions(self, image: np.ndarray) -> List[Dict[str, int]]:
        """Find regions with alterations."""
        regions = []

        # Perform ELA to find altered regions
        ela_map = self._create_ela_map(image)

        # Find high-energy regions
        threshold = np.mean(ela_map) + 2 * np.std(ela_map)
        mask = ela_map > threshold

        # Find connected components
        from scipy import ndimage

        labeled, num_features = ndimage.label(mask)

        for i in range(1, min(num_features + 1, 5)):
            region_mask = labeled == i
            if np.sum(region_mask) > 100:
                y_coords, x_coords = np.where(region_mask)

                regions.append(
                    {
                        "x": int(np.min(x_coords)),
                        "y": int(np.min(y_coords)),
                        "width": int(np.max(x_coords) - np.min(x_coords)),
                        "height": int(np.max(y_coords) - np.min(y_coords)),
                    }
                )

        return regions

    def _create_ela_map(self, image: np.ndarray) -> np.ndarray:
        """Create ELA map for visualization."""
        import io

        from PIL import Image

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_img = Image.fromarray(image)

        # Compress
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        compressed = Image.open(buffer)
        compressed_arr = np.array(compressed)

        # Calculate difference
        if compressed_arr.shape == image.shape:
            diff = np.abs(image.astype(float) - compressed_arr.astype(float))

            # Amplify
            ela = diff * 10
            ela = np.minimum(ela, 255)

            # Convert to grayscale
            if ela.ndim == 3:
                ela = np.mean(ela, axis=2)

            return ela

        return np.zeros_like(image[:, :, 0] if image.ndim == 3 else image)

    def _analyze_signatures(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze signatures in document."""
        signatures = self._detect_signature_regions(image)

        return {
            "count": len(signatures),
            "authentic": (
                all(
                    self._check_ink_consistency(s)
                    and self._check_stroke_patterns(s)
                    and self._check_pressure_variations(s)
                    for s in signatures
                )
                if signatures
                else True
            ),
        }

    def _analyze_stamps(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze stamps in document."""
        stamps = self._detect_stamp_regions(image)

        return {
            "count": len(stamps),
            "authentic": (
                all(
                    self._check_stamp_clarity(s)
                    and self._check_ink_bleeding(s)
                    and self._check_stamp_alignment(s)
                    for s in stamps
                )
                if stamps
                else True
            ),
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.initialized = False
