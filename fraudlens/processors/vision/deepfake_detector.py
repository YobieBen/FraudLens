"""
Advanced Deepfake Detection Module
Implements state-of-the-art deepfake detection for images and videos
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from PIL import Image

# Try to import torch and related modules
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    transforms = None

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from scipy import signal
    from scipy.stats import entropy

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None
    entropy = lambda x: 0.0  # Fallback function
import warnings

warnings.filterwarnings("ignore")


class DeepfakeType(Enum):
    """Types of deepfake detection results."""

    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FULL_SYNTHESIS = "full_synthesis"
    AUDIO_VISUAL_MISMATCH = "audio_visual_mismatch"
    GAN_GENERATED = "gan_generated"
    DIFFUSION_GENERATED = "diffusion_generated"
    MORPHING_ATTACK = "morphing_attack"


@dataclass
class DeepfakeDetectionResult:
    """Result of deepfake detection analysis."""

    is_deepfake: bool
    confidence: float
    deepfake_type: Optional[DeepfakeType]
    detection_methods: Dict[str, float]
    facial_analysis: Dict[str, Any]
    texture_analysis: Dict[str, float]
    frequency_analysis: Dict[str, float]
    gan_fingerprints: Dict[str, float]
    explanation: str
    visualization: Optional[np.ndarray]
    metadata: Dict[str, Any]


# Only define torch-dependent classes if torch is available
if TORCH_AVAILABLE:
    class FaceForensicsNet(nn.Module):
        """Enhanced CNN for deepfake detection."""

        def __init__(self, num_classes: int = 2):
            super(FaceForensicsNet, self).__init__()

            # Convolutional layers for feature extraction
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Sigmoid(),
            )

            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            # Feature extraction
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)

            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, 2)

            x = F.relu(self.bn4(self.conv4(x)))

            # Apply attention
            attention_weights = self.attention(x)
            x = x * attention_weights

            # Global pooling
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)

            # Classification
            output = self.classifier(x)

            return output, attention_weights
else:
    # Placeholder when torch is not available
    FaceForensicsNet = None


class DeepfakeDetector:
    """Advanced deepfake detection system for images and videos."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize deepfake detector with advanced models."""
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
            logger.warning("PyTorch not available, deepfake detection features will be limited")

        # Initialize face detection and analysis if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            self.face_detection = None
            self.face_mesh = None
            logger.warning("MediaPipe not available, using basic face detection")

        # Initialize deepfake detection model (only if torch is available)
        if TORCH_AVAILABLE and FaceForensicsNet is not None:
            self.model = FaceForensicsNet().to(self.device)
            if model_path and Path(model_path).exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # Image preprocessing
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.model = None
            self.transform = None

        # Detection thresholds
        self.deepfake_threshold = 0.7
        self.high_confidence_threshold = 0.85
        self.gan_fingerprint_threshold = 0.6

        logger.info(f"DeepfakeDetector initialized on {self.device}")

    async def detect_image_deepfake(
        self, image_path: str, return_visualization: bool = False
    ) -> DeepfakeDetectionResult:
        """
        Detect deepfake in a single image.

        Args:
            image_path: Path to image file
            return_visualization: Whether to return visualization

        Returns:
            DeepfakeDetectionResult with comprehensive analysis
        """
        logger.info(f"Analyzing image for deepfake: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Face detection and extraction
        if not self.face_mesh:
            # Fallback to basic detection using OpenCV
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return DeepfakeDetectionResult(
                    is_deepfake=False,
                    confidence=0.0,
                    deepfake_type=None,
                    detection_methods={},
                    facial_analysis={},
                    texture_analysis={},
                    frequency_analysis={},
                    gan_fingerprints={},
                    explanation="No faces detected in the image",
                    visualization=None,
                    metadata={"faces_detected": 0},
                )

            # Use first detected face
            x, y, w, h = faces[0]
            face_roi = image[y : y + h, x : x + w]
            face_landmarks = None
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
        else:
            face_results = self.face_mesh.process(rgb_image)

            if not face_results.multi_face_landmarks:
                logger.warning("No faces detected in image")
                return DeepfakeDetectionResult(
                    is_deepfake=False,
                    confidence=0.0,
                    deepfake_type=None,
                    detection_methods={},
                    facial_analysis={},
                    texture_analysis={},
                    frequency_analysis={},
                    gan_fingerprints={},
                    explanation="No faces detected in the image",
                    visualization=None,
                    metadata={"faces_detected": 0},
                )

            # Extract face region
            h, w = image.shape[:2]
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])

            x_min, y_min = landmarks_array.min(axis=0).astype(int)
            x_max, y_max = landmarks_array.max(axis=0).astype(int)

            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            face_roi = image[y_min:y_max, x_min:x_max]

        # Run multiple detection methods
        detection_methods = {}

        # 1. CNN-based detection
        cnn_score, attention_map = self._detect_with_cnn(face_roi)
        detection_methods["cnn_detection"] = cnn_score

        # 2. Texture analysis
        texture_analysis = self._analyze_texture(face_roi)
        detection_methods["texture_anomaly"] = texture_analysis["anomaly_score"]

        # 3. Frequency domain analysis
        frequency_analysis = self._analyze_frequency_spectrum(face_roi)
        detection_methods["frequency_anomaly"] = frequency_analysis["high_freq_anomaly"]

        # 4. GAN fingerprint detection
        gan_fingerprints = self._detect_gan_fingerprints(face_roi)
        detection_methods["gan_fingerprint"] = gan_fingerprints["fingerprint_score"]

        # 5. Facial landmark analysis
        facial_analysis = self._analyze_facial_features(landmarks_array, face_roi)
        detection_methods["facial_anomaly"] = facial_analysis["anomaly_score"]

        # 6. Color space analysis
        color_analysis = self._analyze_color_consistency(face_roi)
        detection_methods["color_anomaly"] = color_analysis

        # 7. Compression artifact analysis
        compression_score = self._detect_compression_artifacts(face_roi)
        detection_methods["compression_artifacts"] = compression_score

        # Ensemble decision
        confidence_scores = list(detection_methods.values())
        overall_confidence = np.mean(confidence_scores)

        # Weighted ensemble based on method reliability
        weights = {
            "cnn_detection": 0.3,
            "texture_anomaly": 0.15,
            "frequency_anomaly": 0.15,
            "gan_fingerprint": 0.15,
            "facial_anomaly": 0.1,
            "color_anomaly": 0.1,
            "compression_artifacts": 0.05,
        }

        weighted_confidence = sum(
            detection_methods.get(method, 0) * weight for method, weight in weights.items()
        )

        is_deepfake = weighted_confidence > self.deepfake_threshold

        # Determine deepfake type
        deepfake_type = self._determine_deepfake_type(
            detection_methods, gan_fingerprints, facial_analysis
        )

        # Generate visualization if requested
        visualization = None
        if return_visualization:
            visualization = self._create_visualization(
                image,
                (x_min, y_min, x_max, y_max),
                attention_map if cnn_score > 0.5 else None,
                detection_methods,
            )

        # Generate explanation
        explanation = self._generate_explanation(
            is_deepfake, weighted_confidence, detection_methods, deepfake_type
        )

        return DeepfakeDetectionResult(
            is_deepfake=is_deepfake,
            confidence=weighted_confidence,
            deepfake_type=deepfake_type,
            detection_methods=detection_methods,
            facial_analysis=facial_analysis,
            texture_analysis=texture_analysis,
            frequency_analysis=frequency_analysis,
            gan_fingerprints=gan_fingerprints,
            explanation=explanation,
            visualization=visualization,
            metadata={
                "faces_detected": len(face_results.multi_face_landmarks),
                "image_size": (h, w),
                "face_region": (x_min, y_min, x_max, y_max),
            },
        )

    async def detect_video_deepfake(
        self, video_path: str, sample_rate: int = 10, max_frames: int = 100
    ) -> DeepfakeDetectionResult:
        """
        Detect deepfake in video with temporal analysis.

        Args:
            video_path: Path to video file
            sample_rate: Sample every N frames
            max_frames: Maximum frames to analyze

        Returns:
            DeepfakeDetectionResult with video analysis
        """
        logger.info(f"Analyzing video for deepfake: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_scores = []
        temporal_consistency_scores = []
        detection_results = []

        frame_count = 0
        analyzed_count = 0
        prev_landmarks = None

        while cap.isOpened() and analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Analyze frame
                result = await self.detect_image_deepfake(
                    self._frame_to_temp_image(frame), return_visualization=False
                )

                frame_scores.append(result.confidence)
                detection_results.append(result.detection_methods)

                # Temporal consistency check
                if (
                    prev_landmarks is not None
                    and result.facial_analysis.get("landmarks") is not None
                ):
                    consistency = self._check_temporal_consistency(
                        prev_landmarks, result.facial_analysis["landmarks"]
                    )
                    temporal_consistency_scores.append(consistency)

                if result.facial_analysis.get("landmarks") is not None:
                    prev_landmarks = result.facial_analysis["landmarks"]

                analyzed_count += 1

            frame_count += 1

        cap.release()

        # Aggregate results
        avg_confidence = np.mean(frame_scores) if frame_scores else 0.0

        # Check temporal consistency
        temporal_anomaly = 0.0
        if temporal_consistency_scores:
            temporal_anomaly = 1.0 - np.mean(temporal_consistency_scores)

        # Combine frame and temporal analysis
        final_confidence = avg_confidence * 0.8 + temporal_anomaly * 0.2

        is_deepfake = final_confidence > self.deepfake_threshold

        # Aggregate detection methods
        aggregated_methods = {}
        for methods in detection_results:
            for key, value in methods.items():
                if key not in aggregated_methods:
                    aggregated_methods[key] = []
                aggregated_methods[key].append(value)

        avg_methods = {key: np.mean(values) for key, values in aggregated_methods.items()}
        avg_methods["temporal_anomaly"] = temporal_anomaly

        explanation = f"Video analysis: {analyzed_count} frames analyzed. "
        if is_deepfake:
            explanation += f"Deepfake detected with {final_confidence:.1%} confidence. "
            if temporal_anomaly > 0.5:
                explanation += "Temporal inconsistencies detected between frames."
        else:
            explanation += "Video appears authentic."

        return DeepfakeDetectionResult(
            is_deepfake=is_deepfake,
            confidence=final_confidence,
            deepfake_type=DeepfakeType.FACE_REENACTMENT if is_deepfake else None,
            detection_methods=avg_methods,
            facial_analysis={"temporal_consistency": 1.0 - temporal_anomaly},
            texture_analysis={},
            frequency_analysis={},
            gan_fingerprints={},
            explanation=explanation,
            visualization=None,
            metadata={
                "fps": fps,
                "total_frames": total_frames,
                "analyzed_frames": analyzed_count,
                "temporal_anomaly": temporal_anomaly,
            },
        )

    def _detect_with_cnn(self, face_image: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """Run CNN-based deepfake detection."""
        if not TORCH_AVAILABLE or self.model is None or self.transform is None:
            logger.warning("PyTorch or model not available, skipping CNN detection")
            return 0.0, None

        try:
            # Prepare image
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

            # Run model
            with torch.no_grad():
                output, attention = self.model(face_tensor)
                probabilities = F.softmax(output, dim=1)
                deepfake_prob = probabilities[0, 1].item()

            # Extract attention map
            attention_map = None
            if attention is not None:
                attention_map = attention[0].mean(dim=0).cpu().numpy()

            return deepfake_prob, attention_map

        except Exception as e:
            logger.warning(f"CNN detection error: {e}")
            return 0.0, None

    def _analyze_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture patterns for anomalies."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Local Binary Patterns
        lbp_hist = self._compute_lbp_histogram(gray)

        # Co-occurrence matrix
        glcm_features = self._compute_glcm_features(gray)

        # Detect unnatural smoothness
        smoothness = np.var(gray) / (np.mean(gray) + 1e-6)

        # Detect texture anomalies
        anomaly_score = 0.0

        # Check for over-smoothing (common in GANs)
        if smoothness < 0.1:
            anomaly_score += 0.3

        # Check for repetitive patterns
        if glcm_features["homogeneity"] > 0.9:
            anomaly_score += 0.3

        # Check for unusual LBP distribution
        lbp_entropy = entropy(lbp_hist)
        if lbp_entropy < 2.0:
            anomaly_score += 0.4

        return {
            "anomaly_score": min(anomaly_score, 1.0),
            "smoothness": smoothness,
            "homogeneity": glcm_features["homogeneity"],
            "lbp_entropy": lbp_entropy,
        }

    def _analyze_frequency_spectrum(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze frequency domain for GAN artifacts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Analyze radial average
        h, w = gray.shape
        center = (h // 2, w // 2)

        radial_profile = self._compute_radial_profile(magnitude_spectrum, center)

        # Look for GAN-specific patterns
        high_freq_energy = np.sum(radial_profile[len(radial_profile) // 2 :])
        low_freq_energy = np.sum(radial_profile[: len(radial_profile) // 2])

        freq_ratio = high_freq_energy / (low_freq_energy + 1e-6)

        # Detect periodic patterns (common in GANs)
        if SCIPY_AVAILABLE and signal:
            peaks = signal.find_peaks(radial_profile)[0]
            periodic_score = len(peaks) / len(radial_profile)
        else:
            # Simple peak detection fallback
            peaks = []
            for i in range(1, len(radial_profile) - 1):
                if (
                    radial_profile[i] > radial_profile[i - 1]
                    and radial_profile[i] > radial_profile[i + 1]
                ):
                    peaks.append(i)
            periodic_score = len(peaks) / len(radial_profile)

        return {
            "high_freq_anomaly": min(freq_ratio * 0.1, 1.0),
            "periodic_patterns": periodic_score,
            "frequency_ratio": freq_ratio,
        }

    def _detect_gan_fingerprints(self, image: np.ndarray) -> Dict[str, float]:
        """Detect GAN-specific fingerprints."""
        fingerprints = {}

        # Check for checkerboard artifacts
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Look for regular patterns in gradients
        gx_fft = np.fft.fft2(gx)
        gy_fft = np.fft.fft2(gy)

        # Check for peaks at specific frequencies (GAN artifacts)
        gx_spectrum = np.abs(np.fft.fftshift(gx_fft))
        gy_spectrum = np.abs(np.fft.fftshift(gy_fft))

        # Detect regular patterns
        pattern_score = self._detect_regular_patterns(gx_spectrum, gy_spectrum)

        # Check for color bleeding
        color_bleeding = self._detect_color_bleeding(image)

        fingerprints["fingerprint_score"] = (pattern_score + color_bleeding) / 2
        fingerprints["checkerboard_artifacts"] = pattern_score
        fingerprints["color_bleeding"] = color_bleeding

        return fingerprints

    def _analyze_facial_features(
        self, landmarks: np.ndarray, face_image: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze facial features for authenticity."""
        analysis = {}

        # Check facial symmetry
        if len(landmarks) >= 468:  # MediaPipe face landmarks
            left_eye = landmarks[33:42]
            right_eye = landmarks[263:272]

            face_center = np.mean(landmarks, axis=0)

            left_dist = np.mean(np.linalg.norm(left_eye - face_center, axis=1))
            right_dist = np.mean(np.linalg.norm(right_eye - face_center, axis=1))

            symmetry = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)

            # Check for unnatural eye movements
            eye_aspect_ratio = self._compute_eye_aspect_ratio(landmarks)

            # Analyze skin texture in specific regions
            skin_consistency = self._analyze_skin_consistency(face_image, landmarks)

            analysis["symmetry"] = symmetry
            analysis["eye_aspect_ratio"] = eye_aspect_ratio
            analysis["skin_consistency"] = skin_consistency
            analysis["landmarks"] = landmarks

            # Compute anomaly score
            anomaly = 0.0
            if symmetry < 0.8:
                anomaly += 0.3
            if eye_aspect_ratio < 0.15 or eye_aspect_ratio > 0.35:
                anomaly += 0.3
            if skin_consistency < 0.7:
                anomaly += 0.4

            analysis["anomaly_score"] = min(anomaly, 1.0)
        else:
            analysis["anomaly_score"] = 0.0

        return analysis

    def _analyze_color_consistency(self, image: np.ndarray) -> float:
        """Analyze color consistency across face regions."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Check for unnatural color distributions
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        # Normalize histograms
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)

        # Check for unusual peaks (synthetic images often have them)
        h_entropy = entropy(h_hist)
        s_entropy = entropy(s_hist)

        # Low entropy suggests artificial color distribution
        color_anomaly = 0.0
        if h_entropy < 3.0:
            color_anomaly += 0.5
        if s_entropy < 4.0:
            color_anomaly += 0.5

        return min(color_anomaly, 1.0)

    def _detect_compression_artifacts(self, image: np.ndarray) -> float:
        """Detect JPEG compression artifacts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect 8x8 block artifacts
        h, w = gray.shape
        block_size = 8

        block_boundaries = []
        for i in range(block_size, h - block_size, block_size):
            for j in range(block_size, w - block_size, block_size):
                # Check discontinuity at block boundaries
                top_diff = np.mean(
                    np.abs(gray[i - 1, j : j + block_size] - gray[i, j : j + block_size])
                )
                left_diff = np.mean(
                    np.abs(gray[i : i + block_size, j - 1] - gray[i : i + block_size, j])
                )

                block_boundaries.append(max(top_diff, left_diff))

        if block_boundaries:
            artifact_score = np.percentile(block_boundaries, 90) / 255.0
            return min(artifact_score * 2, 1.0)

        return 0.0

    def _determine_deepfake_type(
        self,
        detection_methods: Dict[str, float],
        gan_fingerprints: Dict[str, float],
        facial_analysis: Dict[str, Any],
    ) -> Optional[DeepfakeType]:
        """Determine the type of deepfake based on detection patterns."""
        if detection_methods.get("cnn_detection", 0) < 0.5:
            return None

        # High GAN fingerprints suggest GAN-generated
        if gan_fingerprints.get("fingerprint_score", 0) > 0.7:
            return DeepfakeType.GAN_GENERATED

        # Poor facial symmetry suggests face swap
        if facial_analysis.get("symmetry", 1.0) < 0.7:
            return DeepfakeType.FACE_SWAP

        # High frequency anomalies suggest full synthesis
        if detection_methods.get("frequency_anomaly", 0) > 0.7:
            return DeepfakeType.FULL_SYNTHESIS

        # Default to face reenactment
        return DeepfakeType.FACE_REENACTMENT

    def _create_visualization(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        attention_map: Optional[np.ndarray],
        detection_scores: Dict[str, float],
    ) -> np.ndarray:
        """Create visualization of detection results."""
        vis_image = image.copy()
        x_min, y_min, x_max, y_max = face_bbox

        # Draw face bounding box
        color = (0, 0, 255) if any(s > 0.7 for s in detection_scores.values()) else (0, 255, 0)
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add detection scores
        y_offset = 30
        for method, score in detection_scores.items():
            text = f"{method}: {score:.2f}"
            cv2.putText(
                vis_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 20

        # Overlay attention map if available
        if attention_map is not None:
            attention_resized = cv2.resize(attention_map, (x_max - x_min, y_max - y_min))
            attention_colored = cv2.applyColorMap(
                (attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            vis_image[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                vis_image[y_min:y_max, x_min:x_max], 0.7, attention_colored, 0.3, 0
            )

        return vis_image

    def _generate_explanation(
        self,
        is_deepfake: bool,
        confidence: float,
        detection_methods: Dict[str, float],
        deepfake_type: Optional[DeepfakeType],
    ) -> str:
        """Generate detailed explanation of detection results."""
        if not is_deepfake:
            return f"Image appears authentic (confidence: {1-confidence:.1%})"

        explanation = f"Deepfake detected with {confidence:.1%} confidence.\n"

        if deepfake_type:
            explanation += f"Type: {deepfake_type.value}\n"

        # Add top detection indicators
        top_methods = sorted(detection_methods.items(), key=lambda x: x[1], reverse=True)[:3]
        explanation += "Key indicators:\n"
        for method, score in top_methods:
            if score > 0.5:
                explanation += f"• {method.replace('_', ' ').title()}: {score:.1%}\n"

        return explanation.strip()

    # Helper methods
    def _frame_to_temp_image(self, frame: np.ndarray) -> str:
        """Save frame to temporary image file."""
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)
        return temp_file.name

    def _check_temporal_consistency(
        self, prev_landmarks: np.ndarray, curr_landmarks: np.ndarray
    ) -> float:
        """Check temporal consistency between frames."""
        if prev_landmarks.shape != curr_landmarks.shape:
            return 0.5

        # Calculate landmark movement
        movement = np.linalg.norm(curr_landmarks - prev_landmarks, axis=1)
        avg_movement = np.mean(movement)

        # Excessive movement indicates potential manipulation
        if avg_movement > 10:
            return 0.0
        elif avg_movement < 0.1:
            return 0.5  # Too static, might be frozen
        else:
            return 1.0 - (avg_movement / 10)

    def _compute_lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern histogram."""
        # Simplified LBP implementation
        h, w = gray.shape
        lbp = np.zeros_like(gray)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i - 1, j - 1] >= center) << 7
                code |= (gray[i - 1, j] >= center) << 6
                code |= (gray[i - 1, j + 1] >= center) << 5
                code |= (gray[i, j + 1] >= center) << 4
                code |= (gray[i + 1, j + 1] >= center) << 3
                code |= (gray[i + 1, j] >= center) << 2
                code |= (gray[i + 1, j - 1] >= center) << 1
                code |= (gray[i, j - 1] >= center) << 0
                lbp[i, j] = code

        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist / np.sum(hist)

    def _compute_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Compute Gray Level Co-occurrence Matrix features."""
        # Simplified GLCM computation
        levels = 256
        glcm = np.zeros((levels, levels))

        # Compute co-occurrence for horizontal neighbors
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1] - 1):
                glcm[gray[i, j], gray[i, j + 1]] += 1

        # Normalize
        glcm = glcm / np.sum(glcm)

        # Compute features
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(levels)[:, None] - np.arange(levels))))

        return {"homogeneity": homogeneity}

    def _compute_radial_profile(
        self, magnitude_spectrum: np.ndarray, center: Tuple[int, int]
    ) -> np.ndarray:
        """Compute radial profile of frequency spectrum."""
        y, x = np.ogrid[: magnitude_spectrum.shape[0], : magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r = r.astype(int)

        radial_profile = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
        return radial_profile

    def _detect_regular_patterns(self, gx_spectrum: np.ndarray, gy_spectrum: np.ndarray) -> float:
        """Detect regular patterns in frequency domain."""
        # Look for peaks in spectrum
        if SCIPY_AVAILABLE and signal:
            gx_peaks = signal.find_peaks(gx_spectrum.flatten())[0]
            gy_peaks = signal.find_peaks(gy_spectrum.flatten())[0]
        else:
            # Simple peak detection fallback
            gx_flat = gx_spectrum.flatten()
            gy_flat = gy_spectrum.flatten()
            gx_peaks = []
            gy_peaks = []
            for i in range(1, len(gx_flat) - 1):
                if gx_flat[i] > gx_flat[i - 1] and gx_flat[i] > gx_flat[i + 1]:
                    gx_peaks.append(i)
            for i in range(1, len(gy_flat) - 1):
                if gy_flat[i] > gy_flat[i - 1] and gy_flat[i] > gy_flat[i + 1]:
                    gy_peaks.append(i)

        # Many regular peaks suggest GAN artifacts
        pattern_score = (len(gx_peaks) + len(gy_peaks)) / (gx_spectrum.size + gy_spectrum.size)
        return min(pattern_score * 100, 1.0)

    def _detect_color_bleeding(self, image: np.ndarray) -> float:
        """Detect color bleeding artifacts."""
        # Check for color bleeding at edges
        edges = cv2.Canny(image, 100, 200)

        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Check color variance near edges
        edge_mask = dilated_edges > 0

        # Calculate color variance in edge regions
        b, g, r = cv2.split(image)
        edge_color_var = np.var(image[edge_mask])

        # High variance near edges suggests bleeding
        bleeding_score = min(edge_color_var / 1000, 1.0)
        return bleeding_score

    def _compute_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Compute eye aspect ratio from landmarks."""
        if len(landmarks) < 468:
            return 0.25  # Default

        # Simplified eye aspect ratio
        left_eye_height = np.linalg.norm(landmarks[159] - landmarks[145])
        left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])

        if left_eye_width > 0:
            return left_eye_height / left_eye_width
        return 0.25

    def _analyze_skin_consistency(self, face_image: np.ndarray, landmarks: np.ndarray) -> float:
        """Analyze skin texture consistency."""
        # Extract skin regions (forehead, cheeks)
        h, w = face_image.shape[:2]

        # Simple skin region extraction
        skin_regions = []

        # Forehead region (approximate)
        if len(landmarks) > 10:
            forehead_y = int(np.min(landmarks[:, 1]))
            forehead_region = face_image[max(0, forehead_y - 20) : forehead_y + 20, :]
            if forehead_region.size > 0:
                skin_regions.append(forehead_region)

        if not skin_regions:
            return 1.0

        # Analyze texture consistency
        consistencies = []
        for region in skin_regions:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            consistency = 1.0 - (np.std(gray_region) / (np.mean(gray_region) + 1e-6))
            consistencies.append(consistency)

        return np.mean(consistencies) if consistencies else 1.0
