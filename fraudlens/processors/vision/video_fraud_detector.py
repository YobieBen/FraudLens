"""
Advanced Video Fraud Detection Module
Implements video analysis for fraud detection including deepfakes
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import hashlib
from loguru import logger
from enum import Enum
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None
    
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None


class VideoFraudType(Enum):
    """Types of video fraud."""
    DEEPFAKE = "deepfake"
    FACE_SWAP = "face_swap"
    MANIPULATION = "manipulation"
    SYNTHETIC = "synthetic"
    MORPHING = "morphing"
    REENACTMENT = "reenactment"
    COMPRESSION_ARTIFACT = "compression_artifact"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"


@dataclass
class VideoAnalysisResult:
    """Result of video fraud analysis."""
    is_fraudulent: bool
    confidence: float
    fraud_types: List[VideoFraudType]
    frame_scores: List[float]
    temporal_consistency: float
    facial_landmarks_score: float
    frequency_analysis_score: float
    compression_score: float
    deepfake_probability: float
    manipulation_regions: List[Dict[str, Any]]
    suspicious_frames: List[int]
    explanation: str
    metadata: Dict[str, Any]


class DeepfakeDetector(nn.Module):
    """Neural network for deepfake detection."""
    
    def __init__(self, input_size: int = 512):
        super(DeepfakeDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.features(x)


class VideoFraudDetector:
    """Advanced video fraud detection system."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize video fraud detector."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize face detection if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_detection = None
            self.face_mesh = None
            logger.warning("MediaPipe not available, face detection will be limited")
        
        # Initialize deepfake detection model
        self.deepfake_model = DeepfakeDetector().to(self.device)
        if model_path and Path(model_path).exists():
            self.deepfake_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.deepfake_model.eval()
        
        # Initialize dlib for facial landmarks if available
        if DLIB_AVAILABLE:
            try:
                self.detector = dlib.get_frontal_face_detector()
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if Path(predictor_path).exists():
                    self.predictor = dlib.shape_predictor(predictor_path)
                else:
                    self.predictor = None
                    logger.warning("Dlib predictor not found, some features will be limited")
            except Exception as e:
                logger.warning(f"Could not initialize dlib: {e}")
                self.detector = None
                self.predictor = None
        else:
            self.detector = None
            self.predictor = None
            logger.warning("Dlib not available, facial landmark detection will be limited")
        
        # Thresholds
        self.deepfake_threshold = 0.7
        self.manipulation_threshold = 0.6
        self.temporal_threshold = 0.5
        
        logger.info(f"VideoFraudDetector initialized on {self.device}")
    
    async def analyze_video(
        self,
        video_path: str,
        sample_rate: int = 10,
        max_frames: int = 300
    ) -> VideoAnalysisResult:
        """
        Analyze video for fraud detection.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every N frames
            max_frames: Maximum frames to analyze
            
        Returns:
            VideoAnalysisResult with detection details
        """
        logger.info(f"Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize analysis variables
        frame_scores = []
        face_landmarks_history = []
        optical_flow_scores = []
        frequency_scores = []
        manipulation_regions = []
        suspicious_frames = []
        
        frame_count = 0
        analyzed_count = 0
        prev_gray = None
        
        while cap.isOpened() and analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Analyze frame
                frame_result = await self._analyze_frame(
                    frame,
                    frame_count,
                    prev_gray
                )
                
                frame_scores.append(frame_result["deepfake_score"])
                
                if frame_result["face_landmarks"] is not None:
                    face_landmarks_history.append(frame_result["face_landmarks"])
                
                if frame_result["optical_flow_score"] is not None:
                    optical_flow_scores.append(frame_result["optical_flow_score"])
                
                frequency_scores.append(frame_result["frequency_score"])
                
                if frame_result["is_suspicious"]:
                    suspicious_frames.append(frame_count)
                    if frame_result["manipulation_region"]:
                        manipulation_regions.append({
                            "frame": frame_count,
                            "region": frame_result["manipulation_region"],
                            "confidence": frame_result["manipulation_confidence"]
                        })
                
                # Update previous frame
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                analyzed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Compute overall scores
        deepfake_probability = np.mean(frame_scores) if frame_scores else 0.0
        
        # Temporal consistency analysis
        temporal_consistency = self._analyze_temporal_consistency(
            face_landmarks_history,
            optical_flow_scores
        )
        
        # Facial landmarks score
        facial_landmarks_score = self._analyze_facial_landmarks(
            face_landmarks_history
        )
        
        # Frequency analysis score
        frequency_analysis_score = np.mean(frequency_scores) if frequency_scores else 0.0
        
        # Compression artifact analysis
        compression_score = self._analyze_compression_artifacts(video_path)
        
        # Determine fraud types
        fraud_types = []
        if deepfake_probability > self.deepfake_threshold:
            fraud_types.append(VideoFraudType.DEEPFAKE)
        
        if temporal_consistency < self.temporal_threshold:
            fraud_types.append(VideoFraudType.TEMPORAL_INCONSISTENCY)
        
        if len(manipulation_regions) > 0:
            fraud_types.append(VideoFraudType.MANIPULATION)
        
        if compression_score > 0.7:
            fraud_types.append(VideoFraudType.COMPRESSION_ARTIFACT)
        
        # Calculate overall confidence
        confidence_scores = [
            deepfake_probability,
            1 - temporal_consistency,  # Invert for fraud score
            frequency_analysis_score,
            compression_score
        ]
        overall_confidence = np.mean([s for s in confidence_scores if s > 0])
        
        # Determine if fraudulent
        is_fraudulent = (
            deepfake_probability > self.deepfake_threshold or
            temporal_consistency < self.temporal_threshold or
            len(manipulation_regions) > 2
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            is_fraudulent,
            fraud_types,
            deepfake_probability,
            temporal_consistency,
            manipulation_regions
        )
        
        return VideoAnalysisResult(
            is_fraudulent=is_fraudulent,
            confidence=overall_confidence,
            fraud_types=fraud_types,
            frame_scores=frame_scores,
            temporal_consistency=temporal_consistency,
            facial_landmarks_score=facial_landmarks_score,
            frequency_analysis_score=frequency_analysis_score,
            compression_score=compression_score,
            deepfake_probability=deepfake_probability,
            manipulation_regions=manipulation_regions,
            suspicious_frames=suspicious_frames,
            explanation=explanation,
            metadata={
                "fps": fps,
                "total_frames": total_frames,
                "analyzed_frames": analyzed_count,
                "sample_rate": sample_rate
            }
        )
    
    async def _analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        prev_gray: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze a single frame for fraud indicators."""
        result = {
            "deepfake_score": 0.0,
            "face_landmarks": None,
            "optical_flow_score": None,
            "frequency_score": 0.0,
            "is_suspicious": False,
            "manipulation_region": None,
            "manipulation_confidence": 0.0
        }
        
        # Convert to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection and landmark extraction
        if not self.face_mesh:
            # Use OpenCV face detection as fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    face_resized = cv2.resize(face_roi, (224, 224))
                    deepfake_score = self._detect_deepfake(face_resized)
                    result["deepfake_score"] = deepfake_score
                    
                    if deepfake_score > self.deepfake_threshold:
                        result["is_suspicious"] = True
                
                # Create simple landmarks array
                result["face_landmarks"] = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        else:
            face_results = self.face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                # Extract face region for deepfake detection
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Get face bounding box
                h, w = frame.shape[:2]
                landmarks_array = np.array([
                    [lm.x * w, lm.y * h]
                    for lm in face_landmarks.landmark
                ])
                
                x_min, y_min = landmarks_array.min(axis=0).astype(int)
                x_max, y_max = landmarks_array.max(axis=0).astype(int)
                
                # Extract and resize face
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    face_resized = cv2.resize(face_roi, (224, 224))
                    
                    # Run deepfake detection
                    deepfake_score = self._detect_deepfake(face_resized)
                    result["deepfake_score"] = deepfake_score
                    
                    if deepfake_score > self.deepfake_threshold:
                        result["is_suspicious"] = True
                
                result["face_landmarks"] = landmarks_array
        
        # Optical flow analysis
        if prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_score = self._analyze_optical_flow(prev_gray, gray)
            result["optical_flow_score"] = flow_score
            
            if flow_score > 0.7:
                result["is_suspicious"] = True
        
        # Frequency domain analysis
        frequency_score = self._analyze_frequency_domain(frame)
        result["frequency_score"] = frequency_score
        
        # Manipulation detection
        manipulation_mask = self._detect_manipulation(frame)
        if manipulation_mask is not None:
            contours, _ = cv2.findContours(
                manipulation_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > 1000:  # Significant manipulation area
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    result["manipulation_region"] = (x, y, w, h)
                    result["manipulation_confidence"] = min(area / (frame.shape[0] * frame.shape[1]), 1.0)
                    result["is_suspicious"] = True
        
        return result
    
    def _detect_deepfake(self, face_image: np.ndarray) -> float:
        """Detect deepfake using neural network."""
        try:
            # Preprocess image
            face_tensor = torch.from_numpy(face_image).float()
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Extract features (simplified)
            features = face_tensor.flatten(1).mean(dim=1, keepdim=True)
            
            # Pad to expected input size
            if features.shape[1] < 512:
                features = F.pad(features, (0, 512 - features.shape[1]))
            
            # Run model
            with torch.no_grad():
                output = self.deepfake_model(features)
                probabilities = F.softmax(output, dim=1)
                deepfake_prob = probabilities[0, 1].item()
            
            return deepfake_prob
            
        except Exception as e:
            logger.warning(f"Error in deepfake detection: {e}")
            return 0.0
    
    def _analyze_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> float:
        """Analyze optical flow for temporal inconsistencies."""
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Look for unusual patterns
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            
            # High variance in flow might indicate manipulation
            if std_magnitude > mean_magnitude * 2:
                return min(std_magnitude / (mean_magnitude + 1e-6), 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error in optical flow analysis: {e}")
            return 0.0
    
    def _analyze_frequency_domain(self, frame: np.ndarray) -> float:
        """Analyze frequency domain for signs of manipulation."""
        try:
            if not SCIPY_AVAILABLE:
                # Fallback to simple frequency analysis using numpy
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.abs(f_shift)
                
                # Simple high frequency detection
                center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
                radius = min(center_y, center_x) // 2
                
                high_freq_area = magnitude_spectrum.copy()
                y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                high_freq_area[mask] = 0
                
                high_freq_ratio = np.sum(high_freq_area) / (np.sum(magnitude_spectrum) + 1e-6)
                return min(high_freq_ratio * 10, 1.0)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT
            f_transform = fft(gray.flatten())
            frequencies = fftfreq(len(f_transform))
            
            # Get power spectrum
            power_spectrum = np.abs(f_transform) ** 2
            
            # Look for unusual frequency patterns
            high_freq_power = np.sum(power_spectrum[np.abs(frequencies) > 0.4])
            total_power = np.sum(power_spectrum)
            
            # High frequency artifacts might indicate manipulation
            high_freq_ratio = high_freq_power / (total_power + 1e-6)
            
            return min(high_freq_ratio * 10, 1.0)  # Scale and cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error in frequency analysis: {e}")
            return 0.0
    
    def _detect_manipulation(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect manipulated regions in frame."""
        try:
            # Edge detection for finding inconsistencies
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Sobel operator
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute gradient magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Look for inconsistent edges
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude)
            
            # Threshold for anomalies
            threshold = mean_mag + 2 * std_mag
            manipulation_mask = (magnitude > threshold).astype(np.uint8) * 255
            
            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            manipulation_mask = cv2.morphologyEx(manipulation_mask, cv2.MORPH_CLOSE, kernel)
            manipulation_mask = cv2.morphologyEx(manipulation_mask, cv2.MORPH_OPEN, kernel)
            
            return manipulation_mask
            
        except Exception as e:
            logger.warning(f"Error in manipulation detection: {e}")
            return None
    
    def _analyze_temporal_consistency(
        self,
        face_landmarks_history: List[np.ndarray],
        optical_flow_scores: List[float]
    ) -> float:
        """Analyze temporal consistency across frames."""
        if len(face_landmarks_history) < 2:
            return 1.0  # Not enough data
        
        # Calculate landmark movement consistency
        landmark_distances = []
        for i in range(1, len(face_landmarks_history)):
            prev_landmarks = face_landmarks_history[i-1]
            curr_landmarks = face_landmarks_history[i]
            
            if prev_landmarks.shape == curr_landmarks.shape:
                distance = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
                landmark_distances.append(distance)
        
        if landmark_distances:
            # Look for sudden jumps
            mean_distance = np.mean(landmark_distances)
            std_distance = np.std(landmark_distances)
            
            # Count anomalies
            anomalies = sum(1 for d in landmark_distances if d > mean_distance + 2 * std_distance)
            consistency_score = 1.0 - (anomalies / len(landmark_distances))
            
            # Factor in optical flow
            if optical_flow_scores:
                flow_consistency = 1.0 - np.mean(optical_flow_scores)
                consistency_score = (consistency_score + flow_consistency) / 2
            
            return consistency_score
        
        return 1.0
    
    def _analyze_facial_landmarks(
        self,
        face_landmarks_history: List[np.ndarray]
    ) -> float:
        """Analyze facial landmarks for authenticity."""
        if not face_landmarks_history:
            return 0.0
        
        # Check landmark stability and natural movement
        scores = []
        
        for landmarks in face_landmarks_history:
            if landmarks is not None and len(landmarks) > 0:
                # Check for symmetry
                left_eye_region = landmarks[33:42]  # Approximate eye regions
                right_eye_region = landmarks[133:142]
                
                if len(left_eye_region) > 0 and len(right_eye_region) > 0:
                    # Calculate symmetry
                    left_center = np.mean(left_eye_region, axis=0)
                    right_center = np.mean(right_eye_region, axis=0)
                    face_center = np.mean(landmarks, axis=0)
                    
                    left_dist = np.linalg.norm(left_center - face_center)
                    right_dist = np.linalg.norm(right_center - face_center)
                    
                    symmetry = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
                    scores.append(symmetry)
        
        return np.mean(scores) if scores else 0.0
    
    def _analyze_compression_artifacts(self, video_path: str) -> float:
        """Analyze video for compression artifacts indicating manipulation."""
        try:
            # Quick analysis of video compression
            cap = cv2.VideoCapture(video_path)
            
            artifact_scores = []
            frame_count = 0
            max_frames = 10  # Sample a few frames
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check for blockiness (JPEG-like artifacts)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect 8x8 block artifacts
                h, w = gray.shape
                block_size = 8
                
                block_diffs = []
                for i in range(0, h - block_size, block_size):
                    for j in range(0, w - block_size, block_size):
                        block = gray[i:i+block_size, j:j+block_size]
                        
                        # Check variance within block
                        block_var = np.var(block)
                        
                        # Check edge strength at block boundaries
                        if i > 0:
                            top_edge = np.mean(np.abs(gray[i-1, j:j+block_size] - gray[i, j:j+block_size]))
                            block_diffs.append(top_edge / (block_var + 1e-6))
                
                if block_diffs:
                    # High ratios indicate block artifacts
                    artifact_score = np.percentile(block_diffs, 90)
                    artifact_scores.append(min(artifact_score / 10, 1.0))
                
                frame_count += 1
            
            cap.release()
            
            return np.mean(artifact_scores) if artifact_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error in compression analysis: {e}")
            return 0.0
    
    def _generate_explanation(
        self,
        is_fraudulent: bool,
        fraud_types: List[VideoFraudType],
        deepfake_probability: float,
        temporal_consistency: float,
        manipulation_regions: List[Dict]
    ) -> str:
        """Generate human-readable explanation of analysis results."""
        if not is_fraudulent:
            return "Video appears to be authentic. No significant signs of manipulation or deepfake detected."
        
        explanation_parts = ["Video analysis detected potential fraud:"]
        
        if VideoFraudType.DEEPFAKE in fraud_types:
            explanation_parts.append(
                f"• Deepfake indicators detected (confidence: {deepfake_probability:.1%})"
            )
        
        if VideoFraudType.TEMPORAL_INCONSISTENCY in fraud_types:
            explanation_parts.append(
                f"• Temporal inconsistencies found (consistency score: {temporal_consistency:.1%})"
            )
        
        if VideoFraudType.MANIPULATION in fraud_types:
            explanation_parts.append(
                f"• {len(manipulation_regions)} manipulated regions detected"
            )
        
        if VideoFraudType.COMPRESSION_ARTIFACT in fraud_types:
            explanation_parts.append(
                "• Compression artifacts suggest potential editing"
            )
        
        return "\n".join(explanation_parts)