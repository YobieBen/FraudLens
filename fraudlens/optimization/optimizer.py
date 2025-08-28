"""
FraudLens Model Optimization Pipeline
Supports quantization, pruning, and performance optimization for Mac M4 deployment
"""

import os
import json
import time
import psutil
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Model optimization libraries
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import prune
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


@dataclass
class MemoryProfile:
    """Memory usage profile for a model"""
    model_size_mb: float
    peak_memory_mb: float
    average_memory_mb: float
    memory_timeline: List[float]
    optimization_potential: float
    recommendations: List[str]


@dataclass
class LatencyReport:
    """Latency analysis report"""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    device_utilization: Dict[str, float]
    bottlenecks: List[str]


@dataclass
class Optimization:
    """Optimization recommendation"""
    name: str
    category: str  # "memory", "latency", "accuracy"
    impact: str  # "high", "medium", "low"
    description: str
    implementation: str
    estimated_improvement: float


class FraudLensOptimizer:
    """
    Model optimization for efficient deployment on Mac M4 and cloud
    """
    
    def __init__(self, cache_dir: str = "./optimization_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration for Mac M4
        self.device_config = self._detect_devices()
        
        # Optimization history
        self.optimization_history = []
        
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available compute devices"""
        config = {
            "cpu": True,
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu": False,
            "metal": False,
            "neural_engine": False
        }
        
        if TORCH_AVAILABLE:
            config["gpu"] = torch.cuda.is_available()
            if hasattr(torch.backends, 'mps'):
                config["metal"] = torch.backends.mps.is_available()
        
        # Check for Neural Engine (M4 specific)
        if COREML_AVAILABLE:
            config["neural_engine"] = self._check_neural_engine()
            
        return config
    
    def _check_neural_engine(self) -> bool:
        """Check if Apple Neural Engine is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def quantize_model(
        self, 
        model_path: str, 
        bits: int = 8,
        calibration_data: Optional[np.ndarray] = None
    ) -> str:
        """
        Quantize model to reduce size and improve inference speed
        
        Args:
            model_path: Path to the model file
            bits: Quantization bits (4, 8, 16)
            calibration_data: Optional calibration dataset
            
        Returns:
            Path to quantized model
        """
        model_path = Path(model_path)
        output_path = self.cache_dir / f"{model_path.stem}_int{bits}.onnx"
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX runtime not available for quantization")
        
        print(f"Quantizing model to INT{bits}...")
        
        # Dynamic quantization for ONNX models
        if model_path.suffix == '.onnx':
            quantize_dynamic(
                model_path,
                output_path,
                weight_type=QuantType.QInt8 if bits == 8 else QuantType.QUInt8
            )
            
            # Verify quantized model
            original_size = model_path.stat().st_size / (1024**2)
            quantized_size = output_path.stat().st_size / (1024**2)
            compression_ratio = original_size / quantized_size
            
            print(f"✅ Quantization complete:")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            
            # Log optimization
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "operation": "quantization",
                "model": str(model_path),
                "bits": bits,
                "original_size_mb": original_size,
                "optimized_size_mb": quantized_size,
                "compression_ratio": compression_ratio
            })
            
        # PyTorch quantization
        elif model_path.suffix in ['.pt', '.pth'] and TORCH_AVAILABLE:
            model = torch.load(model_path, map_location='cpu')
            
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8 if bits == 8 else torch.float16
            )
            
            output_path = output_path.with_suffix('.pt')
            torch.save(quantized_model, output_path)
            
        return str(output_path)
    
    def convert_to_coreml(self, model_path: str) -> str:
        """
        Convert model to CoreML for Apple Neural Engine acceleration
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML tools not available")
        
        model_path = Path(model_path)
        output_path = self.cache_dir / f"{model_path.stem}.mlmodel"
        
        if TORCH_AVAILABLE and model_path.suffix in ['.pt', '.pth']:
            # Load PyTorch model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Trace model
            example_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, example_input)
            
            # Convert to CoreML
            ml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.ALL  # Use all available compute units
            )
            
            ml_model.save(str(output_path))
            print(f"✅ Converted to CoreML: {output_path}")
            
        return str(output_path)
    
    def profile_memory_usage(
        self,
        model_path: str,
        test_inputs: List[np.ndarray],
        duration_seconds: int = 60
    ) -> MemoryProfile:
        """
        Profile memory usage during model inference
        """
        tracemalloc.start()
        memory_timeline = []
        
        # Load model based on format
        model = self._load_model(model_path)
        
        # Get initial memory
        initial_memory = tracemalloc.get_traced_memory()[0] / (1024**2)
        
        # Run inference for duration
        start_time = time.time()
        inference_count = 0
        
        while time.time() - start_time < duration_seconds:
            for input_data in test_inputs:
                # Run inference
                _ = self._run_inference(model, input_data)
                inference_count += 1
                
                # Track memory
                current, peak = tracemalloc.get_traced_memory()
                memory_timeline.append(current / (1024**2))
        
        tracemalloc.stop()
        
        # Analyze memory usage
        peak_memory = max(memory_timeline)
        avg_memory = np.mean(memory_timeline)
        model_size = Path(model_path).stat().st_size / (1024**2)
        
        # Calculate optimization potential
        memory_variance = np.std(memory_timeline)
        optimization_potential = min(1.0, memory_variance / avg_memory)
        
        # Generate recommendations
        recommendations = []
        if optimization_potential > 0.3:
            recommendations.append("High memory variance detected - consider memory pooling")
        if peak_memory > avg_memory * 1.5:
            recommendations.append("Memory spikes detected - implement gradual batch processing")
        if model_size > 100:
            recommendations.append("Large model size - consider quantization or pruning")
        
        return MemoryProfile(
            model_size_mb=model_size,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_timeline=memory_timeline,
            optimization_potential=optimization_potential,
            recommendations=recommendations
        )
    
    def optimize_batch_size(
        self,
        model_path: str,
        test_inputs: List[np.ndarray],
        target_latency_ms: float = 100,
        max_memory_mb: float = 1024
    ) -> int:
        """
        Find optimal batch size for given constraints
        """
        model = self._load_model(model_path)
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results = []
        
        for batch_size in batch_sizes:
            # Create batched input
            if len(test_inputs) >= batch_size:
                batch = np.stack(test_inputs[:batch_size])
            else:
                batch = np.repeat(test_inputs[0][np.newaxis, ...], batch_size, axis=0)
            
            # Measure latency
            start_time = time.perf_counter()
            tracemalloc.start()
            
            try:
                _ = self._run_inference(model, batch)
                latency_ms = (time.perf_counter() - start_time) * 1000
                memory_mb = tracemalloc.get_traced_memory()[0] / (1024**2)
                tracemalloc.stop()
                
                results.append({
                    "batch_size": batch_size,
                    "latency_ms": latency_ms,
                    "memory_mb": memory_mb,
                    "throughput": batch_size / (latency_ms / 1000)
                })
                
                # Check constraints
                if latency_ms > target_latency_ms or memory_mb > max_memory_mb:
                    break
                    
            except Exception as e:
                print(f"Batch size {batch_size} failed: {e}")
                break
        
        # Find optimal batch size
        valid_results = [
            r for r in results 
            if r["latency_ms"] <= target_latency_ms and r["memory_mb"] <= max_memory_mb
        ]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x["throughput"])
            optimal_batch_size = optimal["batch_size"]
            print(f"✅ Optimal batch size: {optimal_batch_size}")
            print(f"  Latency: {optimal['latency_ms']:.2f}ms")
            print(f"  Memory: {optimal['memory_mb']:.2f}MB")
            print(f"  Throughput: {optimal['throughput']:.2f} samples/sec")
        else:
            optimal_batch_size = 1
            print("⚠️ Using batch size 1 due to constraints")
        
        return optimal_batch_size
    
    def benchmark_latency(
        self,
        model_path: str,
        test_inputs: List[np.ndarray],
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> LatencyReport:
        """
        Comprehensive latency benchmarking
        """
        model = self._load_model(model_path)
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup_runs):
            self._run_inference(model, test_inputs[0])
        
        # Benchmark
        print(f"Running {benchmark_runs} benchmark iterations...")
        latencies = []
        cpu_usage = []
        memory_usage = []
        
        for i in range(benchmark_runs):
            input_data = test_inputs[i % len(test_inputs)]
            
            # Monitor resources
            cpu_before = psutil.cpu_percent()
            mem_before = psutil.virtual_memory().percent
            
            # Measure latency
            start_time = time.perf_counter()
            _ = self._run_inference(model, input_data)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            latencies.append(latency_ms)
            cpu_usage.append(psutil.cpu_percent() - cpu_before)
            memory_usage.append(psutil.virtual_memory().percent - mem_before)
        
        # Calculate statistics
        latencies_sorted = sorted(latencies)
        
        report = LatencyReport(
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_qps=1000 / np.mean(latencies),
            device_utilization={
                "cpu": np.mean(cpu_usage),
                "memory": np.mean(memory_usage),
                "gpu": 0.0,  # TODO: Add GPU monitoring
                "metal": 0.0  # TODO: Add Metal monitoring
            },
            bottlenecks=self._identify_bottlenecks(latencies, cpu_usage, memory_usage)
        )
        
        return report
    
    def suggest_optimizations(
        self,
        model_path: str,
        profile: Optional[MemoryProfile] = None,
        latency: Optional[LatencyReport] = None
    ) -> List[Optimization]:
        """
        Suggest optimizations based on profiling results
        """
        optimizations = []
        model_size = Path(model_path).stat().st_size / (1024**2)
        
        # Model size optimizations
        if model_size > 100:
            optimizations.append(Optimization(
                name="Model Quantization",
                category="memory",
                impact="high",
                description="Reduce model size by 4x with INT8 quantization",
                implementation="optimizer.quantize_model(model_path, bits=8)",
                estimated_improvement=0.75
            ))
            
            optimizations.append(Optimization(
                name="Model Pruning",
                category="memory",
                impact="medium",
                description="Remove redundant weights through structured pruning",
                implementation="optimizer.prune_model(model_path, sparsity=0.5)",
                estimated_improvement=0.5
            ))
        
        # Memory optimizations
        if profile and profile.optimization_potential > 0.3:
            optimizations.append(Optimization(
                name="Memory Pooling",
                category="memory",
                impact="medium",
                description="Implement memory pooling to reduce allocation overhead",
                implementation="Enable memory pooling in inference engine",
                estimated_improvement=0.3
            ))
        
        # Latency optimizations
        if latency:
            if latency.avg_latency_ms > 100:
                optimizations.append(Optimization(
                    name="Hardware Acceleration",
                    category="latency",
                    impact="high",
                    description="Use Apple Neural Engine for 10x speedup",
                    implementation="optimizer.convert_to_coreml(model_path)",
                    estimated_improvement=0.9
                ))
            
            if latency.device_utilization["cpu"] > 80:
                optimizations.append(Optimization(
                    name="Batch Processing",
                    category="latency",
                    impact="medium",
                    description="Process multiple inputs simultaneously",
                    implementation="optimizer.optimize_batch_size(model_path)",
                    estimated_improvement=0.4
                ))
        
        # Platform-specific optimizations
        if self.device_config["metal"]:
            optimizations.append(Optimization(
                name="Metal Performance Shaders",
                category="latency",
                impact="high",
                description="Use MPS for GPU acceleration on Mac",
                implementation="Enable MPS backend in PyTorch",
                estimated_improvement=0.7
            ))
        
        if self.device_config["neural_engine"]:
            optimizations.append(Optimization(
                name="Neural Engine Optimization",
                category="latency",
                impact="high",
                description="Optimize model for Apple Neural Engine",
                implementation="Convert to CoreML with ANE optimization",
                estimated_improvement=0.8
            ))
        
        # Sort by impact and improvement
        optimizations.sort(
            key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}[x.impact],
                x.estimated_improvement
            ),
            reverse=True
        )
        
        return optimizations
    
    def prune_model(
        self,
        model_path: str,
        sparsity: float = 0.5,
        structured: bool = True
    ) -> str:
        """
        Prune model weights to reduce size
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for pruning")
        
        model = torch.load(model_path, map_location='cpu')
        output_path = self.cache_dir / f"{Path(model_path).stem}_pruned_{int(sparsity*100)}.pt"
        
        # Apply pruning to linear and conv layers
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if structured:
                    prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
                else:
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        # Save pruned model
        torch.save(model, output_path)
        
        original_size = Path(model_path).stat().st_size / (1024**2)
        pruned_size = output_path.stat().st_size / (1024**2)
        
        print(f"✅ Model pruned with {sparsity*100:.0f}% sparsity")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Pruned size: {pruned_size:.2f} MB")
        
        return str(output_path)
    
    def _load_model(self, model_path: str):
        """Load model based on format"""
        path = Path(model_path)
        
        if path.suffix == '.onnx' and ONNX_AVAILABLE:
            return ort.InferenceSession(str(path))
        elif path.suffix in ['.pt', '.pth'] and TORCH_AVAILABLE:
            return torch.load(path, map_location='cpu')
        elif path.suffix == '.mlmodel' and COREML_AVAILABLE:
            return ct.models.MLModel(str(path))
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")
    
    def _run_inference(self, model, input_data: np.ndarray):
        """Run inference based on model type"""
        if hasattr(model, 'run'):  # ONNX
            return model.run(None, {'input': input_data})
        elif TORCH_AVAILABLE and isinstance(model, nn.Module):  # PyTorch
            with torch.no_grad():
                tensor_input = torch.from_numpy(input_data).float()
                return model(tensor_input).numpy()
        elif hasattr(model, 'predict'):  # CoreML
            return model.predict({'input': input_data})
        else:
            raise ValueError("Unknown model type")
    
    def _identify_bottlenecks(
        self,
        latencies: List[float],
        cpu_usage: List[float],
        memory_usage: List[float]
    ) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # High latency variance indicates inconsistent performance
        if np.std(latencies) > np.mean(latencies) * 0.3:
            bottlenecks.append("High latency variance - consider request batching")
        
        # CPU bottleneck
        if np.mean(cpu_usage) > 70:
            bottlenecks.append("CPU bottleneck - consider hardware acceleration")
        
        # Memory pressure
        if np.mean(memory_usage) > 80:
            bottlenecks.append("Memory pressure - implement model swapping or quantization")
        
        # Tail latency issues
        p99 = np.percentile(latencies, 99)
        p50 = np.percentile(latencies, 50)
        if p99 > p50 * 3:
            bottlenecks.append("Tail latency issues - implement request timeout and retry")
        
        return bottlenecks
    
    def export_optimization_report(self, output_path: str):
        """Export detailed optimization report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "device_config": self.device_config,
            "optimization_history": self.optimization_history,
            "recommendations": []
        }
        
        # Add general recommendations based on platform
        if self.device_config["metal"]:
            report["recommendations"].append(
                "Mac with Metal detected - use MPS backend for 5-10x speedup"
            )
        
        if self.device_config["neural_engine"]:
            report["recommendations"].append(
                "Apple Neural Engine available - convert models to CoreML format"
            )
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Optimization report exported to {output_path}")