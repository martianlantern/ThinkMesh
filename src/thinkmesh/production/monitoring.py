"""
Performance monitoring and metrics collection for ThinkMesh.
"""
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_gb: float
    gpu_memory_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    active_processes: int = 0
    tokens_per_second: float = 0.0
    problems_per_second: float = 0.0
    average_confidence: float = 0.0
    success_rate: float = 0.0


@dataclass
class ExecutionStats:
    """Statistics for a single execution."""
    start_time: float
    end_time: float
    strategy_name: str
    model_name: str
    parallel_count: int
    total_tokens: int
    problems_solved: int
    successful_problems: int
    average_confidence: float
    peak_memory_gb: float
    error_count: int = 0
    errors: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates ThinkMesh metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.execution_history: deque = deque(maxlen=max_history)
        self.performance_history: deque = deque(maxlen=max_history * 10)  # More frequent samples
        self.strategy_stats = defaultdict(list)
        self.model_stats = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_execution(self, stats: ExecutionStats):
        """Record execution statistics."""
        with self._lock:
            self.execution_history.append(stats)
            self.strategy_stats[stats.strategy_name].append(stats)
            self.model_stats[stats.model_name].append(stats)
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.performance_history.append(metrics)
    
    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get summary statistics for a strategy."""
        with self._lock:
            stats = self.strategy_stats[strategy_name]
            
            if not stats:
                return {"error": "No data available for strategy"}
            
            return {
                "executions": len(stats),
                "avg_execution_time": sum(s.end_time - s.start_time for s in stats) / len(stats),
                "avg_tokens": sum(s.total_tokens for s in stats) / len(stats),
                "avg_confidence": sum(s.average_confidence for s in stats) / len(stats),
                "avg_success_rate": sum(s.successful_problems / max(s.problems_solved, 1) for s in stats) / len(stats),
                "total_errors": sum(s.error_count for s in stats),
                "avg_parallel_count": sum(s.parallel_count for s in stats) / len(stats)
            }
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for a model."""
        with self._lock:
            stats = self.model_stats[model_name]
            
            if not stats:
                return {"error": "No data available for model"}
            
            return {
                "executions": len(stats),
                "avg_execution_time": sum(s.end_time - s.start_time for s in stats) / len(stats),
                "avg_peak_memory": sum(s.peak_memory_gb for s in stats) / len(stats),
                "total_tokens": sum(s.total_tokens for s in stats),
                "avg_confidence": sum(s.average_confidence for s in stats) / len(stats),
                "success_rate": sum(s.successful_problems for s in stats) / sum(s.problems_solved for s in stats) if any(s.problems_solved for s in stats) else 0
            }
    
    def get_recent_performance(self, minutes: int = 10) -> List[PerformanceMetrics]:
        """Get performance metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            return [m for m in self.performance_history if m.timestamp >= cutoff_time]
    
    def export_metrics(self, filepath: Path):
        """Export all metrics to JSON file."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "execution_history": [
                    {
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "strategy_name": s.strategy_name,
                        "model_name": s.model_name,
                        "parallel_count": s.parallel_count,
                        "total_tokens": s.total_tokens,
                        "problems_solved": s.problems_solved,
                        "successful_problems": s.successful_problems,
                        "average_confidence": s.average_confidence,
                        "peak_memory_gb": s.peak_memory_gb,
                        "error_count": s.error_count,
                        "errors": s.errors
                    }
                    for s in self.execution_history
                ],
                "performance_history": [
                    {
                        "timestamp": m.timestamp,
                        "cpu_usage_percent": m.cpu_usage_percent,
                        "memory_usage_gb": m.memory_usage_gb,
                        "gpu_memory_gb": m.gpu_memory_gb,
                        "gpu_utilization_percent": m.gpu_utilization_percent,
                        "active_processes": m.active_processes,
                        "tokens_per_second": m.tokens_per_second,
                        "problems_per_second": m.problems_per_second,
                        "average_confidence": m.average_confidence,
                        "success_rate": m.success_rate
                    }
                    for m in self.performance_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class PerformanceMonitor:
    """Monitors ThinkMesh performance in real-time."""
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 sample_interval: float = 5.0,
                 alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.sample_interval = sample_interval
        self.alert_callback = alert_callback
        self._monitoring = False
        self._monitor_thread = None
        
        # Alert thresholds
        self.cpu_threshold = 90.0  # %
        self.memory_threshold = 90.0  # %
        self.gpu_memory_threshold = 95.0  # %
        self.error_rate_threshold = 0.5  # 50% error rate
        
        self._current_stats = None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.sample_interval * 2)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_collector.record_performance(metrics)
                self._check_alerts(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                # Don't let monitoring failures crash the main process
                print(f"Monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        
        # GPU metrics (if available)
        gpu_memory_gb = None
        gpu_utilization = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                # GPU utilization would require nvidia-ml-py
        except ImportError:
            pass
        
        # Process-specific metrics
        active_processes = len(psutil.pids())
        
        # ThinkMesh-specific metrics from current execution
        tokens_per_second = 0.0
        problems_per_second = 0.0
        average_confidence = 0.0
        success_rate = 1.0
        
        if self._current_stats:
            elapsed_time = time.time() - self._current_stats.start_time
            if elapsed_time > 0:
                tokens_per_second = self._current_stats.total_tokens / elapsed_time
                problems_per_second = self._current_stats.problems_solved / elapsed_time
                average_confidence = self._current_stats.average_confidence
                success_rate = (self._current_stats.successful_problems / 
                              max(self._current_stats.problems_solved, 1))
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=cpu_percent,
            memory_usage_gb=memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            gpu_utilization_percent=gpu_utilization,
            active_processes=active_processes,
            tokens_per_second=tokens_per_second,
            problems_per_second=problems_per_second,
            average_confidence=average_confidence,
            success_rate=success_rate
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions."""
        if not self.alert_callback:
            return
        
        # High CPU usage
        if metrics.cpu_usage_percent > self.cpu_threshold:
            self.alert_callback("high_cpu", {
                "cpu_percent": metrics.cpu_usage_percent,
                "threshold": self.cpu_threshold
            })
        
        # High memory usage
        memory_percent = (metrics.memory_usage_gb / psutil.virtual_memory().total) * 100
        if memory_percent > self.memory_threshold:
            self.alert_callback("high_memory", {
                "memory_percent": memory_percent,
                "memory_gb": metrics.memory_usage_gb,
                "threshold": self.memory_threshold
            })
        
        # High GPU memory usage
        if metrics.gpu_memory_gb:
            try:
                import torch
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_percent = (metrics.gpu_memory_gb / total_gpu_memory) * 100
                
                if gpu_percent > self.gpu_memory_threshold:
                    self.alert_callback("high_gpu_memory", {
                        "gpu_memory_percent": gpu_percent,
                        "gpu_memory_gb": metrics.gpu_memory_gb,
                        "threshold": self.gpu_memory_threshold
                    })
            except ImportError:
                pass
        
        # Low success rate
        if metrics.success_rate < (1.0 - self.error_rate_threshold):
            self.alert_callback("high_error_rate", {
                "success_rate": metrics.success_rate,
                "error_rate": 1.0 - metrics.success_rate,
                "threshold": self.error_rate_threshold
            })
    
    def start_execution_tracking(self, strategy_name: str, model_name: str, parallel_count: int):
        """Start tracking a new execution."""
        self._current_stats = ExecutionStats(
            start_time=time.time(),
            end_time=0,
            strategy_name=strategy_name,
            model_name=model_name,
            parallel_count=parallel_count,
            total_tokens=0,
            problems_solved=0,
            successful_problems=0,
            average_confidence=0.0,
            peak_memory_gb=0.0
        )
    
    def update_execution_progress(self, 
                                tokens_used: int = 0,
                                problems_completed: int = 0,
                                successful: int = 0,
                                confidence: float = 0.0,
                                error: Optional[str] = None):
        """Update current execution progress."""
        if not self._current_stats:
            return
        
        self._current_stats.total_tokens += tokens_used
        self._current_stats.problems_solved += problems_completed
        self._current_stats.successful_problems += successful
        
        if confidence > 0:
            # Update running average confidence
            total_problems = self._current_stats.problems_solved
            if total_problems > 0:
                self._current_stats.average_confidence = (
                    (self._current_stats.average_confidence * (total_problems - 1) + confidence) / total_problems
                )
        
        # Track peak memory usage
        try:
            import torch
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / (1024**3)
                self._current_stats.peak_memory_gb = max(
                    self._current_stats.peak_memory_gb, 
                    current_memory
                )
        except ImportError:
            pass
        
        # Track errors
        if error:
            self._current_stats.error_count += 1
            self._current_stats.errors.append(error)
    
    def finish_execution_tracking(self):
        """Finish tracking current execution."""
        if not self._current_stats:
            return
        
        self._current_stats.end_time = time.time()
        self.metrics_collector.record_execution(self._current_stats)
        self._current_stats = None
    
    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance snapshot."""
        return self._collect_metrics()
    
    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes."""
        recent_metrics = self.metrics_collector.get_recent_performance(minutes)
        
        if not recent_metrics:
            return {"error": "No performance data available"}
        
        return {
            "time_period_minutes": minutes,
            "samples": len(recent_metrics),
            "avg_cpu_percent": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            "max_cpu_percent": max(m.cpu_usage_percent for m in recent_metrics),
            "avg_memory_gb": sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics),
            "max_memory_gb": max(m.memory_usage_gb for m in recent_metrics),
            "avg_gpu_memory_gb": sum(m.gpu_memory_gb for m in recent_metrics if m.gpu_memory_gb) / max(1, sum(1 for m in recent_metrics if m.gpu_memory_gb)),
            "max_gpu_memory_gb": max((m.gpu_memory_gb for m in recent_metrics if m.gpu_memory_gb), default=0),
            "avg_tokens_per_second": sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
            "max_tokens_per_second": max(m.tokens_per_second for m in recent_metrics),
            "avg_success_rate": sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        }
