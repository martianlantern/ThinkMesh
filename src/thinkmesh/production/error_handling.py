"""
Comprehensive error handling for ThinkMesh.
"""
import traceback
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    model_name: Optional[str] = None
    strategy_name: Optional[str] = None
    parallel_count: Optional[int] = None
    step: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None


class ThinkMeshError(Exception):
    """Base exception for ThinkMesh errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.traceback_str = traceback.format_exc() if cause else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "severity": self.severity.value,
            "context": {
                "model_name": self.context.model_name,
                "strategy_name": self.context.strategy_name, 
                "parallel_count": self.context.parallel_count,
                "step": self.context.step,
                "config": self.context.config,
                "system_info": self.context.system_info
            },
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str
        }


class ConfigurationError(ThinkMeshError):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.field = field


class ModelError(ThinkMeshError):
    """Raised when there's a model-related error."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        if model_name:
            context.model_name = model_name
        kwargs['context'] = context
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class StrategyError(ThinkMeshError):
    """Raised when there's a strategy execution error."""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        if strategy_name:
            context.strategy_name = strategy_name
        kwargs['context'] = context
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)


class ResourceError(ThinkMeshError):
    """Raised when there are resource constraints (memory, compute, etc)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)
        self.resource_type = resource_type


class TimeoutError(ThinkMeshError):
    """Raised when operations exceed time limits."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.timeout_seconds = timeout_seconds


class ValidationError(ThinkMeshError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        super().__init__(message, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.validation_type = validation_type


def handle_model_loading_error(e: Exception, model_name: str) -> ModelError:
    """Convert model loading exceptions to ModelError."""
    if "CUDA out of memory" in str(e).lower():
        return ModelError(
            f"GPU out of memory loading model '{model_name}'. Try smaller batch_size or use CPU.",
            model_name=model_name,
            cause=e
        )
    elif "not found" in str(e).lower():
        return ModelError(
            f"Model '{model_name}' not found. Check model name and internet connection.",
            model_name=model_name,
            cause=e
        )
    else:
        return ModelError(
            f"Failed to load model '{model_name}': {str(e)}",
            model_name=model_name,
            cause=e
        )


def handle_strategy_error(e: Exception, strategy_name: str, step: Optional[int] = None) -> StrategyError:
    """Convert strategy execution exceptions to StrategyError."""
    context = ErrorContext(strategy_name=strategy_name, step=step)
    
    if "timeout" in str(e).lower():
        return StrategyError(
            f"Strategy '{strategy_name}' timed out at step {step}. Consider increasing budget or reducing parallel count.",
            context=context,
            cause=e
        )
    elif "memory" in str(e).lower():
        return StrategyError(
            f"Strategy '{strategy_name}' ran out of memory. Try reducing parallel count or batch size.",
            context=context,
            cause=e
        )
    else:
        return StrategyError(
            f"Strategy '{strategy_name}' failed: {str(e)}",
            context=context,
            cause=e
        )


def handle_resource_error(e: Exception, resource_type: str = "unknown") -> ResourceError:
    """Convert resource-related exceptions to ResourceError."""
    if "memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
        return ResourceError(
            f"GPU memory exhausted. Try: 1) Reduce batch_size, 2) Reduce parallel count, 3) Use smaller model, 4) Use CPU",
            resource_type="gpu_memory",
            cause=e
        )
    elif "disk" in str(e).lower():
        return ResourceError(
            f"Disk space issue: {str(e)}",
            resource_type="disk",
            cause=e
        )
    else:
        return ResourceError(
            f"Resource constraint: {str(e)}",
            resource_type=resource_type,
            cause=e
        )


class ErrorHandler:
    """Centralized error handling for ThinkMesh."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def handle_exception(self, e: Exception, context: Optional[ErrorContext] = None) -> ThinkMeshError:
        """Convert generic exceptions to ThinkMesh errors with context."""
        
        # Check for specific error types
        if isinstance(e, ThinkMeshError):
            return e
        
        error_str = str(e).lower()
        
        # GPU memory errors
        if "cuda out of memory" in error_str:
            return handle_resource_error(e, "gpu_memory")
        
        # Model loading errors
        elif "could not load" in error_str or "failed to load" in error_str:
            model_name = context.model_name if context else "unknown"
            return handle_model_loading_error(e, model_name)
        
        # Timeout errors  
        elif "timeout" in error_str or "timed out" in error_str:
            return TimeoutError(
                f"Operation timed out: {str(e)}",
                context=context,
                cause=e
            )
        
        # Network errors
        elif "connection" in error_str or "network" in error_str:
            return ThinkMeshError(
                f"Network error: {str(e)}. Check internet connection and API endpoints.",
                severity=ErrorSeverity.MEDIUM,
                context=context,
                cause=e
            )
        
        # Generic fallback
        else:
            return ThinkMeshError(
                f"Unexpected error: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=context,
                cause=e
            )
    
    def log_error(self, error: ThinkMeshError):
        """Log error with appropriate level."""
        if not self.logger:
            return
        
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical ThinkMesh error", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity ThinkMesh error", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity ThinkMesh error", extra=error_dict)
        else:
            self.logger.info("Low severity ThinkMesh error", extra=error_dict)
