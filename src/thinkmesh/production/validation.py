"""
Configuration validation for ThinkMesh.
"""
import torch
from typing import List, Dict, Any, Optional
from ..config import ThinkConfig, ModelSpec, StrategySpec
from .error_handling import ConfigurationError, ValidationError


class ConfigValidator:
    """Validates ThinkMesh configurations for production use."""
    
    @staticmethod
    def validate_model_spec(model: ModelSpec) -> List[str]:
        """Validate model specification."""
        warnings = []
        
        # Check backend compatibility
        if model.backend == "transformers":
            try:
                import transformers
            except ImportError:
                raise ConfigurationError(
                    "transformers backend requires 'transformers' package. Install with: pip install transformers",
                    field="model.backend"
                )
            
            # Check device compatibility
            device = model.extra.get("device", "cpu")
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise ConfigurationError(
                        f"CUDA device '{device}' specified but CUDA is not available",
                        field="model.extra.device"
                    )
                
                # Extract device index
                if ":" in device:
                    try:
                        device_idx = int(device.split(":")[1])
                        if device_idx >= torch.cuda.device_count():
                            raise ConfigurationError(
                                f"CUDA device {device_idx} not available. Available devices: 0-{torch.cuda.device_count()-1}",
                                field="model.extra.device"
                            )
                    except ValueError:
                        raise ConfigurationError(
                            f"Invalid CUDA device format: '{device}'. Use 'cuda:0', 'cuda:1', etc.",
                            field="model.extra.device"
                        )
        
        elif model.backend == "vllm":
            try:
                import openai
            except ImportError:
                raise ConfigurationError(
                    "vLLM backend requires 'openai' package. Install with: pip install openai",
                    field="model.backend"
                )
        
        elif model.backend == "openai":
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                warnings.append("OpenAI API key not found in environment variables")
        
        # Check token limits
        if model.max_tokens <= 0:
            raise ConfigurationError(
                "max_tokens must be positive",
                field="model.max_tokens"
            )
        
        if model.max_tokens > 8192:
            warnings.append(f"max_tokens={model.max_tokens} is very large and may be slow")
        
        # Check temperature
        if not 0.0 <= model.temperature <= 2.0:
            warnings.append(f"temperature={model.temperature} is outside typical range [0.0, 2.0]")
        
        # Check batch size
        batch_size = model.extra.get("batch_size", 4)
        if batch_size <= 0:
            raise ConfigurationError(
                "batch_size must be positive",
                field="model.extra.batch_size"
            )
        
        if batch_size > 32:
            warnings.append(f"batch_size={batch_size} is very large and may cause memory issues")
        
        return warnings
    
    @staticmethod
    def validate_strategy_spec(strategy: StrategySpec) -> List[str]:
        """Validate strategy specification."""
        warnings = []
        
        # Check parallel count
        if strategy.parallel <= 0:
            raise ConfigurationError(
                "parallel count must be positive",
                field="strategy.parallel"
            )
        
        if strategy.parallel > 64:
            warnings.append(f"parallel={strategy.parallel} is very large and may be inefficient")
        
        # Check max steps
        if strategy.max_steps <= 0:
            raise ConfigurationError(
                "max_steps must be positive",
                field="strategy.max_steps"
            )
        
        if strategy.max_steps > 10:
            warnings.append(f"max_steps={strategy.max_steps} is large and may be slow")
        
        # Strategy-specific validation
        if strategy.name == "deepconf":
            deepconf_params = strategy.deepconf
            
            k = deepconf_params.get("k", 5)
            if k <= 0:
                raise ConfigurationError(
                    "deepconf.k must be positive",
                    field="strategy.deepconf.k"
                )
            
            tau_low = deepconf_params.get("tau_low", -1.25)
            if tau_low > 0:
                warnings.append(f"deepconf.tau_low={tau_low} is positive; typical values are negative")
            
            tau_ent = deepconf_params.get("tau_ent", 2.2)
            if tau_ent <= 0:
                warnings.append(f"deepconf.tau_ent={tau_ent} is non-positive; typical values are positive")
            
            realloc_top_p = deepconf_params.get("realloc_top_p", 0.4)
            if not 0.0 < realloc_top_p <= 1.0:
                raise ConfigurationError(
                    "deepconf.realloc_top_p must be in (0.0, 1.0]",
                    field="strategy.deepconf.realloc_top_p"
                )
        
        elif strategy.name == "debate":
            rounds = strategy.debate.get("rounds", 2)
            if rounds <= 0:
                raise ConfigurationError(
                    "debate.rounds must be positive",
                    field="strategy.debate.rounds"
                )
            
            if rounds > 10:
                warnings.append(f"debate.rounds={rounds} is large and may be very slow")
        
        elif strategy.name == "tree":
            branches = strategy.tree.get("branches", 3)
            depth = strategy.tree.get("depth", 2)
            
            if branches <= 0:
                raise ConfigurationError(
                    "tree.branches must be positive",
                    field="strategy.tree.branches"
                )
            
            if depth <= 0:
                raise ConfigurationError(
                    "tree.depth must be positive", 
                    field="strategy.tree.depth"
                )
            
            # Check for exponential explosion
            total_nodes = sum(branches ** d for d in range(depth + 1))
            if total_nodes > 1000:
                warnings.append(f"tree configuration creates {total_nodes} nodes; may be very slow")
        
        return warnings
    
    @staticmethod
    def validate_budgets(budgets: Dict[str, Any]) -> List[str]:
        """Validate budget constraints."""
        warnings = []
        
        # Check wall clock budget
        wall_clock = budgets.get("wall_clock_s")
        if wall_clock and wall_clock <= 0:
            raise ConfigurationError(
                "wall_clock_s budget must be positive",
                field="budgets.wall_clock_s"
            )
        
        if wall_clock and wall_clock > 3600:  # 1 hour
            warnings.append(f"wall_clock_s={wall_clock} is very large (>1 hour)")
        
        # Check token budget
        tokens = budgets.get("tokens")
        if tokens and tokens <= 0:
            raise ConfigurationError(
                "tokens budget must be positive",
                field="budgets.tokens"
            )
        
        if tokens and tokens > 100000:
            warnings.append(f"tokens={tokens} is very large and may be expensive")
        
        return warnings
    
    @staticmethod
    def validate_system_compatibility(config: ThinkConfig) -> List[str]:
        """Validate system compatibility."""
        warnings = []
        
        # Check GPU memory requirements
        if config.model.extra.get("device", "cpu").startswith("cuda"):
            if torch.cuda.is_available():
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Estimate memory requirements
                estimated_memory = 0
                
                # Base model memory (rough estimates)
                if "large" in config.model.model_name.lower():
                    estimated_memory += 8  # ~8GB for large models
                elif "medium" in config.model.model_name.lower():
                    estimated_memory += 4  # ~4GB for medium models
                else:
                    estimated_memory += 2  # ~2GB for small models
                
                # Batch processing memory
                batch_size = config.model.extra.get("batch_size", 4)
                parallel = config.strategy.parallel
                estimated_memory += (batch_size * parallel * config.model.max_tokens) / 100000  # Rough estimate
                
                if estimated_memory > total_memory_gb * 0.8:  # 80% threshold
                    warnings.append(
                        f"Estimated memory usage ({estimated_memory:.1f}GB) may exceed "
                        f"available GPU memory ({total_memory_gb:.1f}GB)"
                    )
        
        # Check CPU vs GPU backend compatibility
        device = config.model.extra.get("device", "cpu")
        if device == "cpu" and config.strategy.parallel > 8:
            warnings.append("High parallelism on CPU may be inefficient; consider GPU or lower parallel count")
        
        return warnings


def validate_config(config: ThinkConfig) -> Dict[str, List[str]]:
    """
    Comprehensive configuration validation.
    
    Returns:
        Dictionary with validation results:
        - "errors": List of configuration errors (will raise exceptions)
        - "warnings": List of warnings (suboptimal but valid configurations)
    """
    validator = ConfigValidator()
    all_warnings = []
    
    try:
        # Validate model spec
        model_warnings = validator.validate_model_spec(config.model)
        all_warnings.extend(model_warnings)
        
        # Validate strategy spec
        strategy_warnings = validator.validate_strategy_spec(config.strategy)
        all_warnings.extend(strategy_warnings)
        
        # Validate budgets
        budget_warnings = validator.validate_budgets(config.budgets)
        all_warnings.extend(budget_warnings)
        
        # Validate system compatibility
        system_warnings = validator.validate_system_compatibility(config)
        all_warnings.extend(system_warnings)
        
        return {
            "errors": [],
            "warnings": all_warnings
        }
        
    except (ConfigurationError, ValidationError) as e:
        return {
            "errors": [str(e)],
            "warnings": all_warnings
        }
