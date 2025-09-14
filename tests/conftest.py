"""
Test configuration and fixtures for ThinkMesh tests.
"""
import pytest
import torch
from typing import List, Dict, Any
from thinkmesh.config import ThinkConfig, ModelSpec, StrategySpec
from thinkmesh.adapters.base import GenResult, Thinker


class MockThinker:
    """Mock thinker for testing without actual model inference."""
    
    def __init__(self, responses: List[str] = None, supports_logprobs: bool = True):
        self.responses = responses or ["Test response 1", "Test response 2", "Test response 3"]
        self._supports_logprobs = supports_logprobs
        self.call_count = 0
    
    def supports_logprobs(self) -> bool:
        return self._supports_logprobs
    
    def max_batch_size(self) -> int:
        return 8
    
    async def generate(self, prompts: List[str], *, params: Dict[str, Any]) -> List[GenResult]:
        results = []
        for i, prompt in enumerate(prompts):
            response_idx = (self.call_count + i) % len(self.responses)
            text = self.responses[response_idx]
            
            # Mock logprobs if supported
            logprobs = [-0.1, -0.2, -0.15, -0.3, -0.25] if self._supports_logprobs else None
            tokens = text.split() if self._supports_logprobs else None
            
            results.append(GenResult(
                text=text,
                tokens=tokens,
                token_logprobs=logprobs,
                finish_reason="stop",
                meta={"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())}
            ))
        
        self.call_count += len(prompts)
        return results


@pytest.fixture
def mock_thinker():
    """Provides a mock thinker for testing."""
    return MockThinker()


@pytest.fixture
def mock_thinker_math():
    """Mock thinker with math-specific responses."""
    responses = [
        "Let me solve this step by step. 2 + 3 = 5. Final Answer: 5",
        "I need to calculate 2 + 3. The result is 5. Final Answer: 5",
        "Adding 2 and 3 gives us 5. Final Answer: 5",
        "Step 1: 2 + 3 = 5. Therefore, the answer is 5. Final Answer: 5"
    ]
    return MockThinker(responses)


@pytest.fixture
def basic_config():
    """Basic ThinkMesh configuration for testing."""
    return ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="self_consistency",
            parallel=4,
            max_steps=1
        ),
        budgets={"wall_clock_s": 30, "tokens": 2000}
    )


@pytest.fixture
def deepconf_config():
    """DeepConf strategy configuration."""
    return ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="deepconf",
            parallel=8,
            max_steps=2,
            deepconf={"k": 5, "tau_low": -1.25, "tau_ent": 2.2, "realloc_top_p": 0.4}
        ),
        budgets={"wall_clock_s": 30, "tokens": 4000}
    )


@pytest.fixture
def debate_config():
    """Debate strategy configuration."""
    return ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="debate",
            parallel=4,
            max_steps=2,
            debate={"rounds": 2}
        ),
        budgets={"wall_clock_s": 30, "tokens": 3000}
    )


@pytest.fixture
def tree_config():
    """Tree of Thoughts strategy configuration."""
    return ThinkConfig(
        model=ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=256,
            temperature=0.7,
            extra={"device": "cpu"}
        ),
        strategy=StrategySpec(
            name="tree",
            parallel=6,
            max_steps=2,
            tree={"branches": 3, "depth": 2}
        ),
        budgets={"wall_clock_s": 30, "tokens": 3500}
    )


@pytest.fixture
def cuda_available():
    """Check if CUDA is available for GPU tests."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")


@pytest.fixture
def a100_available():
    """Check if A100 GPU is available."""
    if not torch.cuda.is_available():
        return False
    
    # Check GPU memory (A100 has 80GB)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    return gpu_memory > 70 * 1024**3  # > 70GB indicates A100


@pytest.fixture
def skip_if_no_a100():
    """Skip test if A100 is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping A100 test")
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory < 70 * 1024**3:
        pytest.skip("A100 GPU not available, skipping A100-specific test")


# Test data
MATH_PROBLEMS = [
    {
        "problem": "What is 2 + 3?",
        "answer": "5",
        "difficulty": "easy"
    },
    {
        "problem": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        "answer": "150",
        "difficulty": "medium"
    },
    {
        "problem": "Solve for x: 2x + 5 = 13",
        "answer": "4", 
        "difficulty": "medium"
    }
]

GSM8K_SAMPLE = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents decided to give her twice as much as her parents. How much more money does Betty need?",
        "answer": "15"
    }
]
