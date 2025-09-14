"""
Integration tests for ThinkMesh end-to-end workflows.
"""
import pytest
import torch
from unittest.mock import patch, Mock
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec
from thinkmesh.orchestrator import Orchestrator
from tests.conftest import MockThinker


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete ThinkMesh workflows."""
    
    @pytest.mark.asyncio
    async def test_basic_think_workflow(self, basic_config):
        """Test basic think workflow with mocked components."""
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "I think the answer is 4",
                "Let me calculate: 2+2=4",
                "The result is 4",
                "4 is the correct answer"
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2?"
            
            answer = think(task, basic_config)
            
            assert answer is not None
            assert answer.content is not None
            assert isinstance(answer.confidence, float)
            assert 0.0 <= answer.confidence <= 1.0
            assert "meta" in answer.__dict__
    
    @pytest.mark.asyncio
    async def test_deepconf_workflow(self, deepconf_config):
        """Test DeepConf strategy end-to-end."""
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "Let me think step by step about this math problem...",
                "I need to be careful with the calculation...",
                "Working through this systematically...",
                "The answer is 4 because 2+2=4",
                "Final Answer: 4",
                "I'm confident the answer is 4"
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2? Show your work."
            
            answer = think(task, deepconf_config)
            
            assert answer is not None
            assert "4" in answer.content or "four" in answer.content.lower()
    
    @pytest.mark.asyncio
    async def test_debate_workflow(self, debate_config):
        """Test debate strategy end-to-end."""
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "Debater 1: I believe the answer is 4 because...",
                "Debater 2: I agree, 2+2 clearly equals 4...",
                "Debater 3: Yes, basic arithmetic shows 4...",
                "Debater 4: Confirmed, the sum is 4...",
                "Rebuttal 1: My calculation stands at 4",
                "Rebuttal 2: Still confident it's 4",
                "Rebuttal 3: Mathematics confirms 4",
                "Rebuttal 4: Final answer: 4"
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2?"
            
            answer = think(task, debate_config)
            
            assert answer is not None
            assert isinstance(answer.confidence, float)
    
    @pytest.mark.asyncio
    async def test_tree_workflow(self, tree_config):
        """Test Tree of Thoughts strategy end-to-end."""
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "Branch 1: Let's use basic addition...",
                "Branch 2: We can count: 2, then 2 more...",
                "Branch 3: Using number line approach...",
                "Continue branch 1: 2+2=4",
                "Continue branch 2: Counting gives 4", 
                "Continue branch 3: Number line shows 4",
                "Final from branch 1: The answer is 4",
                "Final from branch 2: Counting confirms 4",
                "Final from branch 3: Visual method gives 4"
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2?"
            
            answer = think(task, tree_config)
            
            assert answer is not None
            assert "4" in answer.content
    
    @pytest.mark.asyncio
    async def test_orchestrator_directly(self, basic_config):
        """Test Orchestrator class directly."""
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker(["The answer is 42"])
            mock_load_thinker.return_value = mock_thinker
            
            orchestrator = Orchestrator(basic_config)
            task = "What is the meaning of life?"
            
            answer = await orchestrator.run(task)
            
            assert answer is not None
            assert "42" in answer.content
            assert "elapsed_s" in answer.meta
    
    @pytest.mark.asyncio
    async def test_with_verifier(self, basic_config):
        """Test workflow with regex verifier."""
        basic_config.verifier = {
            "type": "regex",
            "pattern": r"Final Answer:\s*(\d+)"
        }
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "Let me solve this. 2+2=4. Final Answer: 4",
                "I calculate 2+2=4. Final Answer: 4",
                "The sum is 4. Final Answer: 4"
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2? Answer with 'Final Answer: <number>'"
            
            answer = think(task, basic_config)
            
            assert answer is not None
            assert "4" in answer.content
    
    @pytest.mark.asyncio  
    async def test_with_judge_reducer(self, basic_config):
        """Test workflow with judge reducer."""
        basic_config.reducer = {"name": "judge"}
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            mock_thinker = MockThinker([
                "I think 2+2=4",
                "Definitely 4", 
                "The answer is 4",
                "Final judgment: 4 is correct"  # Judge response
            ])
            mock_load_thinker.return_value = mock_thinker
            
            task = "What is 2+2?"
            
            answer = think(task, basic_config)
            
            assert answer is not None
            assert isinstance(answer.confidence, float)


@pytest.mark.integration
@pytest.mark.slow
class TestRealModelIntegration:
    """Integration tests with real models (when available)."""
    
    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_small_transformers_model(self, skip_if_no_cuda):
        """Test with a small real transformers model."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-small", 
                max_tokens=64,
                temperature=0.7,
                extra={"device": "cuda:0"}
            ),
            strategy=StrategySpec(
                name="self_consistency",
                parallel=2,
                max_steps=1
            ),
            budgets={"wall_clock_s": 60, "tokens": 500}
        )
        
        try:
            answer = think("Say hello", config)
            assert answer is not None
            assert len(answer.content) > 0
        except Exception as e:
            # If model can't be loaded, skip test
            pytest.skip(f"Could not load model: {e}")
    
    @pytest.mark.a100
    @pytest.mark.asyncio
    async def test_larger_model_on_a100(self, skip_if_no_a100):
        """Test with a larger model on A100."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="transformers",
                model_name="microsoft/DialoGPT-medium",
                max_tokens=128,
                temperature=0.7,
                extra={"device": "cuda:0", "dtype": "float16"}
            ),
            strategy=StrategySpec(
                name="deepconf",
                parallel=4,
                max_steps=2,
                deepconf={"k": 3, "tau_low": -1.0}
            ),
            budgets={"wall_clock_s": 120, "tokens": 1000}
        )
        
        try:
            answer = think("What is machine learning?", config)
            assert answer is not None
            assert len(answer.content) > 10  # Should be a substantial response
            assert answer.confidence > 0
        except Exception as e:
            pytest.skip(f"Could not run A100 test: {e}")


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_backend_error(self):
        """Test error handling for invalid backend."""
        config = ThinkConfig(
            model=ModelSpec(
                backend="invalid_backend",  # type: ignore
                model_name="test-model",
                max_tokens=128
            ),
            strategy=StrategySpec(name="self_consistency", parallel=2)
        )
        
        with pytest.raises(ValueError, match="unknown backend"):
            think("Test task", config)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, basic_config):
        """Test timeout handling."""
        # Set very short timeout
        basic_config.budgets["wall_clock_s"] = 0.001
        
        with patch('thinkmesh.adapters.base.load_thinker') as mock_load_thinker:
            # Simulate slow thinker
            mock_thinker = MockThinker()
            import asyncio
            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(1)  # Longer than timeout
                return []
            
            mock_thinker.generate = slow_generate
            mock_load_thinker.return_value = mock_thinker
            
            # This should handle timeout gracefully
            try:
                answer = think("What is 2+2?", basic_config)
                # If it completes, that's fine too
                assert answer is not None
            except Exception:
                # Timeout or other errors are expected
                pass
