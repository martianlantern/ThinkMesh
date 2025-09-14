"""
Unit tests for ThinkMesh reasoning strategies.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from thinkmesh.strategies.deepconf import deepconf_run, make_variants
from thinkmesh.strategies.self_consistency import self_consistency_run
from thinkmesh.strategies.debate import debate_run, seed_prompts
from thinkmesh.strategies.tree import tree_run
from thinkmesh.strategies.graph import graph_run
from thinkmesh.strategies.base import load_strategy
from thinkmesh.adapters.base import GenResult


@pytest.mark.unit
class TestStrategyLoader:
    """Test strategy loading functionality."""
    
    def test_load_strategy_deepconf(self):
        """Test loading deepconf strategy."""
        strategy = load_strategy("deepconf")
        assert strategy == deepconf_run
    
    def test_load_strategy_self_consistency(self):
        """Test loading self_consistency strategy."""
        strategy = load_strategy("self_consistency")
        assert strategy == self_consistency_run
    
    def test_load_strategy_debate(self):
        """Test loading debate strategy."""
        strategy = load_strategy("debate")
        assert strategy == debate_run
    
    def test_load_strategy_tree(self):
        """Test loading tree strategy."""
        strategy = load_strategy("tree")
        assert strategy == tree_run
    
    def test_load_strategy_graph(self):
        """Test loading graph strategy."""
        strategy = load_strategy("graph")
        assert strategy == graph_run
    
    def test_load_strategy_unknown(self):
        """Test loading unknown strategy raises error."""
        with pytest.raises(ValueError, match="unknown strategy"):
            load_strategy("unknown_strategy")


@pytest.mark.unit
class TestDeepConfStrategy:
    """Test DeepConf strategy."""
    
    def test_make_variants(self):
        """Test make_variants function."""
        task = "Solve 2 + 2"
        k = 3
        variants = make_variants(task, k)
        
        assert len(variants) == k
        for i, variant in enumerate(variants):
            assert task in variant
            assert f"Variant #{i+1}" in variant
            assert "Continue reasoning step by step" in variant
    
    @pytest.mark.asyncio
    async def test_deepconf_run_basic(self, mock_thinker, deepconf_config):
        """Test basic deepconf execution."""
        # Mock runner
        mock_runner = Mock()
        step1_results = [
            GenResult("Step 1 reasoning A", ["step", "1"], [-0.1, -0.2], "stop", {}),
            GenResult("Step 1 reasoning B", ["step", "1"], [-0.15, -0.25], "stop", {}),
            GenResult("Step 1 reasoning C", ["step", "1"], [-0.3, -0.4], "stop", {}),
        ]
        step2_results = [
            GenResult("Final answer A", ["final"], [-0.1], "stop", {}),
            GenResult("Final answer B", ["final"], [-0.2], "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [step1_results, step2_results]
        
        task = "What is 2 + 2?"
        result = await deepconf_run(mock_runner, mock_thinker, task, deepconf_config)
        
        assert "candidates" in result
        assert "trace" in result
        assert len(result["candidates"]) == 2  # step2 results
        assert all("scores" in cand for cand in result["candidates"])
        assert all("conf" in cand["scores"] for cand in result["candidates"])
        
        # Verify runner was called twice (step1 and step2)
        assert mock_runner.generate_batched.call_count == 2
    
    @pytest.mark.asyncio
    async def test_deepconf_run_no_logprobs(self, deepconf_config):
        """Test deepconf with thinker that doesn't support logprobs."""
        mock_thinker = Mock()
        mock_thinker.supports_logprobs.return_value = False
        
        mock_runner = Mock()
        step1_results = [
            GenResult("I'm confident in this", None, None, "stop", {}),
            GenResult("This seems right", None, None, "stop", {}),
        ]
        step2_results = [
            GenResult("Final: 4", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [step1_results, step2_results]
        
        task = "What is 2 + 2?"
        result = await deepconf_run(mock_runner, mock_thinker, task, deepconf_config)
        
        assert len(result["candidates"]) == 1
        assert "scores" in result["candidates"][0]
    
    @pytest.mark.asyncio
    async def test_deepconf_run_filtering(self, mock_thinker, deepconf_config):
        """Test deepconf confidence filtering."""
        mock_runner = Mock()
        # Low confidence results that should be filtered
        step1_results = [
            GenResult("Bad reasoning", ["bad"], [-2.0, -3.0], "stop", {}),  # Below tau_low
            GenResult("Good reasoning", ["good"], [-0.1, -0.2], "stop", {}),  # Above tau_low
        ]
        step2_results = [
            GenResult("Final answer", ["final"], [-0.1], "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [step1_results, step2_results]
        
        task = "What is 2 + 2?"
        result = await deepconf_run(mock_runner, mock_thinker, task, deepconf_config)
        
        # Should have filtered out the low-confidence result
        assert len(result["candidates"]) == 1


@pytest.mark.unit
class TestSelfConsistencyStrategy:
    """Test self-consistency strategy."""
    
    @pytest.mark.asyncio
    async def test_self_consistency_run_basic(self, mock_thinker_math, basic_config):
        """Test basic self-consistency execution."""
        mock_runner = Mock()
        results = [
            GenResult("Answer: 5", ["answer"], [-0.1, -0.2], "stop", {}),
            GenResult("The result is 5", ["result"], [-0.15, -0.25], "stop", {}),
            GenResult("I get 5", ["get"], [-0.2, -0.3], "stop", {}),
            GenResult("5 is correct", ["correct"], [-0.1, -0.1], "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock(return_value=results)
        
        task = "What is 2 + 3?"
        result = await self_consistency_run(mock_runner, mock_thinker_math, task, basic_config)
        
        assert "candidates" in result
        assert "trace" in result
        assert len(result["candidates"]) == 4
        
        # Check all candidates have confidence scores
        for cand in result["candidates"]:
            assert "scores" in cand
            assert "conf" in cand["scores"]
            assert "text" in cand
        
        # Verify prompts were generated correctly
        mock_runner.generate_batched.assert_called_once()
        call_args = mock_runner.generate_batched.call_args
        prompts = call_args[0][1]  # Second argument (prompts)
        assert len(prompts) == basic_config.strategy.parallel
        assert all(task in prompt for prompt in prompts)
    
    @pytest.mark.asyncio
    async def test_self_consistency_no_logprobs(self, basic_config):
        """Test self-consistency without logprobs."""
        mock_thinker = Mock()
        mock_thinker.supports_logprobs.return_value = False
        
        mock_runner = Mock()
        results = [
            GenResult("I'm confident: 5", None, None, "stop", {}),
            GenResult("Pretty sure: 5", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock(return_value=results)
        
        task = "What is 2 + 3?"
        result = await self_consistency_run(mock_runner, mock_thinker, task, basic_config)
        
        assert len(result["candidates"]) == 2
        # Should use self-rated confidence when no logprobs
        for cand in result["candidates"]:
            assert "conf" in cand["scores"]


@pytest.mark.unit
class TestDebateStrategy:
    """Test debate strategy."""
    
    def test_seed_prompts(self):
        """Test seed prompt generation."""
        task = "Is the sky blue?"
        k = 3
        prompts = seed_prompts(task, k)
        
        assert len(prompts) == k
        for i, prompt in enumerate(prompts):
            assert f"Debater {i+1}" in prompt
            assert task in prompt
            assert "propose a solution" in prompt
    
    @pytest.mark.asyncio
    async def test_debate_run_single_round(self, mock_thinker, debate_config):
        """Test debate with single round."""
        # Set rounds to 1
        debate_config.strategy.debate["rounds"] = 1
        
        mock_runner = Mock()
        results = [
            GenResult("Debater 1 argument", None, None, "stop", {}),
            GenResult("Debater 2 argument", None, None, "stop", {}),
            GenResult("Debater 3 argument", None, None, "stop", {}),
            GenResult("Debater 4 argument", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock(return_value=results)
        
        task = "Should we use renewable energy?"
        result = await debate_run(mock_runner, mock_thinker, task, debate_config)
        
        assert len(result["candidates"]) == 4
        # Only one round, so only one call to generate_batched
        assert mock_runner.generate_batched.call_count == 1
    
    @pytest.mark.asyncio
    async def test_debate_run_multiple_rounds(self, mock_thinker, debate_config):
        """Test debate with multiple rounds."""
        debate_config.strategy.debate["rounds"] = 2
        
        mock_runner = Mock()
        # First round results
        round1_results = [
            GenResult("Initial argument A", None, None, "stop", {}),
            GenResult("Initial argument B", None, None, "stop", {}),
        ]
        # Second round results (rebuttals)
        round2_results = [
            GenResult("Rebuttal A", None, None, "stop", {}),
            GenResult("Rebuttal B", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [round1_results, round2_results]
        
        task = "Is AI beneficial?"
        result = await debate_run(mock_runner, mock_thinker, task, debate_config)
        
        assert len(result["candidates"]) == 2
        # Two calls: initial arguments + rebuttals
        assert mock_runner.generate_batched.call_count == 2
        
        # Check rebuttal prompts include opponent arguments
        second_call_args = mock_runner.generate_batched.call_args_list[1]
        rebuttal_prompts = second_call_args[0][1]
        assert len(rebuttal_prompts) == 2
        for prompt in rebuttal_prompts:
            assert "rebuttal" in prompt.lower()
            assert "opponents" in prompt.lower()


@pytest.mark.unit  
class TestTreeStrategy:
    """Test Tree of Thoughts strategy."""
    
    @pytest.mark.asyncio
    async def test_tree_run_basic(self, mock_thinker, tree_config):
        """Test basic tree execution."""
        mock_runner = Mock()
        
        # Mock multiple calls for tree expansion
        branch_results_1 = [
            GenResult("Branch 1.1", None, None, "stop", {}),
            GenResult("Branch 1.2", None, None, "stop", {}),
            GenResult("Branch 1.3", None, None, "stop", {}),
        ]
        branch_results_2 = [
            GenResult("Branch 2.1", None, None, "stop", {}),
            GenResult("Branch 2.2", None, None, "stop", {}),
            GenResult("Branch 2.3", None, None, "stop", {}),
        ]
        final_results = [
            GenResult("Final answer A", None, None, "stop", {}),
            GenResult("Final answer B", None, None, "stop", {}),
            GenResult("Final answer C", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [
            branch_results_1, branch_results_2, final_results
        ]
        
        task = "Prove that 2+2=4"
        result = await tree_run(mock_runner, mock_thinker, task, tree_config)
        
        assert len(result["candidates"]) == 3  # final_results length
        assert mock_runner.generate_batched.call_count == 3  # depth=2 + final
        
        for cand in result["candidates"]:
            assert "scores" in cand
            assert "conf" in cand["scores"]
    
    @pytest.mark.asyncio
    async def test_tree_run_single_depth(self, mock_thinker, tree_config):
        """Test tree with depth=1."""
        tree_config.strategy.tree["depth"] = 1
        
        mock_runner = Mock()
        branch_results = [
            GenResult("Branch 1", None, None, "stop", {}),
            GenResult("Branch 2", None, None, "stop", {}),
            GenResult("Branch 3", None, None, "stop", {}),
        ]
        final_results = [
            GenResult("Final A", None, None, "stop", {}),
            GenResult("Final B", None, None, "stop", {}),
            GenResult("Final C", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [branch_results, final_results]
        
        task = "Simple problem"
        result = await tree_run(mock_runner, mock_thinker, task, tree_config)
        
        assert len(result["candidates"]) == 3
        assert mock_runner.generate_batched.call_count == 2  # depth=1 + final


@pytest.mark.unit
class TestGraphStrategy:
    """Test graph-based reasoning strategy."""
    
    @pytest.mark.asyncio
    async def test_graph_run_basic(self, mock_thinker, basic_config):
        """Test basic graph strategy execution."""
        basic_config.strategy.name = "graph"
        
        mock_runner = Mock()
        mid_results = [
            GenResult("Path 1 reasoning", None, None, "stop", {}),
            GenResult("Path 2 reasoning", None, None, "stop", {}),
            GenResult("Path 3 reasoning", None, None, "stop", {}),
            GenResult("Path 4 reasoning", None, None, "stop", {}),
        ]
        final_results = [
            GenResult("Final conclusion A", None, None, "stop", {}),
            GenResult("Final conclusion B", None, None, "stop", {}),
            GenResult("Final conclusion C", None, None, "stop", {}),
            GenResult("Final conclusion D", None, None, "stop", {}),
        ]
        
        mock_runner.generate_batched = AsyncMock()
        mock_runner.generate_batched.side_effect = [mid_results, final_results]
        
        task = "Analyze complex problem"
        result = await graph_run(mock_runner, mock_thinker, task, basic_config)
        
        assert len(result["candidates"]) == 4
        assert mock_runner.generate_batched.call_count == 2  # mid + final
        
        # Check prompts contain path identifiers
        first_call_args = mock_runner.generate_batched.call_args_list[0]
        mid_prompts = first_call_args[0][1]
        for i, prompt in enumerate(mid_prompts):
            assert f"Path {i+1}" in prompt
            assert task in prompt
        
        # Check continuation prompts
        second_call_args = mock_runner.generate_batched.call_args_list[1]
        final_prompts = second_call_args[0][1]
        for prompt in final_prompts:
            assert "Cross-check assumptions" in prompt
