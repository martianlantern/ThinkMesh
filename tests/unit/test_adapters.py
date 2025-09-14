"""
Unit tests for ThinkMesh adapters.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from thinkmesh.adapters.base import load_thinker, GenResult
from thinkmesh.adapters.transformers_local import TransformersLocal
from thinkmesh.adapters.vllm import VLLMAdapter
from thinkmesh.config import ModelSpec


@pytest.mark.unit
class TestBaseAdapter:
    """Test base adapter functionality."""
    
    @pytest.mark.asyncio
    async def test_load_thinker_transformers(self):
        """Test loading transformers adapter."""
        model_spec = ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=128,
            extra={"device": "cpu"}
        )
        
        with patch('thinkmesh.adapters.transformers_local.TransformersLocal.create') as mock_create:
            mock_thinker = Mock()
            mock_create.return_value = mock_thinker
            
            result = await load_thinker(model_spec)
            assert result == mock_thinker
            mock_create.assert_called_once_with(model_spec)
    
    @pytest.mark.asyncio
    async def test_load_thinker_vllm(self):
        """Test loading vLLM adapter."""
        model_spec = ModelSpec(
            backend="vllm",
            model_name="test-model",
            max_tokens=128
        )
        
        result = await load_thinker(model_spec)
        assert isinstance(result, VLLMAdapter)
        assert result.model == model_spec
    
    @pytest.mark.asyncio
    async def test_load_thinker_unknown_backend(self):
        """Test loading unknown backend raises error."""
        model_spec = ModelSpec(
            backend="unknown",  # type: ignore
            model_name="test-model",
            max_tokens=128
        )
        
        with pytest.raises(ValueError, match="unknown backend"):
            await load_thinker(model_spec)


@pytest.mark.unit
class TestTransformersLocalAdapter:
    """Test TransformersLocal adapter."""
    
    def create_mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["test", "tokens"]
        mock_tokenizer.decode.return_value = "test response"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        
        # Mock generate output
        mock_output = Mock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])  # input + generated
        mock_output.scores = [torch.randn(1, 1000), torch.randn(1, 1000)]  # 2 generated tokens
        mock_model.generate.return_value = mock_output
        
        return mock_model, mock_tokenizer
    
    @pytest.mark.asyncio
    @patch('thinkmesh.adapters.transformers_local.AutoModelForCausalLM')
    @patch('thinkmesh.adapters.transformers_local.AutoTokenizer')
    @patch('thinkmesh.adapters.transformers_local.torch')
    async def test_create_transformers_adapter(self, mock_torch, mock_tokenizer_class, mock_model_class):
        """Test creating TransformersLocal adapter."""
        mock_torch.float16 = torch.float16
        mock_model, mock_tokenizer = self.create_mock_model_and_tokenizer()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model_spec = ModelSpec(
            backend="transformers",
            model_name="test-model",
            max_tokens=128,
            extra={"device": "cuda:0", "dtype": "float16"}
        )
        
        adapter = await TransformersLocal.create(model_spec)
        
        assert isinstance(adapter, TransformersLocal)
        assert adapter.model == model_spec
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transformers_supports_logprobs(self):
        """Test that TransformersLocal supports logprobs."""
        model_spec = ModelSpec(backend="transformers", model_name="test", max_tokens=128)
        mock_model, mock_tokenizer = self.create_mock_model_and_tokenizer()
        
        adapter = TransformersLocal(model_spec, (mock_model, mock_tokenizer, "cpu"))
        
        assert adapter.supports_logprobs() is True
    
    @pytest.mark.asyncio
    async def test_transformers_generate(self):
        """Test TransformersLocal generate method."""
        model_spec = ModelSpec(
            backend="transformers", 
            model_name="test", 
            max_tokens=128,
            temperature=0.7
        )
        mock_model, mock_tokenizer = self.create_mock_model_and_tokenizer()
        
        adapter = TransformersLocal(model_spec, (mock_model, mock_tokenizer, "cpu"))
        
        prompts = ["What is 2+2?"]
        params = {"max_tokens": 50}
        
        results = await adapter.generate(prompts, params=params)
        
        assert len(results) == 1
        assert isinstance(results[0], GenResult)
        assert results[0].text == "test response"
        assert results[0].tokens == ["test", "tokens"]
        assert results[0].token_logprobs is not None
        assert len(results[0].token_logprobs) == 2  # 2 generated tokens
    
    def test_transformers_max_batch_size(self):
        """Test TransformersLocal max_batch_size."""
        model_spec = ModelSpec(
            backend="transformers",
            model_name="test",
            max_tokens=128,
            extra={"batch_size": 8}
        )
        mock_model, mock_tokenizer = self.create_mock_model_and_tokenizer()
        
        adapter = TransformersLocal(model_spec, (mock_model, mock_tokenizer, "cpu"))
        
        assert adapter.max_batch_size() == 8


@pytest.mark.unit
class TestVLLMAdapter:
    """Test vLLM adapter."""
    
    def test_vllm_adapter_init(self):
        """Test VLLMAdapter initialization."""
        model_spec = ModelSpec(
            backend="vllm",
            model_name="test-model",
            max_tokens=128
        )
        
        adapter = VLLMAdapter(model_spec)
        
        assert adapter.model == model_spec
    
    def test_vllm_supports_logprobs(self):
        """Test that vLLM adapter doesn't support logprobs by default."""
        model_spec = ModelSpec(backend="vllm", model_name="test", max_tokens=128)
        adapter = VLLMAdapter(model_spec)
        
        assert adapter.supports_logprobs() is False
    
    def test_vllm_max_batch_size(self):
        """Test vLLM max_batch_size."""
        model_spec = ModelSpec(
            backend="vllm",
            model_name="test",
            max_tokens=128,
            extra={"batch_size": 16}
        )
        adapter = VLLMAdapter(model_spec)
        
        assert adapter.max_batch_size() == 16
    
    @pytest.mark.asyncio
    @patch('thinkmesh.adapters.vllm.AsyncOpenAI')
    async def test_vllm_generate(self, mock_openai):
        """Test vLLM generate method."""
        # Mock OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].text = "Generated response"
        
        mock_client = Mock()
        mock_client.completions = Mock()
        mock_client.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        model_spec = ModelSpec(
            backend="vllm",
            model_name="test-model",
            max_tokens=128,
            temperature=0.7,
            extra={"base_url": "http://localhost:8000/v1", "api_key": "test-key"}
        )
        adapter = VLLMAdapter(model_spec)
        
        prompts = ["What is 2+2?"]
        params = {"max_tokens": 50}
        
        results = await adapter.generate(prompts, params=params)
        
        assert len(results) == 1
        assert isinstance(results[0], GenResult)
        assert results[0].text == "Generated response"
        assert results[0].tokens is None  # vLLM doesn't return tokens
        assert results[0].token_logprobs is None  # vLLM doesn't return logprobs
        assert results[0].finish_reason == "stop"
        
        mock_client.completions.create.assert_called_once()


@pytest.mark.gpu
@pytest.mark.slow
class TestTransformersLocalGPU:
    """GPU tests for TransformersLocal adapter."""
    
    @pytest.mark.asyncio
    async def test_transformers_gpu_basic(self, skip_if_no_cuda):
        """Test basic GPU functionality."""
        # This would require a real model, so we'll mock it for now
        # In a real scenario, you'd use a small model like gpt2
        pass
    
    @pytest.mark.asyncio  
    async def test_transformers_memory_usage(self, skip_if_no_a100):
        """Test memory usage on A100."""
        # This would test actual memory consumption with real models
        pass
