"""MedGemma LLM clients."""

from typing import Optional, List, Dict, Any, Generator, Union
import requests
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime


@dataclass
class LLMResponse:
    """LLM response data."""
    
    content: str
    model: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata and timestamp."""
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.tokens_total or (self.tokens_prompt + self.tokens_completion)


class MedGemmaClient:
    """MedGemma LLM client for API interaction."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "medgemma-7b",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize MedGemma client.
        
        Args:
            base_url: Base URL for the LLM server
            model: Model name/identifier
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        logger.info(f"Initialized MedGemmaClient: {model} at {base_url}")
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], stream: bool = False) -> Any:
        """Make HTTP request to LLM server.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            stream: Whether to stream response
            
        Returns:
            Response content or generator for streaming
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    stream=stream
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return None
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> LLMResponse:
        """Generate completion for a prompt.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        
        logger.debug(f"Sending completion request: {self.model}")
        
        try:
            response = self._make_request("/completion", payload)
            data = response.json()
            
            # Parse response based on server format
            if "choices" in data:
                # OpenAI-like format
                content = data["choices"][0]["text"].strip()
                tokens_prompt = data.get("usage", {}).get("prompt_tokens", 0)
                tokens_completion = data.get("usage", {}).get("completion_tokens", 0)
            else:
                # Simple format
                content = data.get("response", data.get("text", "")).strip()
                tokens_prompt = 0
                tokens_completion = 0
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                metadata={"raw_response": data}
            )
        
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise
    
    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion.
        
        Args:
            messages: Chat messages (list of {"role": ..., "content": ...})
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }
        
        logger.debug(f"Sending chat completion request: {self.model}")
        
        try:
            response = self._make_request("/chat/completions", payload)
            data = response.json()
            
            # Parse response
            if "choices" in data:
                content = data["choices"][0]["message"]["content"].strip()
            else:
                content = data.get("response", data.get("text", "")).strip()
            
            tokens_prompt = data.get("usage", {}).get("prompt_tokens", 0)
            tokens_completion = data.get("usage", {}).get("completion_tokens", 0)
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                metadata={"raw_response": data}
            )
        
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def complete_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming completion.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they arrive
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
            **kwargs
        }
        
        logger.debug(f"Sending streaming completion request: {self.model}")
        
        try:
            response = self._make_request("/completion", payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8') if isinstance(line, bytes) else line
                    if line.startswith('data: '):
                        line = line[6:]
                    
                    if line:
                        try:
                            data = json.loads(line)
                            if "choices" in data:
                                chunk = data["choices"][0].get("text", "")
                            else:
                                chunk = data.get("response", "")
                            
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            raise
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming chat completion.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they arrive
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
            **kwargs
        }
        
        logger.debug(f"Sending streaming chat completion request: {self.model}")
        
        try:
            response = self._make_request("/chat/completions", payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8') if isinstance(line, bytes) else line
                    if line.startswith('data: '):
                        line = line[6:]
                    
                    if line and line != "[DONE]":
                        try:
                            data = json.loads(line)
                            if "choices" in data:
                                chunk = data["choices"][0]["delta"].get("content", "")
                            else:
                                chunk = data.get("response", "")
                            
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if LLM server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup connections."""
        try:
            self.session.close()
        except:
            pass


class HuggingFaceMedGemmaClient:
    """MedGemma client using Hugging Face Transformers pipeline."""
    
    def __init__(
        self,
        model_id: str = "google/medgemma-7b-it",
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize Hugging Face MedGemma client.
        
        Args:
            model_id: Model ID on Hugging Face Hub
            device: Device to use ("cpu", "cuda", "auto", etc.)
            torch_dtype: Torch dtype ("auto", "float16", "float32", etc.)
            cache_dir: Cache directory for model
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceMedGemmaClient. "
                "Install it with: pip install transformers torch"
            )
        
        self.model_id = model_id
        self.device = device
        import torch
        
        # Map dtype strings to torch dtypes
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, "auto") if torch_dtype else "auto"
        self.cache_dir = cache_dir
        
        logger.info(f"Loading MedGemma model from HF: {model_id}")
        
        # Load pipeline
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_id,
            device_map=device if device != "cpu" else None,
            torch_dtype=self.torch_dtype if self.torch_dtype != "auto" else "auto",
            model_kwargs={"cache_dir": cache_dir} if cache_dir else {}
        )
        
        logger.info(f"Successfully loaded MedGemma model: {model_id}")
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> LLMResponse:
        """Generate completion for a prompt using Transformers pipeline.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        logger.debug(f"Generating completion for prompt")
        
        try:
            outputs = self.pipeline(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                do_sample=temperature > 0,
                **kwargs
            )
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                if isinstance(outputs[0], dict):
                    content = outputs[0].get("generated_text", "").strip()
                else:
                    content = str(outputs[0]).strip()
            else:
                content = str(outputs).strip()
            
            return LLMResponse(
                content=content,
                model=self.model_id,
                tokens_prompt=len(prompt.split()),  # Approximate
                tokens_completion=len(content.split()),  # Approximate
                metadata={"prompt_length": len(prompt)}
            )
        
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise
    
    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion (converts to text for MedGemma).
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens
            top_p: Top-p sampling parameter
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        # Convert chat format to text prompt
        prompt_parts = []
        
        if system:
            prompt_parts.append(f"System: {system}")
        
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts)
        
        return self.complete(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs
        )
    
    def complete_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming completion (non-streaming for HF by default).
        
        Note: This returns all tokens at once as HF pipeline doesn't support
        streaming by default. For streaming, use TextIteratorStreamer.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens
            top_p: Top-p parameter
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        response = self.complete(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Yield words one by one to simulate streaming
        for word in response.content.split():
            yield word + " "
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming chat completion.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens
            top_p: Top-p parameter
            system: Optional system prompt
            **kwargs: Additional parameters
            
        Yields:
            Text chunks
        """
        response = self.complete_chat(
            messages=messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            system=system,
            **kwargs
        )
        
        # Yield words one by one to simulate streaming
        for word in response.content.split():
            yield word + " "
