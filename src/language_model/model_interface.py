"""
Model interface for interacting with language models (OpenAI, Claude)
"""
import os
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelInterface:
    """Interface for LLM interactions with different providers"""
    
    def __init__(self):
        self.provider = os.getenv("DEFAULT_LLM", "openai")
        self.model = os.getenv("DEFAULT_MODEL", "gpt4o-2024-11-20" if self.provider == "openai" else "claude-3-7-sonnet-20250219")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        
        # Initialize appropriate client
        if self.provider == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        logger.info(f"Initialized ModelInterface with {self.provider} ({self.model})")
        
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from the language model
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            stream: Whether to stream the response
            
        Returns:
            Dictionary with model response and metadata
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"Generating response with {self.provider} model {self.model}")
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                return self._generate_openai(messages, temp, tokens, stream)
            elif self.provider == "anthropic":
                return self._generate_anthropic(messages, temp, tokens, stream)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "content": "I encountered an error while generating a response.",
                "error": str(e),
                "success": False
            }
            
    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            # Handle streaming vs. non-streaming responses
            if stream:
                # For streaming, we'd typically yield chunks
                # For simplicity here, we'll return a placeholder
                return {
                    "content": "<streaming response>",
                    "success": True,
                    "streaming": True
                }
            else:
                # Process standard response
                content = response.choices[0].message.content
                
                return {
                    "content": content,
                    "model": self.model,
                    "usage": {
                        "total_tokens": response.usage.total_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
            
    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        try:
            # Convert messages to Anthropic format if needed
            anthropic_messages = messages
            
            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            # Handle streaming vs. non-streaming responses
            if stream:
                # For streaming, we'd typically yield chunks
                # For simplicity here, we'll return a placeholder
                return {
                    "content": "<streaming response>",
                    "success": True,
                    "streaming": True
                }
            else:
                # Process standard response
                content = response.content[0].text
                
                return {
                    "content": content,
                    "model": self.model,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    },
                    "stop_reason": response.stop_reason,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
            
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            try:
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
                response = self.client.embeddings.create(
                    model=embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {str(e)}")
                raise
        else:
            # Default to OpenAI for embeddings even when using Claude
            # As Anthropic doesn't have an embedding API yet
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            try:
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
                response = openai.embeddings.create(
                    model=embedding_model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {str(e)}")
                raise