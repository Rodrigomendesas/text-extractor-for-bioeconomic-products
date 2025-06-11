"""OpenAI API client wrapper with rate limiting and error handling."""

import time
import logging
import os
from typing import Dict, Any, List, Optional, Union, BinaryIO
from dataclasses import dataclass
import openai
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response data."""
    content: str
    tokens_used: int
    model: str
    processing_time: float
    finish_reason: str
    usage_details: Dict[str, int]


class OpenAIClient:
    """OpenAI API client with enhanced error handling and rate limiting."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key (uses settings.openai_api_key if None)
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=self.api_key)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        self.request_count = 0

        # Default parameters
        self.default_model = settings.openai_model.value  # Get the enum value
        self.default_max_tokens = settings.openai_max_tokens
        self.default_temperature = settings.openai_temperature

        logger.info(f"OpenAI client initialized with model: {self.default_model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Create a chat completion with the OpenAI API.

        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to settings)
            temperature: Temperature parameter (defaults to settings)
            max_tokens: Maximum tokens (defaults to settings)
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse with the completion result
        """
        start_time = time.time()

        # Use defaults if not provided
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        logger.debug(f"Making chat completion request with model: {model}")

        # Enforce rate limiting
        self._enforce_rate_limit()

        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                **{k: v for k, v in kwargs.items() if k not in ["top_p", "frequency_penalty", "presence_penalty"]}
            }

            # Make the API call
            response = self.client.chat.completions.create(**api_params)

            processing_time = time.time() - start_time
            self.request_count += 1

            # Extract response data
            choice = response.choices[0]
            usage = response.usage

            llm_response = LLMResponse(
                content=choice.message.content,
                tokens_used=usage.total_tokens,
                model=model,
                processing_time=processing_time,
                finish_reason=choice.finish_reason,
                usage_details={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            )

            logger.debug(f"Request completed in {processing_time:.2f}s, used {usage.total_tokens} tokens")
            return llm_response

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            # Exponential backoff
            wait_time = min(60, 2 ** (self.request_count % 5))
            logger.info(f"Waiting {wait_time}s before retry")
            time.sleep(wait_time)
            return self.chat_completion(messages, model, temperature, max_tokens, **kwargs)

        except openai.APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise RuntimeError(f"Failed to connect to OpenAI API: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise RuntimeError(f"Chat completion failed: {e}")

    def simple_completion(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Simple completion with a single user prompt.

        Args:
            prompt: User prompt text
            **kwargs: Additional parameters for chat_completion

        Returns:
            LLMResponse with the completion result
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, **kwargs)

    def system_user_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Completion with system and user prompts.

        Args:
            system_prompt: System prompt to set context
            user_prompt: User prompt with the actual request
            **kwargs: Additional parameters for chat_completion

        Returns:
            LLMResponse with the completion result
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat_completion(messages, **kwargs)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for text (approximate).

        Args:
            text: Text to count tokens for
            model: Model to use for counting (affects tokenization)

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        # This is rough but useful for basic estimation
        return len(text) // 4

    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def file_completion(
        self,
        file_path: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Process a file by sending it directly to the OpenAI API.

        This method sends PDF and DOC/DOCX files directly to the OpenAI API without
        extracting text first, allowing the API to process the file content directly.

        Args:
            file_path: Path to the file to process
            system_prompt: System prompt to set context
            user_prompt: User prompt with the actual request
            model: Model to use (defaults to settings)
            temperature: Temperature parameter (defaults to settings)
            max_tokens: Maximum tokens (defaults to settings)
            **kwargs: Additional parameters for the API call

        Returns:
            LLMResponse with the completion result
        """
        start_time = time.time()

        # Use defaults if not provided
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        logger.debug(f"Processing file and sending to model: {model}")

        # Enforce rate limiting
        self._enforce_rate_limit()

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)

            # For PDF and DOC/DOCX files, send directly to the OpenAI API
            if file_extension in ['.pdf', '.doc', '.docx']:
                logger.info(f"Sending file {file_name} directly to OpenAI API")

                # Send file directly to OpenAI API using the chat completions API
                try:
                    # Upload the file to OpenAI
                    with open(file_path, "rb") as file_data:
                        file_upload = self.client.files.create(
                            file=file_data,
                            purpose="assistants"
                        )

                    # Create messages with the system and user prompts
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    # Prepare API parameters
                    api_params = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": kwargs.get("top_p", 1.0),
                        "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                        "presence_penalty": kwargs.get("presence_penalty", 0.0),
                        **{k: v for k, v in kwargs.items() if k not in ["top_p", "frequency_penalty", "presence_penalty"]}
                    }

                    # Make the API call using the OpenAI client
                    response = self.client.chat.completions.create(**api_params)

                    # Extract response data
                    choice = response.choices[0]
                    usage = response.usage
                    content = choice.message.content

                    # Clean up the uploaded file
                    try:
                        self.client.files.delete(file_upload.id)
                    except Exception as e:
                        logger.warning(f"Error cleaning up file: {e}")

                    # Get token usage from the response
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Create a response object
                    processing_time = time.time() - start_time
                    self.request_count += 1

                    llm_response = LLMResponse(
                        content=content,
                        tokens_used=total_tokens,
                        model=model,
                        processing_time=processing_time,
                        finish_reason=choice.finish_reason,
                        usage_details={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    )

                    logger.debug(f"File request completed in {processing_time:.2f}s, used {total_tokens} tokens")
                    return llm_response


                except Exception as e:
                    logger.warning(f"File processing or API communication failed: {e}")
                    raise RuntimeError(f"File processing failed: {e}")
            else:
                # For other file types, use a fallback approach
                logger.warning(f"Direct file upload not supported for {file_extension} files. Using fallback approach.")
                fallback_prompt = f"{user_prompt}\n\n[Arquivo não suportado para upload direto: {file_path}]"
                fallback_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fallback_prompt}
                ]

                api_params = {
                    "model": model,
                    "messages": fallback_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": kwargs.get("top_p", 1.0),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0),
                    **{k: v for k, v in kwargs.items() if k not in ["top_p", "frequency_penalty", "presence_penalty"]}
                }

                response = self.client.chat.completions.create(**api_params)

                processing_time = time.time() - start_time
                self.request_count += 1

                # Extract response data
                choice = response.choices[0]
                usage = response.usage

                llm_response = LLMResponse(
                    content=choice.message.content,
                    tokens_used=usage.total_tokens,
                    model=model,
                    processing_time=processing_time,
                    finish_reason=choice.finish_reason,
                    usage_details={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    }
                )

                logger.debug(f"File request completed in {processing_time:.2f}s, used {usage.total_tokens} tokens")
                return llm_response

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            # Exponential backoff
            wait_time = min(60, 2 ** (self.request_count % 5))
            logger.info(f"Waiting {wait_time}s before retry")
            time.sleep(wait_time)
            return self.file_completion(file_path, system_prompt, user_prompt, model, temperature, max_tokens, **kwargs)

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

        except openai.APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise RuntimeError(f"Failed to connect to OpenAI API: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")

        except ValueError as e:
            logger.error(f"Value error in file completion: {e}")
            raise RuntimeError(f"File completion value error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in file completion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"File completion failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "request_count": self.request_count,
            "default_model": self.default_model,
            "rate_limit_interval": self.min_request_interval
        }
