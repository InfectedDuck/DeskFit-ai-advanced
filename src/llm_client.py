"""LLM client for EPAM DIAL (OpenAI-compatible API) for DeskFit AI."""

from collections.abc import Generator

from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError


class LLMClient:
    """Client for EPAM DIAL using the OpenAI-compatible API format."""

    def __init__(self, api_base: str, api_key: str, model_name: str) -> None:
        self._client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        self._model = model_name

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request and return the full response text."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError:
            return "Error: Invalid API key. Please check your DIAL_API_KEY in .env"
        except RateLimitError:
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        except APIConnectionError:
            return "Error: Could not connect to EPAM DIAL. Please check your DIAL_API_BASE in .env"
        except Exception as e:
            return f"Error: {e}"

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Send a streaming chat completion request. Yields content chunks."""
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except AuthenticationError:
            yield "Error: Invalid API key. Please check your DIAL_API_KEY in .env"
        except RateLimitError:
            yield "Error: Rate limit exceeded. Please wait a moment and try again."
        except APIConnectionError:
            yield "Error: Could not connect to EPAM DIAL. Please check your DIAL_API_BASE in .env"
        except Exception as e:
            yield f"Error: {e}"

    def health_check(self) -> bool:
        """Test connectivity with a minimal request."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception:
            return False
