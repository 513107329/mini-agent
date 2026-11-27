from abc import abstractmethod
from asyncio import streams
from dataclasses import dataclass, field
import os
from typing import Any, Iterator, Optional
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tools.tool import toolRegistry

# OpenAI 兼容厂商（换 config 即可）
OPENAI_COMPAT_PROVIDERS = ("deepseek", "kimi")
# 原生协议厂商（走 native_adapters 里的 Adapter）
NATIVE_PROVIDERS = ("anthropic", "gemini")
SUPPORTED_PROVIDERS = OPENAI_COMPAT_PROVIDERS + NATIVE_PROVIDERS


@dataclass
class ProviderConfig:
    """
    Provider configuration.
    """

    name: str
    api_key: str
    base_url: str
    timeout: int
    model_name: str


@dataclass
class ToolCallRequest:
    """
    Tool call request for a provider.
    """

    tool_id: str
    tool_name: str
    tool_args: str


@dataclass
class TokenUsage:
    """
    Usage record for a provider.
    """

    prompt_token: int = 0
    completion_token: int = 0
    total_token: int = 0
    model: str = ""
    provider: str = ""

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """
        Add two token usages together.
        """
        return TokenUsage(
            prompt_token=self.prompt_token + other.prompt_token,
            completion_token=self.completion_token + other.completion_token,
            total_token=self.total_token + other.total_token,
        )

    def __str__(self) -> str:
        return f"prompt token: {self.prompt_token}, completion token: {self.completion_token}, total token: {self.total_token}, [model: {self.model}, provider: {self.provider}]"


@dataclass
class UsageTracker:
    """
    Usage tracker for a provider.
    """

    records: list[TokenUsage] = field(default_factory=list)

    def add_record(self, record: TokenUsage):
        """
        Add a usage record to the tracker.
        """
        self.records.append(record)

    @property
    def total(self) -> TokenUsage:
        """
        Total usage of the provider.
        """
        acc = TokenUsage()
        for item in self.records:
            acc += item
        return acc

    def summary(self) -> dict:
        """
        Summary of the usage tracker.
        """
        t = self.total
        return (
            f"calls={len(self.records)}, prompt={t.prompt_token}, "
            f"completion={t.completion_token}, total={t.total_token}"
        )

    def reset(self):
        """
        Reset the usage tracker.
        """
        self.records.clear()


@dataclass
class ChatResponse:
    """
    Chat response from a provider.
    """

    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: Optional[TokenUsage] = None
    provider_name: str = ""

    def has_tool_calls(self) -> bool:
        """
        Check if the response has tool calls.
        """
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def to_assistant_message(self) -> dict[str, Any]:
        """
        Convert the response to an assistant message.
        """
        message = {"role": "assistant", "content": self.content, "tool_calls": []}
        if self.tool_calls and len(self.tool_calls) > 0:
            for tc in self.tool_calls:
                message["tool_calls"].append(
                    {
                        "id": tc.tool_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.tool_args,
                        },
                    }
                )
        return message


def _env(key: str, default: str | None = None) -> str:
    """
    Get an environment variable.
    """
    value = os.getenv(key, default)
    if value is None and value.strip() == "":
        return None
    return value.strip()


def _load_config(name: str) -> ProviderConfig:
    """
    Load a provider configuration.
    """
    api_key = _env(f"{name}_API_KEY")
    base_url = _env(f"{name}_BASE_URL")
    timeout = _env(f"{name}_TIMEOUT", "30")
    model_name = _env(f"{name}_MODEL_NAME")

    missing = [
        label
        for label, value in [
            ("api_key", api_key),
            ("base_url", base_url),
            ("timeout", timeout),
            ("model_name", model_name),
        ]
        if value is None or value.strip() == ""
    ]

    if missing:
        raise ValueError(f"Missing configuration for {', '.join(missing)}")

    return ProviderConfig(
        name=name,
        api_key=api_key,
        base_url=base_url,
        timeout=int(timeout),
        model_name=model_name,
    )


class BaseProvider:
    """
    Base class for all providers.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.usage_tracker = UsageTracker()
        self.iter = 0
        self.max_iter = 10
        self.verbose = False

    @abstractmethod
    def chat(
        self,
        *,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto"
    ) -> ChatResponse:
        """
        Chat with the provider.
        Args:
            prompt: The prompt to chat with.
        Returns:
            The response from the provider.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        *,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto"
    ) -> ChatResponse:
        """
        Chat with the provider.
        Args:
            prompt: The prompt to chat with.
        Returns:
            The response from the provider.
        """
        raise NotImplementedError


class OpenAiCompatibleProvider(BaseProvider):
    """
    OpenAI Compatible provider.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout
        )

    def _extract_usage(self, response: any) -> TokenUsage | None:
        print("usage", response)
        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        return TokenUsage(
            prompt_token=getattr(usage, "prompt_tokens", 0),
            completion_token=getattr(usage, "completion_tokens", 0),
            total_token=getattr(usage, "total_tokens", 0),
            model=self.config.model_name,
            provider=self.config.name,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> str:
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)

        response_message = response.choices[0].message

        tool_calls: list[ToolCallRequest] = []

        if response_message.tool_calls and len(response_message.tool_calls) > 0:
            for tc in response_message.tool_calls:
                tool_calls.append(
                    ToolCallRequest(
                        tool_id=tc.id,
                        tool_name=tc.function.name,
                        tool_args=tc.function.arguments or "{}",
                    )
                )

        usage = self._extract_usage(response)
        self.usage_tracker.add_record(usage)

        response = ChatResponse(
            content=response_message.content,
            tool_calls=tool_calls,
            usage=usage,
            provider_name=self.config.name,
        )
        return response

    def chat_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Iterator[dict]:
        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if tools and len(tools) > 0:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        final_usage: Optional[TokenUsage] = None
        for chunk in response:
            if getattr(chunk, "usage", None) is not None:
                final_usage = self._extract_usage(chunk)
            if not chunk.choices:
                continue
            yield chunk.choices[0].delta

        self.usage_tracker.add_record(final_usage)


def get_provider(name: str) -> BaseProvider:
    """
    Get a provider by name.
    """
    chosen = (name or _env("DEFAULT_PROVIDER", "deepseek") or "deepseek").lower()
    print('chosen', chosen)

    if chosen not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {chosen}")

    if chosen in OPENAI_COMPAT_PROVIDERS:
        return OpenAiCompatibleProvider(_load_config(name))
    # elif chosen in NATIVE_PROVIDERS:
    #     return NativeProvider(_load_config(name))

    raise ValueError(f"Unsupported provider: {chosen}")


def _is_failoverable(exc: BaseException) -> bool:
    """
    Check if an exception is failoverable.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


def get_failover_provider(names: list) -> FailoverProvider:
    providers = [get_provider(name) for name in names]
    return FailoverProvider(providers)


class FailoverProvider(BaseProvider):
    def __init__(self, providers: BaseProvider):
        super().__init__(providers.config)
        self.providers = providers
        self._active_index = 0

    @property
    def active_provider(self):
        return self.providers[self._active_index]

    def _provider_indicies(self):
        start = self._active_index
        return list(range(start, len(self.providers)))

    def record_usage(self, usage: TokenUsage):
        self.usage_tracker.add_record(usage)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ChatResponse:
        errs = []
        last_exc = None
        indicies = self._provider_indicies()

        for pos, index in enumerate(indicies):
            provider = self.providers[index]

            try:
                resposne = provider.chat(messages, tools, temperature, tool_choice)
                self._active_index = index
                self.record_usage(resposne.usage)
                self.config = provider.config
                return resposne
            except Exception as e:
                last_exc = e
                errs.append(f"provider {provider.config.name} error: {e}")
                can_try_next = _is_failoverable(e) and pos < len(indicies) - 1
                if not can_try_next:
                    break

        detail = "|".join(errs) if errs else str(last_exc)
        raise ValueError(f"failover error: {detail}") from last_exc

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Iterator[Any]:
        last_exc = None
        errs = []
        indicies = self._provider_indicies()
        for pos, index in enumerate(indicies):
            provider = self.providers[index]

            try:
                stream = provider.chat_stream(messages, tools, temperature, tool_choice)
                first = next(stream, None)
            except Exception as e:
                last_exc = e
                errs.append(f"provider {provider.config.name} error: {e}")
                can_try_next = _is_failoverable(e) and pos < len(indicies) - 1
                if not can_try_next:
                    break
                continue

            if first is not None:
                yield first
            yield from stream
            self._active_index = index
            self.usage_tracker.records.extend(provider.usage_tracker.records)
            self.config = provider.config
            return

        detail = "|".join(errs) if errs else str(last_exc)
        raise ValueError(f"failover error: {detail}") from last_exc
