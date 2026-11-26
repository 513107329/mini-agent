import json
from providers import (
    ChatResponse,
    ToolCallRequest,
    get_provider,
    SUPPORTED_PROVIDERS,
    get_failover_provider,
)
from tools.tool import toolRegistry
import argparse


class ReactLoop:
    def __init__(
        self,
        failover: str | None = None,
        provider: str | None = None,
        max_iter: int = 3,
        verbose: bool = False,
        no_stream: bool = False,
    ):
        self.max_iter = max_iter
        self.verbose = verbose
        self.no_stream = no_stream
        self.messages = [
            {
                "role": "system",
                "content": "你是助手。需要计算时调用 calculator 工具；需要天气时调用 get_weather。",
            }
        ]
        self.iter = 0
        self.verbose = verbose
        self.no_stream = no_stream

        if failover is not None:
            self.provider = get_failover_provider(names=failover)
        elif provider is not None:
            self.provider = get_provider(name=provider)
        else:
            raise ValueError("请输入正确的provider或failover")

    def run(self, input_str: str) -> None:
        """
        执行react loop
        """
        print('no_stream', self.no_stream)
        self.messages.append({"role": "user", "content": input_str})
        while True:
            if self.iter >= self.max_iter:
                self.iter = 0
                raise ValueError("超出了最大迭代轮数")
            try:
                self.iter += 1
                if self.verbose:
                    print(f"第{self.iter}轮迭代: Thought: {self.messages}")
                if self.no_stream:
                    result = self.provider.chat(
                        messages=self.messages,
                        tools=toolRegistry.tool_schemas,
                        temperature=0.3,
                    )
                    if not result.tool_calls:
                        self.messages.append(
                            {"role": "assistant", "content": result.content}
                        )

                        self.iter = 0
                        if self.verbose:
                            print("[Task Complete] 最终回答已生成。")
                            print(
                                f"[Usage Total] {self.provider.usage_tracker.summary()}"
                            )
                        return result.content

                    self.messages.append(result.to_assistant_message())

                    if self.verbose:
                        tool_names = [tc.tool_name for tc in result.tool_calls]
                        print(
                            f"[Action] 模型决定调用工具 ({len(result.tool_calls)} 个): {tool_names}"
                        )

                    for tc in result.tool_calls:
                        tool_id = tc.tool_id
                        tool_name = tc.tool_name
                        tool_args = json.loads(tc.tool_args)
                        tool_result = toolRegistry.get_tool(tool_name).run(tool_args)
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": tool_result,
                                "tool_call_id": tool_id,
                            }
                        )
                        if self.verbose:
                            print("Observation:", tool_result)
                else:
                    response = self.provider.chat_stream(
                        messages=self.messages, tools=toolRegistry.tool_schemas
                    )
                    acc = {}
                    content = ""
                    for delta in response:
                        if delta.content:
                            print(delta.content, end="", flush=True)
                            content += delta.content
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                slot = acc.setdefault(
                                    tc.index,
                                    {"tool_id": "", "tool_name": "", "tool_args": ""},
                                )
                                if tc.id:
                                    slot["tool_id"] += tc.id
                                if tc.function.name:
                                    slot["tool_name"] += tc.function.name
                                if tc.function.arguments:
                                    slot["tool_args"] += tc.function.arguments

                    if not acc:
                        self.messages.append({"role": "assistant", "content": content})
                        if self.verbose:
                            print("[Task Complete] 最终回答已生成。")
                            print(
                                f"[Usage Total] {self.provider.usage_tracker.summary()}"
                            )
                        
                        self.iter = 0
                        return content

                    tool_calls = [
                        ToolCallRequest(
                            tool_id=tc.get('tool_id'), tool_name=tc.get('tool_name'), tool_args=tc.get('tool_args')
                        )
                        for tc in acc.values()
                    ]

                    result = ChatResponse(
                        tool_calls=tool_calls,
                        content=content,
                        provider_name=self.provider.config.name,
                    )
                    self.messages.append(result.to_assistant_message())

                    if self.verbose:
                        tool_names = [tc.get('tool_name') for tc in acc.values()]
                        print(
                            f"[Action] 模型决定调用工具 ({len(acc)} 个): {tool_names}"
                        )

                    for tc in tool_calls:
                        tool_id = tc.tool_id
                        tool_name = tc.tool_name
                        tool_args = json.loads(tc.tool_args)
                        tool_result = toolRegistry.get_tool(tool_name).run(tool_args)
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": tool_result,
                                "tool_call_id": tool_id,
                            }
                        )

                        print(f"Observation: {tool_result}")

            except Exception as e:
                raise ValueError(f"error: {e}")


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("请传入 true 或 false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sum the integers at the command line")
    parser.add_argument("--provider", help="提供的provider", default="deepseek")
    parser.add_argument(
        "--failover", help="多provider执行失败切换provider", default=None
    )
    parser.add_argument("--max-iter", help="最大执行轮数", type=int, default=3)
    parser.add_argument(
        "--no-stream",
        help="关闭流式输出",
        type=str_to_bool,
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="是否打印详细信息",
        type=str_to_bool,
        default=False,
    )
    parser.add_argument("--query", help="用户的输入", default="")

    args = parser.parse_args()

    max_iter = args.max_iter
    no_stream = args.no_stream
    verbose = args.verbose

    react_loop = ReactLoop(
        failover=args.failover,
        provider=args.provider,
        max_iter=max_iter,
        verbose=verbose,
        no_stream=no_stream,
    )

    while True:
        try:
            query = input("请输入你的问题: ")
        except (KeyboardInterrupt, EOFError):
            print("\n退出")
            break
        query = query.strip()
        if not query:
            continue
        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/exit":
                break

            if cmd == "/reset":
                react_loop.messages = react_loop.messages[:1]
                react_loop.iter = 0
                print("重置成功")
                continue

            if cmd == "/provider":
                switch_provider = None
                if not arg:
                    print("请输入正确的provider")
                    continue
                try:
                    switch_provider = get_provider(name=arg)
                    print(f"切换成功: {switch_provider.config.name}")
                except Exception as e:
                    print(f"切换失败: {e}")
                react_loop.provider = switch_provider
                continue

            if cmd == "/usage":
                print(f"Usage: {react_loop.provider.usage_tracker.summary()}")
                continue

            print("Unknown command")
            continue

        result = react_loop.run(query)
        print(f"最终回答: {result}")
