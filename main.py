import argparse
import json
from providers import (
    ChatResponse,
    ToolCallRequest,
    get_provider,
    get_failover_provider,
)
from tools.tool import toolRegistry


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
        self.last_tool_call_fp: str = ""   # 上一次调用的工具指纹
        self.consecutive_repeats: int = 0  # 连续重复调用次数

        if failover is not None:
            self.provider = get_failover_provider(names=failover)
        elif provider is not None:
            self.provider = get_provider(name=provider)
        else:
            raise ValueError("请输入正确的provider或failover")

    def _reset_loop_state(self) -> None:
        """重置单次任务的运行状态（迭代轮次与死循环检测指纹）"""
        self.iter = 0
        self.last_tool_call_fp = ""
        self.consecutive_repeats = 0

    def _check_tool_loop(self, tool_name: str, tool_args: dict) -> tuple[bool, str]:
        """
        死循环检测
        返回: (是否熔断, 警告提示信息)
        """
        fp = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"

        if fp == self.last_tool_call_fp:
            self.consecutive_repeats += 1
        else:
            self.last_tool_call_fp = fp
            self.consecutive_repeats = 1

        # 连续 3 次 -> 触发硬熔断
        if self.consecutive_repeats >= 3:
            return True, "连续三次完全相同的工具调用，触发硬熔断"

        # 连续 2 次 -> 注入给模型的警告 Observation
        if self.consecutive_repeats == 2:
            warning = "\n[系统警告]: 检测到你连续两次使用了完全相同的参数调用此工具，且结果未变化。请更换思路或调整参数，切勿重复调用！"
            return False, warning

        return False, ""

    def _execute_tool_calls(self, tool_calls: list[ToolCallRequest]) -> str | None:
        """统一执行流式/非流式工具调用，包含死循环检测与悬空 tool_call 兜底"""
        for idx, tc in enumerate(tool_calls):
            tool_id = tc.tool_id
            tool_name = tc.tool_name

            try:
                tool_args = json.loads(tc.tool_args) if tc.tool_args else {}
            except json.JSONDecodeError as e:
                tool_result = f"错误：工具参数 JSON 解析失败 ({e})"
                self.messages.append(
                    {"role": "tool", "content": tool_result, "tool_call_id": tool_id}
                )
                if self.verbose:
                    print(f"Observation: {tool_result}")
                continue

            is_circuit_break, warning_msg = self._check_tool_loop(tool_name, tool_args)
            if is_circuit_break:
                print(f"\n[Circuit Breaker]: {warning_msg}")
                # 为当前及后续所有未执行的 tool_call 补齐占位 tool 消息，防止悬空 tool_call_id 导致下次请求报 400
                for remaining_tc in tool_calls[idx:]:
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": f"已跳过执行：{warning_msg}",
                            "tool_call_id": remaining_tc.tool_id,
                        }
                    )
                break_msg = f"任务因死循环熔断终止：{warning_msg}"
                self.messages.append({"role": "assistant", "content": break_msg})
                self._reset_loop_state()
                return break_msg

            tool = toolRegistry.get_tool(tool_name)
            if tool is None:
                tool_result = f"错误：未注册的工具 '{tool_name}'"
            else:
                try:
                    tool_result = str(tool.run(tool_args))
                except Exception as e:
                    tool_result = f"工具执行异常：{e}"

            if warning_msg:
                print(f"\n[Warning]: {warning_msg.strip()}")
                tool_result = str(tool_result) + warning_msg

            self.messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_id,
                }
            )
            if self.verbose:
                print(f"Observation: {tool_result}")

        return None

    def run(self, input_str: str) -> str:
        """
        执行 react loop
        """
        self._reset_loop_state()
        self.messages.append({"role": "user", "content": input_str})

        while True:
            if self.iter >= self.max_iter:
                max_iter_msg = f"已超出最大迭代轮数 ({self.max_iter})，任务强制终止。"
                self.messages.append({"role": "assistant", "content": max_iter_msg})
                self._reset_loop_state()
                return max_iter_msg
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
                        self._reset_loop_state()
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

                    break_msg = self._execute_tool_calls(result.tool_calls)
                    if break_msg is not None:
                        return break_msg
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

                    if content and not acc:
                        print()

                    if not acc:
                        self.messages.append({"role": "assistant", "content": content})
                        self._reset_loop_state()
                        if self.verbose:
                            print("[Task Complete] 最终回答已生成。")
                            print(
                                f"[Usage Total] {self.provider.usage_tracker.summary()}"
                            )
                        return content

                    tool_calls = [
                        ToolCallRequest(
                            tool_id=tc.get("tool_id"),
                            tool_name=tc.get("tool_name"),
                            tool_args=tc.get("tool_args"),
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
                        tool_names = [tc.get("tool_name") for tc in acc.values()]
                        print(
                            f"[Action] 模型决定调用工具 ({len(acc)} 个): {tool_names}"
                        )

                    break_msg = self._execute_tool_calls(tool_calls)
                    if break_msg is not None:
                        return break_msg

            except Exception as e:
                self._reset_loop_state()
                return f"执行出错：{e}"


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

    react_loop = ReactLoop(
        failover=args.failover,
        provider=args.provider,
        max_iter=args.max_iter,
        verbose=args.verbose,
        no_stream=args.no_stream,
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
                react_loop._reset_loop_state()
                print("重置成功")
                continue

            if cmd == "/provider":
                if not arg:
                    print("请输入正确的provider")
                    continue
                try:
                    switch_provider = get_provider(name=arg)
                    react_loop.provider = switch_provider
                    print(f"切换成功: {switch_provider.config.name}")
                except Exception as e:
                    print(f"切换失败: {e}")
                continue

            if cmd == "/usage":
                print(f"Usage: {react_loop.provider.usage_tracker.summary()}")
                continue

            print("Unknown command")
            continue

        result = react_loop.run(query)
        print(f"最终回答: {result}")
