from providers import get_provider, SUPPORTED_PROVIDERS, get_failover_provider
from tools.tool import toolRegistry
import argparse

# provider = get_provider("deepseek")
# response = provider.chat_stream(messages=[{"role": "user", "content": "帮我计算下1+2?"}], tools=toolRegistry.tool_schemas)
# acc = {}
# content = ""
# for delta in response:
#     if delta.content:
#         content += delta.content
#     if delta.tool_calls:
#         for tc in delta.tool_calls:
#             slot = acc.setdefault(tc.index, {"tool_id": "", "tool_name": "", "tool_args": ""})
#             if tc.id:
#                 slot["tool_id"] += tc.id
#             if tc.function.name:
#                 slot["tool_name"] += tc.function.name
#             if tc.function.arguments:
#                 slot["tool_args"] += tc.function.arguments

def runWithFailover(self, failover: str):
    messages = [
        {
            "role": "system",
            "content": "你是助手。需要计算时调用 calculator 工具；需要天气时调用 get_weather。",
        },
        {"role": "user", "content": query},
    ]
    try:
        provider = get_failover_provider(names=failover)
        result = provider.chat()
        return result
    except Exception as e:
            raise ValueError(f"error: {e}")

def runWithProvider(self, provider_name: str):
    messages = [
        {
            "role": "system",
            "content": "你是助手。需要计算时调用 calculator 工具；需要天气时调用 get_weather。",
        },
        {"role": "user", "content": query},
    ]
    try:
        provider = get_provider(name=provider_name)
        result = provider.chat(messages=messages,tools=toolRegistry.tool_schemas, temperature=0.3)
        return result
    except Exception as e:
            raise ValueError(f"error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    argparse.add_document("--provider", help="提供的provider")
    argparse.add_argument("--failover", help="多provider执行失败切换provider")
    argparse.add_document("--query", help="用户的输入")

    args = parser.parse_args()

    if args.failover:
        runWithFailover(args.failover)
        return

    if args.provider and args.provider in SUPPORTED_PROVIDERS:
        runWithProvider(args.provider)
        return