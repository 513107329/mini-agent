from providers import get_provider
from tools.tool import toolRegistry

provider = get_provider("deepseek")
print("toolRegistry.tool_schemas", toolRegistry.tool_schemas)
response = provider.chat_stream(messages=[{"role": "user", "content": "帮我计算下1+2?"}], tools=toolRegistry.tool_schemas)
acc = {}
content = ""
for chunk in response:
    print("chunk", chunk)
    # if chunk.get("delta", {}).get("content"):
    #     print("chunk.get('content')", chunk.get("delta", {}).get("content"))
    #     content += chunk.get("delta", {}).get("content")
    # if chunk.get("tool_calls"):
    #     for tc in chunk.get("tool_calls"):
    #         slot = acc.setdefault(tc.index, {"tool_id": "", "tool_name": "", "tool_args": ""})
    #         if tc.get("id"):
    #             slot["tool_id"] = tc.get("id")
    #         if tc.get("function").get("name"):
    #             slot["tool_name"] = tc.get("function").get("name")
    #         if tc.get("function").get("arguments"):
    #             slot["tool_args"] = tc.get("function").get("arguments")

print("acc", acc)
print(provider.usage_tracker.total)
print("content", content)