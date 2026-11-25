from dataclasses import dataclass
from typing import Any, Callable

from tools.calculator import CALCULATOR_TOOL_SCHEMA, calculator
from tools.get_weather import GET_WEATHER_TOOL_SCHEMA, get_weather


@dataclass
class Tool:
    """
    A tool is a function that can be called by the LLM.
    """

    name: str
    func: Callable
    schema: dict[str, dict[str, Any]]


class ToolRegistry:
    def __init__(self):
        self.tool_list: list[Tool] = []

    @property
    def tools(self) -> list[Callable]:
        return [tool.func for tool in self.tool_list]

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tool_list]

    def register_tool(self, tool: Tool):
        self.tool_list.append(tool)

    def get_tool(self, name: str) -> Tool | None:
        for tool in self.tool_list:
            if tool.name == name:
                return tool


toolRegistry = ToolRegistry()
toolRegistry.register_tool(
    Tool(name="calculator", func=calculator, schema=CALCULATOR_TOOL_SCHEMA)
)
toolRegistry.register_tool(
    Tool(name="get_weather", func=get_weather, schema=GET_WEATHER_TOOL_SCHEMA)
)
