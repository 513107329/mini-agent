import ast
import operator
from typing import Union

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_node(node: ast.AST) -> Union[int, float]:
    """
    1、如果节点是表达式类型(ast.Expression)，则取得node的body继续进行计算
    2、如果节点是常量，则返回数字
    3、如果节点是二元运算符(ast.BinOp)，则分别对左右节点进行计算，然后调用BIN_OPS中的对应函数(除零保护机制，指数过大防护机制)
    4、如果节点是一元运算符(ast.UnaryOp)，则对节点进行计算，然后调用UNARY_OPS中的对应函数
    7、如果节点是其他类型，则返回错误
    """
    if isinstance(node, ast.Expression):
        return eval_node(node.body)

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant value: {node.value}")

    elif isinstance(node, ast.BinOp):
        type_op = type(node.op)
        if type_op not in BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {type_op}")

        left = eval_node(node.left)
        right = eval_node(node.right)

        if type_op == ast.Div and right == 0:
            raise ZeroDivisionError("Division by zero")

        if type_op == ast.Pow and abs(right) > 1000 or abs(left) > 1000:
            raise ValueError("Exponent too large")

        op = BIN_OPS[type(node.op)]

        return op(left, right)

    elif isinstance(node, ast.UnaryOp):
        type_op = type(node.op)
        if type_op not in UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {type_op}")

        operand = eval_node(node.operand)

        op = UNARY_OPS[type(node.op)]
        return op(operand)

    raise ValueError(f"Unsupported node type: {type(node)}")


def calculator(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression is required")

    try:
        expression_str = expression.strip()
        node = ast.parse(expression_str, mode="eval")
        result = eval_node(node)

        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)

    except ZeroDivisionError:
        return "计算错误：除零错误"
    except SyntaxError:
        return "计算错误：语法错误"
    except Exception as e:
        return f"计算表达式失败: {e}"


CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate the result of a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to calculate.",
                },
            },
            "required": ["expression"],
        },
    },
}


if __name__ == "__main__":
    print(calculator("1 + 2 * 4"))
    print(calculator("1 + 2 * 3"))
