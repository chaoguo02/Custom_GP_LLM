import json
import logging
import random
import re
import time
from typing import List, Any

import numpy as np
import sympy

from utils.openai_interface import OpenAIInterface

# Part1: 定义全局变量 constraints、init_prompt、crossover_prompt、mutation_prompt
constraints = ["+", "*", "-", "/", "sqrt", "square", "cos", "sin"]
binary_ops = ["+", "-", "*", "/"]
terminals = ["x1", "x2", "-1", "1"]
random_num = None

INIT_PROMPT = """
    You are a mathematical expression generator. 
    ### Generation Rules
    1. Randomly select 2 to 4 operators from the allowed set.
    2. Shuffle the selected operators to ensure diverse orderings
    3. Construct an expression using the selected operators:
       - Use {expression_type} style.
       - Ensure variation in operand placement.
    ### Selected Operators
    {selected_ops}
    ### Allowed Variables & Constants (`terminals`)
    {terminals}
    ### Your Task
    - Generate a unique expression using randomly selected operators.
    - Ensure different structures in each response.
    - Provide no additional text in response. Format output in JSON as {{"expression": "<expression>"}}
"""
CROSSOVER_PROMPT = """
You are given two mathematical expressions {expression}. 

Your task is to recombine these two expressions by performing a single-point crossover, similar to the crossover operation in genetic programming.

Steps:
1. Randomly select one point in each expression.
2. Swap the segments before the selected points in each expression.
3. Combine the first part of the first expression with the second part of the second expression, and vice versa, to create two new expressions.

Please ensure the syntax of the expressions is valid and that the recombined expressions use only the existing terms and operators from the original expressions.

Provide no additional text in response. Format your output in JSON as:{{"expressions": ["<expression>"]}}
"""

MUTATION_PROMPT = """
The goal is to evolve the mathematical expression and create a new expression that differs in structure from the original but still follows mathematical principles.

Given the expression: {expression} 

Use the listed symbols {constraints}.

Provide no additional text in response. Format output in JSON as {{"new_expression": "<new expression>"}}
"""

# Part2: 用于检验LLM生成的表达式是否有效的相关函数
class ProtectedSqrt(sympy.Function):
    # 自定义的平方根函数
    @classmethod
    def eval(cls, x):
        """如果 x 是负数，则返回保护值 1e-6，否则返回标准的 sqrt(x)"""
        if isinstance(x, (sympy.Integer, sympy.Float)):  # 如果是数值类型
            if x < 0:
                return 1e-6  # 对于负数返回保护值
            else:
                return sympy.sqrt(x)  # 对于非负数，返回标准的平方根
        return None  # 对于符号表达式，返回 None 以便进行符号化计算

    @staticmethod
    def _latex(self, printer, *args):
        """定义自定义平方根的 LaTeX 输出格式"""
        return r"\text{protected\_sqrt}(" + printer.doprint(self.args[0]) + r")"

def convert_square_to_root(expression):
    result = ""
    i = 0
    while i < len(expression):
        if expression[i:i + 7] == "square(":
            start = i + 7
            stack = 1
            j = start
            while j < len(expression) and stack > 0:
                if expression[j] == '(':
                    stack += 1
                elif expression[j] == ')':
                    stack -= 1
                j += 1
            inner_expression = expression[start:j - 1]
            result += f"root({inner_expression}, 1/2)"
            i = j
        else:
            result += expression[i]
            i += 1
    return result

def is_valid_expression(expression):
    try:
        # 1. 替换 square(x) -> root(x, 1/2)
        expression = convert_square_to_root(expression)

        # 2. 替换 sqrt(x) -> ProtectedSqrt(x)
        expression = re.sub(r'sqrt\((.*?)\)', r'ProtectedSqrt(\1)', expression)

        # 3. 定义符号变量
        x1, x2 = sympy.symbols('x1 x2')

        # 4. 解析表达式
        expr = sympy.sympify(expression, evaluate=False, locals={'ProtectedSqrt': ProtectedSqrt})

        # 5. 代入数值
        result = expr.subs({x1: 1.1, x2: 1.2})

        # 6. 检查是否为实数
        return result.is_real

    except sympy.SympifyError as e:
        logging.error(f"SympifyError: {e} for expression {expression}")
    except TypeError as e:
        logging.error(f"TypeError: {e} for expression {expression}")
    except Exception as e:
        logging.error(f"Unexpected Error: {e} for expression {expression}")

    return False  # 如果出现错误，返回 False

def check_response_individual_generation(response: str) -> str:
    # 解析LLM生成的JSON响应，提取表达式
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        return "0"
    json_text = match.group(0)
    try:
        expression = json.loads(json_text).get('expression', None)
        if expression is not None and is_valid_expression(expression):
            return expression
        else:
            return "0"
    except json.decoder.JSONDecodeError:
        return "0"


def check_response_crossover(response: str, parents: List[str]):
    """
    解析 LLM 生成的 `response`，提取 `expressions`，如果解析失败，则返回 `parents` 作为备选值。
    """
    try:
        response_cleaned = re.sub(r'```json\n?|```', "", response).replace('\n', '')

        match = re.search(r'\"expressions\"\s*:\s*\[(.*?)\]', response_cleaned, re.DOTALL)
        if not match:
            return parents

        expressions_string = match.group(1)

        expressions = re.findall(r'\{([^\}]+)\}', expressions_string)  # 提取 `{}` 内的表达式
        if expressions:
            cleaned_exprs = [expr.strip().replace('"', '') for expr in expressions]
            return cleaned_exprs if len(cleaned_exprs) == 2 else parents

        final_exprs = [expr.strip().replace('"', '').replace("{", "").replace("}", "")
                       for expr in re.split(r'\s*,\s*', expressions_string)]

        return final_exprs if len(final_exprs) == 2 else parents  # **确保返回两个表达式**

    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"解析 LLM 交叉变异响应失败: {e}")
        return parents  # **解析失败时，返回 `parents` 作为默认值**


def check_mutation_response(response: str, expression: str,) -> str:
    try:

        cleaned_content = re.sub(r'```json\n|```', '', response)  # 去除 Markdown JSON 标记
        cleaned_content = cleaned_content.replace('\n', '')  # 移除换行符

        match = re.search(r'\"new_expression\"\s*:\s*\"(.*?)\"', cleaned_content, re.DOTALL)
        new_expression = match.group(1).strip().replace('"', '') if match else None

        if new_expression and is_valid_expression(new_expression):
            return new_expression  # 返回 LLM 生成的有效表达式
        else:
            return expression  # 如果无效，回退到原始表达式

    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"解析 LLM 变异响应时出错: {e}")
        return expression  # 解析失败，回退到原始表达式

# Part3: 生成init、 crossover、mutation的提示词
def form_prompt_generation(init_prompt) -> str:
    global random_num  # 使用全局变量来追踪上一次的随机数

    # 生成新的 0-1 之间的随机数，并保留两位小数
    new_random_num = str(round(np.random.uniform(0, 1), 2))

    # 如果之前已经有一个随机数，先移除它
    if random_num is not None:
        terminals.remove(random_num)

    # 添加新生成的随机数
    terminals.append(new_random_num)

    # 更新全局变量
    random_num = new_random_num

    num_ops = random.randint(2, 4)
    selected_ops = random.sample(constraints, num_ops)

    # **Ensure at least one binary operator**
    binary_ops = ["+", "-", "*", "/"]
    if not any(op in selected_ops for op in binary_ops):
        selected_ops.append(random.choice(binary_ops))

    # **50% chance to generate `genFull()`, 50% chance to generate `genGrow()`**
    expression_type = "fully-expanded tree (genFull)" if random.random() < 0.5 else "random-growth tree (genGrow)"

    prompt = init_prompt.format(
        expression_type=expression_type,
        selected_ops=selected_ops,
        terminals=terminals,
    )
    # print(f"---------------initial prompt: {prompt}-----------------")

    return prompt

def form_llm_crossover_expressions(expressions, crossover_prompt,) -> str:
    # tree_expressions = []
    # for expr in expressions:
    #     tree_expressions.append(str(expr))
    expressions = " and ".join(expressions)
    prompt = crossover_prompt.format(
        expression=expressions,
    )
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"crossover prompt:{prompt}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return prompt

def form_prompt_rephrase_mutation(
        expression: str, mutation_prompt,
) -> str:

    prompt = mutation_prompt.format(
        expression=expression,
        constraints=constraints,
    )
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"mutation prompt:{prompt}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return prompt

# Part4: 调用LLM，发送请求、接收响应
def collect_llm_generate_expressions(llm_interface: OpenAIInterface, generation_history: list, population_size: int) -> list:
    # 收集LLM生成的表达式，只包括有效的表达式
    expressions = []
    for i in range(population_size):
        # history_prompt = "\n".join(
        #     [f"Generated Expression {idx + 1}: {expr['expression']}" for idx, expr in enumerate(generation_history)]
        # )
        prompt = form_prompt_generation(INIT_PROMPT)
        response = llm_interface.predict_text_logged(prompt, temp=1)
        #
        expression = check_response_individual_generation(response["content"])

        # **存入 `generation_history`**
        generation_history.append({"expression": expression})
        expressions.append(expression)

        print(f"LLM 生成的最终表达式: {expression}")

    return expressions


def llm_crossover_expressions(
    llm_interface: OpenAIInterface,
    parents
) -> List[str]:

    children = parents[:]  # 默认子代继承父代
    # **Step 2: 生成交叉提示语**
    prompt = form_llm_crossover_expressions([_ for _ in parents],CROSSOVER_PROMPT)
    # **Step 3: 调用 LLM 进行交叉**
    response = llm_interface.predict_text_logged(prompt, temp=1.0)
    new_expressions = check_response_crossover(response["content"], parents)
    print(f"new expressions: {new_expressions},type(new_expressions): {type(new_expressions)}")
    if len(new_expressions) == 2 and all(is_valid_expression(expr) for expr in new_expressions):
        children = new_expressions
    else:
        print(is_valid_expression(new_expressions[0]),new_expressions[0])
        print("LLM 生成的表达式无效或数量不足，保持原父代表达式")

    print(f"LLM 交叉后的表达式: {children}")
    return children

def llm_mutated_expressions(
        llm_interface: OpenAIInterface,
        expression: str,
) -> str:

    prompt = form_prompt_rephrase_mutation(expression, MUTATION_PROMPT)
    response = llm_interface.predict_text_logged(prompt, temp=1)

    new_expression = check_mutation_response(response["content"], expression)

    # **检查变异表达式的有效性**
    if new_expression and is_valid_expression(new_expression):
        print(f"LLM生成的变异表达式：{new_expression}")
        return new_expression  # 更新子代表达式
    else:
        print(f"LLM 生成的变异表达式无效，保持原表达式: {expression}")
        return expression
