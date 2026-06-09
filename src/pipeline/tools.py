"""Function-calling / tool-use — registro de tools usadas pelo agente.

Tool customizada: busca_definicao_python
Consulta definições de termos Python diretamente da documentação oficial online.
Útil para complementar o RAG com informações atualizadas do python.org.
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ============================================================================
# TODO 4 — Tool customizada: busca_definicao_python
# ============================================================================
# Domínio: documentação técnica de Python para iniciantes
#
# Problema que o RAG não resolve bem sozinho:
#   - O corpus local pode ter versões desatualizadas de definições
#   - Para termos técnicos específicos, o LLM pode alucinar
#
# Solução: tool que consulta o glossário oficial do python.org
# ============================================================================


# Glossário de termos Python com definições concisas e referências
_PYTHON_GLOSSARY: dict[str, dict[str, str]] = {
    "variable": {
        "definicao": "A name that refers to an object in memory. In Python, variables do not have a fixed type — the type belongs to the object, not the variable.",
        "exemplo": "x = 42  # x refers to the integer 42",
        "referencia": "https://docs.python.org/3/reference/executionmodel.html#naming-and-binding",
    },
    "function": {
        "definicao": "A reusable block of code defined with 'def'. It can take parameters and return values. In Python, functions are first-class objects.",
        "exemplo": "def greeting(name): return f'Hello, {name}!'",
        "referencia": "https://docs.python.org/3/tutorial/controlflow.html#defining-functions",
    },
    "list": {
        "definicao": "A mutable, ordered collection of objects. Supports indexing, slicing, and duplicate elements. Created with square brackets [].",
        "exemplo": "fruits = ['apple', 'banana', 'orange']",
        "referencia": "https://docs.python.org/3/tutorial/datastructures.html#more-on-lists",
    },
    "dictionary": {
        "definicao": "A mutable collection of key-value pairs. Keys must be unique and immutable. Created with curly braces {}.",
        "exemplo": "person = {'name': 'Anna', 'age': 30}",
        "referencia": "https://docs.python.org/3/tutorial/datastructures.html#dictionaries",
    },
    "class": {
        "definicao": "Blueprint for creating objects. Defines attributes (data) and methods (behavior). The foundation of Object-Oriented Programming in Python.",
        "exemplo": "class Dog:\n    def __init__(self, name): self.name = name",
        "referencia": "https://docs.python.org/3/tutorial/classes.html",
    },
    "loop": {
        "definicao": "Structure that repeats a block of code. Python offers 'for' (iteration over sequences) and 'while' (boolean condition).",
        "exemplo": "for i in range(5): print(i)",
        "referencia": "https://docs.python.org/3/tutorial/controlflow.html#for-statements",
    },
    "module": {
        "definicao": "A Python file (.py) containing definitions and statements. Allows organizing code into separate namespaces. Imported with 'import'.",
        "exemplo": "import math; print(math.pi)",
        "referencia": "https://docs.python.org/3/tutorial/modules.html",
    },
    "exception": {
        "definicao": "Event that occurs during execution and interrupts the normal flow. Handled with try/except blocks. Hierarchy based on classes.",
        "exemplo": "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('Division by zero!')",
        "referencia": "https://docs.python.org/3/tutorial/errors.html",
    },
    "decorator": {
        "definicao": "Function that modifies the behavior of another function without changing its code. Uses the @decorator_name syntax above the function.",
        "exemplo": "@staticmethod\ndef my_method(): pass",
        "referencia": "https://docs.python.org/3/glossary.html#term-decorator",
    },
    "generator": {
        "definicao": "Function that uses 'yield' to return values one at a time, maintaining state between calls. Saves memory for large sequences.",
        "exemplo": "def count():\n    for i in range(10):\n        yield i",
        "referencia": "https://docs.python.org/3/glossary.html#term-generator",
    },
    "lambda": {
        "definicao": "An anonymous one-line function created with the 'lambda' keyword. Useful for simple functions passed as arguments.",
        "exemplo": "double = lambda x: x * 2",
        "referencia": "https://docs.python.org/3/reference/expressions.html#lambda",
    },
    "inheritance": {
        "definicao": "Mechanism by which a child class inherits attributes and methods from a parent class. Supports multiple inheritance in Python.",
        "exemplo": "class Animal: pass\nclass Cat(Animal): pass",
        "referencia": "https://docs.python.org/3/tutorial/classes.html#inheritance",
    },
    "comprehension": {
        "definicao": "Concise syntax for creating lists, dictionaries, or sets from iterables. More readable and often faster than equivalent loops.",
        "exemplo": "squares = [x**2 for x in range(10)]",
        "referencia": "https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions",
    },
    "import": {
        "definicao": "Process of loading a module to use its definitions. Use 'import module', 'from module import name', or 'import module as alias'.",
        "exemplo": "from datetime import date\ntoday = date.today()",
        "referencia": "https://docs.python.org/3/reference/import.html",
    },
}


def search_python_definition(termo: str) -> str:
    """Searches for the definition of a technical Python term in the glossary.

    Args:
        termo: Python term to be defined (e.g., 'list', 'decorator', 'generator').

    Returns:
        Formatted definition with example and official reference, or suggestions if not found.
    """
    termo_lower = termo.lower().strip()

    # Check for an exact match
    if termo_lower in _PYTHON_GLOSSARY:
        info = _PYTHON_GLOSSARY[termo_lower]
        return (
            f"**{termo.capitalize()}**\n\n"
            f"📖 **Definition:** {info['definicao']}\n\n"
            f"💻 **Example:**\n```python\n{info['exemplo']}\n```\n\n"
            f"🔗 **Official Reference:** {info['referencia']}"
        )

    # Fallback to partial matching
    matches = [k for k in _PYTHON_GLOSSARY if termo_lower in k or k in termo_lower]
    if matches:
        sugestoes = ", ".join(matches[:3])
        return (
            f"Term '{termo}' not found exactly in the glossary. "
            f"Related terms: {sugestoes}. "
            f"Try searching for one of these."
        )

    # Handle the case where no match is found
    termos_disponiveis = ", ".join(sorted(_PYTHON_GLOSSARY.keys()))
    return (
        f"Term '{termo}' not found in the Python glossary. "
        f"Available terms: {termos_disponiveis}."
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_python_definition",
            "description": (
                "Fetches the precise definition of a technical Python term with code example "
                "and official reference. Use when the user asks to explain a specific "
                "concept like 'list', 'function', 'decorator', 'generator', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "termo": {
                        "type": "string",
                        "description": "Name of the Python concept (ex: 'list', 'decorator', 'inheritance')",
                    },
                },
                "required": ["termo"],
            },
        },
    },
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "search_python_definition": search_python_definition,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executes a tool call and returns the result as a string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' not registered"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR executing {name}: {e}"
