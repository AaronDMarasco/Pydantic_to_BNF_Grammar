import itertools
import pathlib
from pydantic import BaseModel, Field
from enum import Enum
from types import NoneType, UnionType
from typing import List, get_type_hints, Literal, get_args, get_origin, TypeVar
from datetime import datetime
from components.test_cases import Character, KnowledgeGraph, ThoughtAnswerResponse, Cities, CarAndOwner

def is_generic_typing(type_):
    return hasattr(type_, '__origin__')

type_mappings = {
    int: "number",
    float: "number",
    str: "string",
    bool: "boolean",
    datetime: "datetime",
    pathlib.Path: "string",
}

def handle_basic_type(field_name, field_type):
    bnf_type = type_mappings.get(field_type, None)
    if bnf_type is None:
        raise NotImplementedError(f"Type {field_type} for {field_name} is not supported yet")
    return f'"{field_name}" ws ":" ws {bnf_type}'

def handle_enum(field_name, field_type):
    enum_values = ' | '.join([f'"{name}"' for name in field_type])
    return [f"{field_name.capitalize()} ::= {enum_values}"], f'"{field_name}" ws ":" ws {field_name.capitalize()}'

def handle_literal(field_name, field_type):
    literal_values = ' | '.join([f'"{literal}"' for literal in get_args(field_type)])
    return f"{field_name.capitalize()} ::= {literal_values}", f'"{field_name}" ws ":" ws {field_name.capitalize()}'

def handle_generic_typing(field_name, field_type, pydantic_model_to_bnf):
    inner_type = field_type.__args__[0]
    if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
        nested_rules = pydantic_model_to_bnf(inner_type, root=False).split('\n')
        model_rules = f'"{field_name}" ws ":" ws {field_name.capitalize()}list'
        return nested_rules, model_rules
    else:
        basic_type_name = getattr(inner_type, '__name__', None) or inner_type.__class__.__name__.lower()
        return [], f'"{field_name}" ws ":" ws {basic_type_name}list'

def handle_nested_model(field_name, field_type, pydantic_model_to_bnf):
    nested_rules = pydantic_model_to_bnf(field_type, root=False).split('\n')
    model_rules = f'"{field_name}" ws ":" ws {field_type.__name__}'
    return nested_rules, model_rules

def handle_union(field_name, field_type, pydantic_model_to_bnf):
    union_types = list(get_args(field_type)) if isinstance(field_type, UnionType) else field_type
    if NoneType in union_types:
        union_types.remove(NoneType)
    output = [generate_rule_for_field(field_name, t, pydantic_model_to_bnf) for t in union_types]
    additional_rules = list(itertools.chain.from_iterable(x[0] for x in output if x[0]))
    extra_rule_types = set(x[1].split()[-1] for x in output[1:])
    # build the list of types by appending " | type2" for each of the remaining (but remove any dupes)
    # e.g. "int | float" => "number | number" without filtering
    first_type = output[0][1].split()[-1]
    if first_type in extra_rule_types:
        extra_rule_types.remove(first_type)
    rule = " | ".join(itertools.chain([output[0][1]], extra_rule_types))
    return additional_rules, rule

def generate_rule_for_field(field_name, field_type, pydantic_model_to_bnf):
    if isinstance(field_type, TypeVar):
        assert field_type.__constraints__, f"Cannot parse un-constrained TypeVar type {field_type} for field {field_name}"
        # Unpack the possible types and re-parse
        return generate_rule_for_field(field_name, field_type.__constraints__, pydantic_model_to_bnf)
    if isinstance(field_type, UnionType):
        return handle_union(field_name, field_type, pydantic_model_to_bnf)
    # Because you cannot re-generate UnionTypes at runtime, we accept a UnionType OR a tuple of types
    if isinstance(field_type, tuple) and set(type(x) for x in field_type) == {type}:
        return handle_union(field_name, field_type, pydantic_model_to_bnf)
    if is_generic_typing(field_type):
        return handle_generic_typing(field_name, field_type, pydantic_model_to_bnf)
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return handle_nested_model(field_name, field_type, pydantic_model_to_bnf)
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return handle_enum(field_name, field_type)
    if get_origin(field_type) == Literal:
        return handle_literal(field_name, field_type)
    if field_type in type_mappings:
        return [], handle_basic_type(field_name, field_type)
    raise NotImplementedError(f"Type {field_type} for {field_name} is not supported yet")

def add_common_bnf_components():
    return [
        "string ::= '\"' ([^\"]*) '\"'",
        "number ::= [0-9]+ ('.' [0-9]*)?",
        "datetime ::= string",  # Represent datetime as a string in the BNF
        "ws ::= [ \\t\\n]*",
        "boolean ::= 'true' | 'false'",
        "stringlist ::= '[' ws ']' | '[' ws string (ws ',' ws string)* ws ']'",
        "numberlist ::= '[' ws ']' | '[' ws number (ws ',' ws number)* ws ']'",
    ]

def _unique_everseen(iterable):
    """List unique elements, preserving order. Remember all elements ever seen. (copied/modified from itertools help)"""
    seen = set()
    for element in itertools.filterfalse(seen.__contains__, iterable):
        seen.add(element)
        yield element

def pydantic_model_to_bnf(model_cls, root=True):
    fields = get_type_hints(model_cls)
    bnf_grammar = []

    if root:
        bnf_grammar.append(f"root ::= {model_cls.__name__}")

    model_rules = []
    for field_name, field_type in fields.items():
        optional = isinstance(field_type, UnionType) and NoneType in get_args(field_type)
        # We handle optional here and not in generate_rule_for_field to avoid passing optional flag everywhere
        additional_rules, rule = generate_rule_for_field(field_name, field_type, pydantic_model_to_bnf)
        bnf_grammar += additional_rules  # Extend with additional rules without newline split
        model_rules.append(f"[ {rule} ws ',' ] " if optional else f"{rule} ws ',' ")

    if model_rules:
        # Need to remove the last whitespace+comma but it may be inside a bracket if it was optional. Not very pythonic, but JSON's rules not ours.
        bad_trailing = " ws ','"
        last_loc = model_rules[-1].rindex(bad_trailing)
        model_rules[-1] = model_rules[-1][:last_loc] + model_rules[-1][last_loc+len(bad_trailing):]

    fields_bnf = " ws ".join(model_rules)
    bnf_grammar.insert(1, f"{model_cls.__name__} ::= '{{' ws {fields_bnf} ws '}}'")
    # Remove duplicates; e.g. if five fields all use MyCoolType, there would be five copies of MyCoolType in bnf_grammar
    bnf_grammar = list(_unique_everseen(bnf_grammar))
    # Sanity check (make sure after uniqueness check, we don't have two definitions for a single custom type)
    defs = sorted(x.split('::=')[0] for x in bnf_grammar)
    assert len(defs) == len(set(defs)), f"Some definitions seem to repeat! {defs}"

    if root:
        bnf_grammar += add_common_bnf_components()  # Extend with common components without newline split

    return '\n'.join(bnf_grammar)


if __name__ == '__main__':
    for test_case in [Character, KnowledgeGraph, ThoughtAnswerResponse, Cities, CarAndOwner]:     
        bnf_grammar = pydantic_model_to_bnf(test_case)
        print(test_case)
        print(bnf_grammar, '\n\n\n')
    
