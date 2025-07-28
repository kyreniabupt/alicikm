

prompt_template_qc = """\
Given a query and a hierarchical tree structure of product categories, \
determine whether the product category aligns with the query intent.

If any level of the hierarchy does not align with the query intent, return False; otherwise, return True.

Query: {query}
Product Categories: {category}"""

prompt_template_qi = """\
Given a query and a product, please determine whether the product matches the intent of the query.

If the product completely satisfies the user's search query in all aspects (including product type, brand, model, \
attributes, etc.), return True; otherwise, return False.

Query: {query}
Product: {product}"""

prompt_template_dict = {
    'prompt_template_qc': prompt_template_qc,
    'prompt_template_qi': prompt_template_qi
}

####################################################################################################################################################

prompt_template_qc2 = """\
You are a classification assistant.

Given a user query and a category path (with levels separated by commas, from general to specific), determine whether the entire category path aligns with the query intent.

- If any level in the path is unrelated or misleading, respond with `False`.
- Otherwise, respond with `True`.

Answer strictly with either `True` or `False`.

Query: {query}
Category Path: {category}
"""
prompt_template_qi2 = """\
You are a product relevance classification assistant.

Given a user query and an item title, determine whether the item is relevant to the query.

- If the item title clearly matches the intent of the query, respond with `True`.
- If the item title is unrelated, misleading, or only weakly related to the query, respond with `False`.

Answer strictly with either `True` or `False`.

Query: {query}
Item Title: {item_title}
"""
## 只包含Instruction,用于SFT
instruction_template_qc = """\

You are a classification assistant.

Given a user query and a category path (with levels separated by commas, from general to specific), determine whether the entire category path aligns with the query intent.

- If any level in the path is unrelated or misleading, respond with `False`.
- Otherwise, respond with `True`.

Answer strictly with either `True` or `False`.

"""
instruction_template_qi = """\

You are a product relevance classification assistant.

Given a user query and an item title, determine whether the item is relevant to the query.

- If the item title clearly matches the intent of the query, respond with `True`.
- If the item title is unrelated, misleading, or only weakly related to the query, respond with `False`.

Answer strictly with either `True` or `False`.

"""
instruction_template_dict = {
    'instruction_template_qc': instruction_template_qc,
    'instruction_template_qi': instruction_template_qi
}
