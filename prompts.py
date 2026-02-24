from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

qa_template_str = """Your job is to use resources from the International Food
Policy Research Institute to answer questions about women empowerment in agriculture.
Use the following context to answer questions. Be as detailed
as possible, but don't make up any information that's not
from the context and where possible reference related studies and resources as examples
from the context you have. If you don't know an answer, say you don't know.
Do not state that you are referring to the provided context
and respond as if you were in charge of the WEAI helpdesk.

{context}
"""

ref_template_str = """Your job is to use relevant links and email addresses to
direct users to in order to reach and contact the WEAI team. If you don't know
an answer, say you don't know. Do not state that you are referring to the
provided context and respond as if you were in charge of the WEAI helpdesk.

{context}
"""
doc_template_str = """Your job is to use resources from the International Food
Policy Research Institute to answer questions about women empowerment in agriculture.
Use the following context to answer questions. Be as detailed
as possible, but don't make up any information that's not
from the context and where possible reference related studies and resources as examples
from the context you have. If you don't know an answer, say you don't know.
Do not state that you are referring to the provided context
and respond as if you were in charge of the WEAI helpdesk.

{context}
"""
ref_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=ref_template_str,
    )
)

doc_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=doc_template_str,
    )
)

qa_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

ref_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[ref_system_prompt, qa_human_prompt],
)

doc_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[doc_system_prompt, qa_human_prompt],
)

# qa_chain = qa_prompt_template | chat