from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from rag import create_rag_db
from model import load_model
from config import Config

llm = load_model(repo_id=Config.REPO_ID, hf_token=Config.HF_TOKEN, use_hf_endpoint=True, billing_account=Config.BILLING_ACCOUNT)
docs_vector_db, refs_vector_db = create_rag_db(docs_path=Config.DOCS_PATH, refs_path=Config.REFS_PATH)

@dynamic_prompt
def ref_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    ref_content = refs_vector_db.as_retriever(k=10)

    system_message = (
        """Your job is to use relevant links and email addresses to
        direct users to in order to reach and contact the WEAI team. Do not use links 
        or contacts not provided in the context.If you don't know
        an answer, say you don't know. Do not state that you are referring to the
        provided context and respond as if you were in charge of the WEAI helpdesk."""
        f"\n\n{ref_content}"
    )

    return system_message

contact_agent = (create_agent(llm, tools=[], middleware=[ref_context]))

@tool("contact", description="refer users to WEAI team using links and contact details")
def call_contact_agent(query: str):
    result = contact_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@dynamic_prompt
def doc_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    doc_content = docs_vector_db.as_retriever(k=10)

    system_message = (
        """Your job is to use resources from the International Food
            Policy Research Institute to answer questions about women empowerment in agriculture.
            Use the following context to answer questions. Be as detailed
            as possible, but don't make up any information that's not
            from the context and where possible reference related studies and resources
            from the context you have. Use complete paper or article details such as authors, title, publication date, and link if available.
            Do not use publication information not provided in the context and do not combine publication information to make up details. 
            Use complete information as referenced in the context. If you don't know an answer, say you don't know.
            Be concise but thorough in your response and try not to exceed the output token limit.
            Do not state that you are referring to the provided context and respond
            as if you were in charge of the WEAI helpdesk. """
        f"\n\n{doc_content}"
    )

    return system_message

support_agent = (create_agent(llm, tools=[], middleware=[doc_context]))


@tool("support", description="respond to user queries using context in WEAI docs")
def call_support_agent(query: str):
    result = support_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

support_instructions = """
You are in charge of the WEAI helpdesk.
Your job is to answer user queries using provided context and references
and refer users to WEAI personnel as well as relevant resource links where necessary.

Steps:
1. Use the support tool to answer queries to the best of your knowledge.
2. If no contact information or links are provided in the response, use the
   contact tool to add all relevant contact and resource information to the response.
3. Return only a complete response with included contact and resource information.
"""

response_agent = create_agent(model=llm,
                              tools=[call_contact_agent, call_support_agent],
                              system_prompt=support_instructions,
                              )