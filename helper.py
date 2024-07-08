from dotenv import load_dotenv
from IPython.display import display, Markdown
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

#==============================#
# load the LLM and neo4j graph #
#==============================#
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-4o")
graph = Neo4jGraph()


#===================================================#
# Create a Neo4jVector index using OpenAIEmbeddings #
#===================================================#
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


#===============================================#
# Define the Entities Class and Prompt Template #
#===============================================#
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, object, location, or event entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting person, object, location, or event entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query

def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            # Fulltext index query - call entity name "entityIdIndex"
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])

    return result

def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
    """
    return final_data

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be as elaborate as possible.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

def invoke_chain(question: str, chat_history):
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    if chat_history:
        return chain.invoke(
            {
                "question": question,
                "chat_history": chat_history
            }
        )
    else:
        return chain.invoke(
            {
                "question": question,
            }
        )
    

def get_graph():
    return graph

def get_llm():
    return llm
    

############################################################# Graph State #############################################################
class GraphState(TypedDict): # might not actually need langgraph to organise the workflow since only one node "generate"
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search_query: revised question for web search
        context: web_search result
    """
    question : str
    generation : str
    search_query : str
    context : str

# Node - Generate

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("Step: Generating Final Response")
    question = state["question"]

    # Answer Generation
    generation = invoke_chain(question, None)
    return {"generation": generation}

# Conditional Edge, Routing

def route_question(state):
    """
    route question to web search or generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Step: Routing Query")
    question = state['question']
    structured_data = structured_retriever(question)
    
    if len(structured_data) != 0:
        print("Step: Context Found, Routing to Generation")
        return "generate"
    elif len(structured_data) == 0: # Might need to change do deal with a case where context is not found
        print("Step: Context Not Found, Returning to Generation")
        return "generate"
    
def build_workflow():
    """
    Build the workflow for the graph
    """
    # Build the nodes
    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate)

    # Build the edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "generate": "generate",
        },
    )
    workflow.add_edge("generate", END)

    # Compile the workflow
    local_agent = workflow.compile()

    return local_agent

def run_agent(query, local_agent, chat_history):
    output = output = local_agent.invoke({"question": query, "history": chat_history})
    return output['generation']