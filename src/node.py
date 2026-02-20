from src.models.node_schema import AgentState
import logging
import json
from src.utils import extract_query_and_sender, check_guardrail_and_expand_query,get_last_question_details,formulate_answer, get_last_guardrail_question
from src.qdrant.qdrant_utils import get_retrieved_data
from langchain_core.messages import AIMessage, HumanMessage

def guardrail_node(state: AgentState) -> AgentState:
    messages = state.get("messages")
    if not messages:
        raise ValueError("State is empty")
    last_msg = messages[-1]
    raw_content = last_msg.content
    logging.info(f"Raw last message content: {raw_content}")
    if isinstance(raw_content, str):
        try:
            parsed_json: dict = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            parsed_json = {}
    elif isinstance(raw_content, dict):
        parsed_json = raw_content
    elif (
        isinstance(raw_content, list)
        and raw_content
        and isinstance(raw_content[0], dict)
    ):
        parsed_json = raw_content[0]
    else:
        logging.error(
            "Failed to parse last message content as JSON:  Using empty dictionary instead."
        )
        parsed_json = {}
    query, last_sender = extract_query_and_sender(last_msg, parsed_json)
    logging.info(f"last_sender: {last_sender}, query: {query}")
    if not query:
        logging.warning(
            "Query is empty. This may indicate malformed input or failed JSON parsing."
        )
    last_formatted_question,last_answer = (
        get_last_question_details(messages)
    )

    response = check_guardrail_and_expand_query(query,last_formatted_question,last_answer)
    logging.info(f"Expanded query: {response}")
    if not response.guardrail_pass:
        return {**state,"messages":[AIMessage(sender="guardrail",response="Guardrail check failed. Cannot proceed further.",additional_kwargs={"guardrail_pass": False})]}
    ai_msg= AIMessage(
        content=json.dumps({"sender": "guardrail", "query": query}, ensure_ascii=False),
        additional_kwargs={
            "guardrail_pass": response.guardrail_pass,
            "expanded_question": response.expanded_question,
            "content_type": response.content_type,
            
        },
    )

    return  {**state, "messages": [ai_msg]}

def data_retriever_node(state: AgentState) -> AgentState:
    messages = state.get("messages")
    if not messages:
        raise ValueError("State is empty")
    try:
        last_msg = messages[-1]

        query,content_type = last_msg.additional_kwargs.get("expanded_question",""), last_msg.additional_kwargs.get("content_type","")
        logging.info(f"Data retriever received query: {query} with content type: {content_type}")
    except Exception as e:
        logging.error(f"Error extracting query and content type: {e}")
        ai_msg= AIMessage(
        content=json.dumps({"sender": "data_retriever", "retrieved_data": "" }, ensure_ascii=False),)
        return  {**state, "messages": [ai_msg]}
    try:
        data=""    
        retrieved_data=get_retrieved_data(query,content_type)
        logging.info(f"Retrieved data:")
        for doc,score in retrieved_data:
            logging.info(f"Document content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
            logging.info("-" * 50)  
        
        final = "\n".join(
                    [doc.page_content + f" Metadata: {str(doc.metadata)} Score: {score}" 
                    for doc, score in retrieved_data 
                    if score > 0.5]
                )

        ai_msg= AIMessage(
        content=json.dumps({"sender": "data_retriever", "retrieved_data": final }, ensure_ascii=False),)
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        ai_msg= AIMessage(
        content=json.dumps({"sender": "data_retriever", "retrieved_data": "" }, ensure_ascii=False),)
    return  {**state, "messages": [ai_msg]}
    
def answer_formulation_node(state: AgentState) -> AgentState:
    messages= state.get("messages")
    if not messages:
        raise ValueError("State is empty")
    last_msg = messages[-1]
    retrieved_data = json.loads(last_msg.content).get("retrieved_data","")
    query=get_last_guardrail_question(messages)
    logging.info(f"Answer formulation received retrieved data: {retrieved_data}")
    answer =  formulate_answer(query,retrieved_data)
    logging.info(f"Formulated answer: {answer}")
    ai_msg= AIMessage(
        content=json.dumps({"sender": "answer_formulation", "response": answer.answer,"retrieved_data_used": retrieved_data}, ensure_ascii=False),)
        
    return  {**state, "messages": [ai_msg]}