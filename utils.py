from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import logging
import json
import os
import yaml
from schemas.node_schema import ExpandQuery, GenerateAnswer
import asyncio
from typing import Any,Tuple
from client import llm,prompt_data


def get_last_human_question(messages: list)->str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            logging.info(f"Last human message {message}")
            try:
                q_content = json.loads(message.content)
                question = q_content.get("query", message.content)
            except Exception:
                question = message.content
            logging.warning(f"Type of question{type(question)}")
            return str(question)
    return ""


def get_sender(msg):
        if not isinstance(msg, AIMessage):
            return None
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            return data.get("sender")
        except Exception:
            return None

def get_last_question_details(messages: list) -> tuple[str, str, str, str]:
    """Return the last question, answer, sub-category, and main category"""

    formatted_question = ""
    answer = ""

    # Find last guardrail message
    last_guardrail_idx = next(
        (
            i for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], AIMessage)
            and get_sender(messages[i]) == "guardrail"
        ),
        -1,
    )

    if last_guardrail_idx == -1:
        return formatted_question, answer

    guardrail_msg = messages[last_guardrail_idx]

    # If guardrail is the last message, return directly
    if last_guardrail_idx == len(messages) - 1:
        logging.info(f"Last guardrail message {guardrail_msg}")

        try:
            payload = json.loads(guardrail_msg.content)
        except Exception:
            payload = {}

        return (
            guardrail_msg.additional_kwargs.get("formatted_question", ""),
            payload.get("response", ""),
        )

    # Flags
    guardrail_fetch = False
    answer_formulation_fetch = False

    # Walk forward from the last guardrail
    for msg in messages[last_guardrail_idx:]:
        if not isinstance(msg, AIMessage):
            continue

        sender = get_sender(msg)

        try:
            payload = json.loads(msg.content)
        except Exception:
            payload = {}

        if sender == "guardrail" and not guardrail_fetch:
            guardrail_fetch = True
            formatted_question = msg.additional_kwargs.get("formatted_question", "")


        elif sender == "answer_formulation" and not answer_formulation_fetch:
            answer_formulation_fetch = True
            answer = payload.get("response", "")

        
    return formatted_question, answer

def extract_query_and_sender(last_msg:Any,parsed_json:dict)-> Tuple[str,str]:
    
    if isinstance(last_msg,HumanMessage):
        last_sender="human"
        query=parsed_json.get("query","")
    elif parsed_json.get("sender")=="guardrail":
        last_sender ="guardrail"
        query=(last_msg.additional_kwargs or {}).get("expanded_question","") 
    else:
        last_sender=parsed_json.get("sender","unknown")
        query=parsed_json.get("query","")

    return query,last_sender

def get_last_guardrail_question(messages:list)->str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and get_sender(message) == "guardrail":
            logging.info(f"Last guardrail message {message}")
            question=message.additional_kwargs.get("expanded_question","")
            logging.warning(f"Type of question{type(question)}")
            return str(question)
    return ""
def check_guardrail_and_expand_query(query: str, last_question: str | None, last_answer: str | None) -> ExpandQuery:
    
    llm_structured = llm.with_structured_output(ExpandQuery)
    system_prompt = prompt_data['guardrail_system_prompt']
    user_prompt_template = prompt_data['guardrail_user_prompt']
    user_prompt = user_prompt_template.format(query=query, last_question=last_question, last_answer=last_answer)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    result =  llm_structured.invoke(messages)
    return result

def formulate_answer(query: str, retrieved_data: list | None) -> GenerateAnswer:
    llm_structured = llm.with_structured_output(GenerateAnswer)
    system_prompt = prompt_data['answer_formulation_system_prompt']
    user_prompt_template = prompt_data['answer_formulation_user_prompt']
    user_prompt = user_prompt_template.format(query=query, retrieved_data=retrieved_data)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    result =  llm_structured.invoke(messages)
    return result


if __name__ == "__main__":
    # Example usage
    query = "How do I track my shipment that hasn't arrived yet?"
    last_question = None
    last_answer = None
    
    expanded_query = asyncio.run( check_guardrail_and_expand_query(query, last_question, last_answer))
    print(expanded_query)