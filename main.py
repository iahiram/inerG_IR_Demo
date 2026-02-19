from architectures.node_architecture import MultiNodeGraphBuilder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
from client import agent_instances, mymemory, main_config
import logging

builder = MultiNodeGraphBuilder(
    config=main_config,
    memory=mymemory,
    agent_instances=agent_instances,
)

builder.build()

logging.info("Graph built successfully")

def process_request(query: str, message_id: str) -> str:
    if not agent_instances:
        return "No agents available. Please build the graph first."
    
    agent = agent_instances["default"]
    
    config = {
        "recursion_limit": 20,
        "configurable": {"thread_id": message_id},
    }
    
    initial_state = {
        "messages": [HumanMessage(content=json.dumps({"query": query}))]
    }
    
    response = agent.invoke(initial_state, config=config)
    
    # Return the content of the last message
    last_message = response["messages"][-1]
    return last_message.content
