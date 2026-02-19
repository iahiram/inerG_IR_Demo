from pydantic import BaseModel, Field
from typing import TypedDict,List,Annotated,Optional,Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
class ExpandQuery(BaseModel):
    guardrail_pass : bool 
    expanded_question : str
    content_type : Optional[Literal["policy","procedure","guideline","email","incident_note"]]


class GenerateAnswer(BaseModel):
    answer: str
