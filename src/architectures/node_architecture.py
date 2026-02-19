from src.models.node_schema import AgentState
from src.node import guardrail_node, data_retriever_node, answer_formulation_node
from langgraph.graph import END, StateGraph
from src.router import guardrail_router

node_lookup = {
    "guardrail_node": guardrail_node,
    "data_retrieval_node": data_retriever_node,
    "answer_formulation_node": answer_formulation_node,
    "__END__": END,
}

router_lookup = {
    "guardrail_router": guardrail_router
}

class MultiNodeGraphBuilder:
    def __init__(self, config, memory, agent_instances, output_path="graph.png"):
        self.config = config
        self.memory = memory
        self.agent_instances = agent_instances
        self.output_path = output_path or "/tmp/graph.png"
        self.graph = StateGraph(AgentState)

    def add_nodes(self):
        """
        Adds nodes to the state graph based on configuration.
        """
        nodes = self.config["node_architecture"]["nodes"]
        for node in nodes:
            if node not in node_lookup:
                raise ValueError(f"Unknown node: {node}")
            self.graph.add_node(node, node_lookup[node])

    def add_edges(self):
        """
        Adds edges to the graph.
        """
        edges = self.config["node_architecture"]["edges"]
        
        # Direct edges
        for edge in edges.get("edge", []):
            source = END if edge["source"] == "__END__" else edge["source"]
            dest = END if edge["destination"] == "__END__" else edge["destination"]
            self.graph.add_edge(source, dest)
        
        # Conditional edges
        for edge in edges.get("conditional_edges", []):
            router = router_lookup.get(edge["router"])
            if not router:
                raise ValueError(f"Unknown router: {edge['router']}")
            mapping = {
                k: (END if v == "__END__" else v) for k, v in edge["mapping"].items()
            }
            self.graph.add_conditional_edges(edge["source"], router, mapping)

    def set_entrypoint(self):
        """
        Sets the entry point.
        """
        entry = self.config["node_architecture"]["entrypoint"]
        if entry not in node_lookup:
            raise ValueError("Invalid entrypoint")
        self.graph.set_entry_point(entry)

    def compile_and_draw(self):
        """
        Compiles the graph and draws it.
        """
        compiled = self.graph.compile(checkpointer=self.memory)
        try:
            png_bytes = compiled.get_graph().draw_mermaid_png()
            with open(self.output_path, "wb") as f:
                f.write(png_bytes)
        except Exception as e:
            print(f"Graph drawing failed: {e}")
        self.agent_instances["default"] = compiled

    def build(self):
        self.add_nodes()
        self.add_edges()
        self.set_entrypoint()
        self.compile_and_draw()
