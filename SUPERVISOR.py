from typing import Literal
from langgraph.types import Command
from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from TOOL import llm
from RESEARCH_GRAPH import make_supervisor_node, research_graph, State
from WRITER_GRAPH import paper_writing_graph

teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

def call_research_team(state: State) -> Command[Literal["supervisor"]]:
    response = research_graph.invoke({"messages": state["messages"][-1]}, config={"recursion_limit": 25})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
        },
        goto="supervisor",
    )


def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
    response = paper_writing_graph.invoke({"messages": state["messages"][-1]}, config={"recursion_limit": 25})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ]
        },
        goto="supervisor",
    )


# Define the graph.
super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

try:
    png_bytes = super_graph.get_graph(xray=True).draw_mermaid_png()
    with open("Hierarichal_agent_RAG.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph visualization to Hierarichal_agent_RAG.png")
except Exception as e:
    print(f"Could not render graph image ({e}); printing mermaid source instead:\n")
    print(super_graph.get_graph(xray=True).draw_mermaid())
    
if __name__=="__main__":
    
    response = super_graph.invoke(
        {"messages": [HumanMessage(content="Write about transformer variants in production deployments.")]},
        config={"recursion_limit": 25},
    )
    
    final = response["messages"][-1].content
    print("\n--- FINAL ANSWER ---\n")
    print(final)