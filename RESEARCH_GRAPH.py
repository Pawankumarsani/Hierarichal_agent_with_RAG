import json
import groq
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages
from TOOL import internal_tool_1, tavily_tool, scrape_webpages, llm


class State(MessagesState):
    next: str
    
def _clip(text, n):
    text = str(text)
    return text if len(text) <= n else text[:n] + " …[truncated]"

def compact_history(messages, first_n=1500, rest_n=600):
    out = []
    for i, m in enumerate(messages):
        who = getattr(m, "name", None) or "user"
        out.append({
            "role": "user",
            "content": f"[{who}]: {_clip(m.content, first_n if i == 0 else rest_n)}",
        })
    return out
    
def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status."
        " Do NOT choose FINISH before at least one worker has reported."
        " Call each worker at most once, then respond with FINISH." 
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]
        
    router_llm = llm.with_structured_output(Router, method="json_schema")

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [{"role": "system", "content": system_prompt}] + compact_history(state["messages"])
        try:
            response = router_llm.invoke(messages)
        except groq.BadRequestError as e:
            body = e.body if isinstance(e.body, dict) else {}
            failed = body.get("error", {}).get("failed_generation")
            if not failed:
                raise
            response = json.loads(failed)
            
        goto = response["next"]
        reported = {
            m.name for m in state["messages"]
            if getattr(m, "name", None) in members
        }
        if goto != "FINISH" and (goto in reported or reported >= set(members)):
            goto = "FINISH"

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

search_agent = create_agent(llm, tools=[tavily_tool,internal_tool_1])
search_agent

def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content[:4000], name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

### sub agent
web_scraper_agent = create_agent(llm, tools=[scrape_webpages])


def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content[:4000], name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

if __name__=="__main__":
    
    try:
        png_bytes = research_graph.get_graph(xray=True).draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph visualization to graph.png")
    except Exception as e:
        print(f"Could not render graph image ({e}); printing mermaid source instead:\n")
        print(research_graph.get_graph(xray=True).draw_mermaid())