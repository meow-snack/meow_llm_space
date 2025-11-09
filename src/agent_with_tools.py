import os
import requests
from rich import print
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# 1. 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. 创建 LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 3. 定义工具
@tool
def get_current_time() -> str:
    """
    获取当前时间

    returns:
        str: 当前时间
    """
    print(f"调用工具: get_current_time")
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_wikipedia(query: str) -> str:
    """
    查询维基百科

    args:
        query: 搜索关键词

    returns:
        str: 搜索结果摘要
    """
    print(f"调用工具: search_wikipedia")
    try:
        search_url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"标题: {data.get('title', 'Unknown')}\n摘要: {data.get('extract', 'Unknown')}"
    except Exception as e:
        return f"搜索出错: {str(e)}"


tools = [get_current_time, search_wikipedia]

# 4. 绑定工具
llm = llm.bind_tools(tools)


# 5. 图节点定义
def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def agent_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def create_agent_with_tools():
    graph_builder = StateGraph(State)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph_builder.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


def run_demo():
    agent = create_agent_with_tools()
    mermaid_code = agent.get_graph().draw_mermaid()
    print(f"Mermaid 图结构: \n{mermaid_code}")
    print("Mermaid 在线编辑器: https://mermaid.live/")

    config = {"configurable": {"thread_id": "interactive_session"}}
    state: State = {"messages": []}

    while True:
        user_input = input("User> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye~")
            break

        try:
            payload = state | {"messages": [HumanMessage(content=user_input)]}
            state = agent.invoke(payload, config)
            print(f"[blue]Agent>[/] {state['messages'][-1].content}")
        except Exception as e:
            print(f"err: {e}")


if __name__ == "__main__":
    run_demo()
