import os
from langchain_core.messages import HumanMessage, SystemMessage
from rich import print
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

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


# 3. 定义节点函数
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. 构建图
def create_simple_chatbot():
    # 4.1 定义状态图
    graph_builder = StateGraph(State)

    # 4.2 添加节点
    graph_builder.add_node("chatbot", chatbot)

    # 4.3 添加边
    graph_builder.add_edge(start_key=START, end_key="chatbot")
    graph_builder.add_edge(start_key="chatbot", end_key=END)

    # 4.4 编译图
    return graph_builder.compile()


# 5. 使用 Agent
def main():
    # 5.1 创建聊天机器人
    chatbot_agent = create_simple_chatbot()
    mermaid_code = chatbot_agent.get_graph().draw_mermaid()
    print(f"Mermaid 图结构: \n{mermaid_code}")
    print("Mermaid 在线编辑器: https://mermaid.live/")

    # 5.2 准备要发送的消息
    messages = [
        SystemMessage(content="请使用中文作为工作语言, 模仿平泽唯回复用户的问题."),
        HumanMessage(content="请介绍一下你自己~"),
    ]

    # 5.3 调用 Agent
    result = chatbot_agent.invoke({"messages": messages})

    # 5.4 打印结果
    print("用户> 请介绍一下你自己~")
    print(f"Agent> {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
