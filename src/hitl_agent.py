import os
from langgraph.checkpoint import memory
from rich import print
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# 1. 定义图状态
class HITLState(TypedDict):
    """HITL 状态结构
    - messages: 对话信息流
    - task: 用户任务/需求
    - draft: 当前草稿
    - approved: 是否已批准
    - feedback: 人类反馈内容, 为空表示直接批准
    """

    messages: Annotated[list, add_messages]
    task: str
    draft: str
    approved: bool
    feedback: str


# 2. 创建 LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 3. 定义节点函数
def plan_node(state: HITLState):
    system_message = SystemMessage(
        content=(
            "你是一个资深助手, 请根据用户的任务草拟一个清晰, 可执行的方案"
            "你需要用中文进行回答, 且回答内容要结构化, 条理清晰"
        )
    )

    user_message = HumanMessage(content=f"任务: {state['task']}\n请输出一个初稿.")

    response = llm.invoke([system_message, user_message])

    draft = response.content.strip()
    return {"messages": [response], "draft": draft, "approved": False, "feedback": ""}


def human_feedback_node(state: HITLState):
    print(f"当前任务草稿: {state['draft']}")

    approved = False
    feedback = ""

    while True:
        ans = input("是否批准当前草稿?(y/n): ").strip().lower()
        if ans in ["y", "yes"]:
            approved = True
            break
        if ans in ["n", "no"]:
            approved = False
            break
    if not approved:
        print("请输入您的修改建议, 完成后直接回车提交(可多行, 输入单独的'.'结束): ")
        lines = []
        while True:
            line = input()
            if line.strip() == ".":
                break
            lines.append(line)
        feedback = "\n".join(lines).strip()
        if not feedback:
            feedback = "请更加清晰、具体, 并补充细节."

    return {"approved": approved, "feedback": feedback}


def revise_node(state: HITLState):
    system_message = SystemMessage(
        content=(
            "你是一个严谨的编辑, 请基于反馈内容对草稿进行修订."
            "保留原有优点, 针对反馈做明确改进, 必要时补充细节与结构."
        )
    )

    user_message = HumanMessage(
        content=(
            f"原始任务: {state['task']}\n\n"
            f"当前草稿: {state['draft']}\n\n"
            f"用户反馈: {state['feedback']}\n\n"
            f"情书出修订后的完整版本."
        )
    )

    response = llm.invoke([system_message, user_message])

    new_draft = response.content.strip()

    return {"messages": [response], "draft": new_draft, "approved": False}


# 4. 定义路由条件
def route_after_feedback(state: HITLState):
    return "end" if state["approved"] else "revise"


# 5. 构建图


def create_hitl_graph():
    graph_builder = StateGraph(HITLState)

    graph_builder.add_node("plan", plan_node)
    graph_builder.add_node("feedback", human_feedback_node)
    graph_builder.add_node("revise", revise_node)

    graph_builder.add_edge(START, "plan")
    graph_builder.add_edge("plan", "feedback")
    graph_builder.add_conditional_edges(
        "feedback", route_after_feedback, {"end": END, "revise": "revise"}
    )
    graph_builder.add_edge("revise", "feedback")

    memory = MemorySaver()

    return graph_builder.compile(checkpointer=memory)


def run_demo():
    agent = create_hitl_graph()
    mermaid_code = agent.get_graph().draw_mermaid()
    print(f"Mermaid 图结构: \n{mermaid_code}")
    print("Mermaid 在线编辑器: https://mermaid.live/")

    config = {"configurable": {"thread_id": "interactie_session"}}

    task = input("请输入任务描述: ").strip()

    state = {
        "messages": [],
        "task": task,
        "draft": "",
        "approved": False,
        "feedback": "",
    }

    state = agent.invoke(state, config)

    while True:
        state = agent.invoke(state, config)
        if state["approved"]:
            print(f"用户已批准, 最终版本: \n{state['draft']}")
            break
        else:
            print(f"已收到用户反馈, 开始进行修正")


if __name__ == "__main__":
    run_demo()
