import os
from rich import print
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, Optional, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# 1. 定义图状态
class MemoryState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


# 2. 创建 LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 3. 定义节点函数
def agent_node(state: MemoryState):
    messages = []
    summary = (state.get("summary") or "").strip()

    if summary:
        messages.append(
            SystemMessage(content=f"对话的摘要如下, 请在此基础上进行回复: {summary}")
        )

    messages.extend(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_summarize(state: MemoryState):
    """
    为避免上下文过长, 历史消息数过多时, 将对历史消息进行摘要
    """
    messages_length = len(state["messages"]) if state.get("messages") else 0

    if messages_length > 12:
        return "summarize"

    return "end"


def summarize_node(state: MemoryState):
    """
    摘要节点, 对历史对话进行压缩并裁减消息, 以减少上下文
    """
    # 选择最近的 10 条消息进行摘要
    recent_n = 10
    recent_messages = state["messages"][-recent_n:]

    # 提取最近的 10 条消息内容: 角色 - 内容
    recent_text = []
    for msg in recent_messages:
        role = getattr(msg, "type", getattr(msg, "role", "message"))
        content = getattr(msg, "content", "")
        recent_text.append(f"- [{role}]: {content}")
    recent_text = "\n".join(recent_text)

    # 基于已有的 summary 做增量摘要
    prev_summary = (state.get("summary") or "").strip()
    system_message = SystemMessage(
        content=(
            "你是一个对话摘要器, 请将下面的若干轮对话中的要点压缩为一段简洁的摘要."
            "要求: 保留人名、偏好、结论类长期信息, 不要逐字复述."
        )
    )
    user_prompt = (
        f"已有摘要: \n{prev_summary if prev_summary else '(无)'}\n\n"
        f"请基于以下最近对话内容更新摘要, 只需要输出更新后的完整摘要: \n{recent_text}"
    )

    summary_result = llm.invoke([system_message, HumanMessage(content=user_prompt)])
    new_summary = summary_result.content.strip()

    # 裁减消息, 仅保留最近 4 条, 确保上下文连贯
    kept_messages = state["messages"][-4:]

    return {"summary": new_summary, "messages": kept_messages}


# 4. 构建图
def create_memory_chat_agent():
    """
    创建带记忆的 Agent, 使用 MemorySaver 作为 checkpointer 使得:
    1. 同一个 thread_id 的多次 .invoke() 会共享/累积状态
    2. 可以只传入锃亮的消息, 其余历史由 checkpointer 自动合并
    """
    graph = StateGraph(MemoryState)

    graph.add_node("agent", agent_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_summarize, {"summarize": "summarize", "end": END}
    )

    graph.add_edge("summarize", "agent")

    # 使用内存检查点, 启动会话级记忆
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)


def run_demo():
    agent = create_memory_chat_agent()

    mermaid_code = agent.get_graph().draw_mermaid()
    print(f"Mermaid 图结构: \n{mermaid_code}")
    print("Mermaid 在线编辑器: https://mermaid.live/")

    # 固定线程ID, 模拟同一会话的多轮调用
    config = {"configurable": {"thread_id": "demo"}}
    print(config)

    # 初始状态: 包哈可选 summary
    state = {"messages": [], "summary": ""}

    # 第一轮
    print(f"[orange1]User>[/] 你好, 我是二次元爱好者, 我喜欢看动漫.")
    state = agent.invoke(
        state
        | {"messages": [HumanMessage(content="你好, 我是二次元爱好者, 我喜欢看动漫.")]},
        config,
    )
    print(f"[blue]Agent>[/] {state["messages"][-1].content}")

    # 第二轮, 不显式携带上下文, 直接传入本次的用户消息
    print(f"[orange1]User>[/] 我想看点温馨治愈的动漫, 有推荐的吗?")
    state = agent.invoke(
        {"messages": [HumanMessage(content="我想看点温馨治愈的动漫, 有推荐的吗?")]},
        config,
    )

    print(f"[blue]Agent>[/] {state["messages"][-1].content}")

    for text in [
        "有类似于轻音少女一样的音乐番吗?",
        "我不太喜欢吵闹的音乐番, 以及剧情不好的音乐番.",
        "你还有类似风格的番推荐吗?",
        "有没有类似于魔法少女小圆一样的番?",
        "另外, 我不喜欢主要剧情是谈恋爱的番.",
    ]:
        print(f"[orange1]User>[/] {text}")
        state = agent.invoke({"messages": HumanMessage(content=text)}, config)
        print(f"[blue]Agent>[/] {state["messages"][-1].content}")

    print(f"[bold yellow]当前摘要:[/] \n{state.get("summary") or '无'}")


def interactive_chat(thread_id: Optional[str] = None):
    """
    交互式聊天(记忆)
    1. 指定 thread_id 后, 可跨进程复用对话记忆
    2. 不指定则使用当前进程时间戳碎机 ID
    """

    print("输入 'quit' 退出聊天")

    agent = create_memory_chat_agent()
    mermaid_code = agent.get_graph().draw_mermaid()
    print(f"Mermaid 图结构: \n{mermaid_code}")
    print("Mermaid 在线编辑器: https://mermaid.live/")

    config = {"configurable": {"thread_id": thread_id or "interactive_session"}}

    state = {"messages": [], "summary": ""}

    while True:
        user_input = input("User> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye~")
            break

        try:
            state = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )
            print(f"[blue]Agent>[/] {state["messages"][-1].content}")
            print(f"[gray]Summary[/]: {state.get("summary") or '无'}")
        except Exception as e:
            print(f"err: {e}")


if __name__ == "__main__":
    # run_demo()
    interactive_chat("demo")
