from dotenv import load_dotenv

load_dotenv()

import os

import asyncio
from rich.text import Text
from rich.panel import Panel
from rich.console import Console


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.agent_toolkits import FileManagementToolkit

console = Console()


async def chat_loop(agent) -> None:
    welcome_panel = Panel(
        "[bold green]Welcome to Meow-Search Chat![/bold green]\n\n[white]Type your messages below. Use 'exit' or 'quit' to end the conversation.[/white]",
        title="🐱 Meow-Search Chat Console",
        title_align="center",
        padding=(1, 2),
    )

    console.print(welcome_panel)
    history: list[dict[str, str]] = []

    while True:
        try:
            user_prompt = Text("You> ", style="bold cyan")
            user_message = console.input(user_prompt).strip()
        except EOFError:
            console.print("\n[yellow]Input stream closed. Exiting.[/yellow]")
            break

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            farewell_text = Text()
            farewell_text.append(" 🐱 ", style="bold yellow")
            farewell_text.append("Thanks for chatting! Goodbye! ", style="bold green")
            farewell_text.append("👋", style="bold blue")
            console.print(farewell_text)

            if history:
                stats_panel = Panel(
                    f"[green]Chat statistics:[/green]\n"
                    + f"• Total exchanges: {len(history)//2}\n"
                    + f"• Your messages: {len([m for m in history if m['role'] == 'user'])}",
                    title="📊 Session Summary",
                    border_style="blue",
                )
                console.print(stats_panel)
            break

        history.append({"role": "user", "content": user_message})

        thinking_text = Text("Meow-Search is thinking... ", style="bold magenta")

        try:
            with console.status(thinking_text, spinner="dots"):
                result = await agent.ainvoke({"messages": history})
        except Exception as e:
            error_panel = Panel(
                f"[red]{e}[/red]", title="❌ Error", border_style="red", style="white"
            )
            console.print(error_panel)
            history.pop()
            continue

        assistant_message = result["messages"][-1].content
        if not assistant_message:
            assistant_message = "[italic dim]No response generated.[/italic dim]"

        if len(assistant_message) < 100:
            reply_text = Text()
            reply_text.append("🐱 Meow-Search> ", style="bold yellow")
            reply_text.append(assistant_message, style="white")
            console.print(reply_text)
        else:
            reply_panel = Panel(
                assistant_message,
                title="🐱 Meow-Search Response",
                title_align="left",
                border_style="yellow",
                style="white",
                padding=(1, 2),
            )
            console.print(reply_panel)

        history.append({"role": "assistant", "content": assistant_message})


async def main() -> None:
    model = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model=os.getenv("DEEPSEEK_MODEL"),
    )
    mcp_client = MultiServerMCPClient(
        {
            "meow-search": {
                "transport": "stdio",
                "command": "python",
                "args": ["./search_mcp.py"],
            }
        }
    )
    tools = await mcp_client.get_tools()
    toolkit = FileManagementToolkit(root_dir=os.path.join(os.getcwd(), "outputs"))
    file_tools = toolkit.get_tools()
    tools.extend(file_tools)

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=(
            "You are a professional research report assistant. For each task:\n"
            "- Collect supporting evidence before drafting.\n"
            "- Output in this order: Executive Summary → Key Findings → Detailed Analysis → Risks & Uncertainties → Recommendations/Next Steps.\n"
            "- Cite sources or tool calls for key facts; flag assumptions or low confidence.\n"
            "- Keep the tone formal, concise, and objective; clarify complex ideas briefly.\n"
            "- Follow user formatting instructions when they exist; otherwise use this structure."
        ),
    )

    await chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
