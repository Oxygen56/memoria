"""Memoria CLI — interact with your agent memory from the terminal."""

from __future__ import annotations

import argparse
import asyncio
import os


def main() -> None:
    """Entry point for the `memoria` command."""
    parser = argparse.ArgumentParser(
        prog="memoria",
        description="Memory infrastructure for AI agents",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command")

    # remember
    remember_parser = subparsers.add_parser("remember", help="Store a memory")
    remember_parser.add_argument("text", help="Memory content to store")
    remember_parser.add_argument(
        "--type",
        default="fact",
        choices=["fact", "preference", "event", "decision", "relationship", "skill", "constraint"],
        help="Memory type (default: fact)",
    )

    # recall
    recall_parser = subparsers.add_parser("recall", help="Recall relevant memories")
    recall_parser.add_argument("query", help="What to recall")

    # stats
    subparsers.add_parser("stats", help="Show memory statistics")

    args = parser.parse_args()

    if args.version:
        from memoria import __version__

        print(f"memoria {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    asyncio.run(_run_command(args))


async def _run_command(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate async command handler."""
    from memoria import Memoria
    from memoria.embedding.base import DummyEmbeddingProvider, OpenAIEmbeddingProvider

    # Use OpenAI if API key available, otherwise Dummy
    if os.environ.get("OPENAI_API_KEY"):
        embedding = OpenAIEmbeddingProvider()
    else:
        embedding = DummyEmbeddingProvider()

    async with Memoria(embedding=embedding, data_dir="~/.memoria") as mem:
        if args.command == "remember":
            await _cmd_remember(mem, args)
        elif args.command == "recall":
            await _cmd_recall(mem, args)
        elif args.command == "stats":
            await _cmd_stats(mem)


async def _cmd_remember(mem, args: argparse.Namespace) -> None:
    """Store a memory."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    record = await mem.remember(content=args.text, memory_type=args.type)

    console.print(
        Panel(
            f"[bold green]✓ Memory stored[/bold green]\n\n"
            f"[dim]ID:[/dim]   {record.id}\n"
            f"[dim]Type:[/dim] {record.memory_type.value}\n"
            f"[dim]Layer:[/dim] {record.layer.value}\n"
            f"[dim]Content:[/dim] {record.content}",
            title="[bold]memoria remember[/bold]",
            border_style="green",
        )
    )


async def _cmd_recall(mem, args: argparse.Namespace) -> None:
    """Recall relevant memories."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    context = await mem.recall(input_text=args.query)

    # Display hot memories
    if context.hot:
        hot_table = Table(title="🔥 Hot Memories (always injected)", border_style="red")
        hot_table.add_column("ID", style="dim", max_width=16)
        hot_table.add_column("Type", style="cyan")
        hot_table.add_column("Content", style="white")
        for record in context.hot:
            hot_table.add_row(record.id, record.memory_type.value, record.content)
        console.print(hot_table)
        console.print()

    # Display relevant memories
    if context.relevant:
        rel_table = Table(title="🎯 Relevant Memories", border_style="blue")
        rel_table.add_column("ID", style="dim", max_width=16)
        rel_table.add_column("Score", style="yellow", justify="right")
        rel_table.add_column("Type", style="cyan")
        rel_table.add_column("Content", style="white")
        for item in context.relevant:
            rel_table.add_row(
                item.memory.id,
                f"{item.relevance_score:.3f}",
                item.memory.memory_type.value,
                item.memory.content,
            )
        console.print(rel_table)
    elif not context.hot:
        console.print("[yellow]No memories found for query:[/yellow]", args.query)

    console.print(f"\n[dim]Total tokens used: {context.total_tokens}[/dim]")


async def _cmd_stats(mem) -> None:
    """Show memory statistics."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    stats = await mem.stats()

    # Main stats table
    table = Table(title="📊 Memoria Statistics", border_style="magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")

    table.add_row("Total Memories", str(stats.storage.total_memories))
    table.add_row("Backend", stats.storage.backend_type)
    table.add_row("Health Score", f"{stats.health_score:.1f}%")
    table.add_row("Uptime", f"{stats.uptime_seconds:.1f}s")
    table.add_row("Storage Size", _format_bytes(stats.storage.storage_size_bytes))

    console.print(table)

    # Layer breakdown
    if stats.storage.by_layer:
        layer_table = Table(title="Memories by Layer", border_style="dim")
        layer_table.add_column("Layer", style="bold")
        layer_table.add_column("Count", style="cyan", justify="right")
        for layer, count in stats.storage.by_layer.items():
            layer_table.add_row(layer, str(count))
        console.print(layer_table)

    # Type breakdown
    if stats.storage.by_type:
        type_table = Table(title="Memories by Type", border_style="dim")
        type_table.add_column("Type", style="bold")
        type_table.add_column("Count", style="cyan", justify="right")
        for mtype, count in stats.storage.by_type.items():
            type_table.add_row(mtype, str(count))
        console.print(type_table)


def _format_bytes(n: int) -> str:
    """Format byte count in human-readable form."""
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


if __name__ == "__main__":
    main()
