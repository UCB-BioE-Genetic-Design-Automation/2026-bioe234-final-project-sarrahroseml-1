"""Frontend chat agent — backed by the creseq MCP server via stdio transport.

Tools are discovered automatically from the running MCP server, so any tool
added to server.py is immediately available here without any changes to this file.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

UPLOAD_DIR = Path(os.environ.get("CRESEQ_UPLOAD_DIR", Path.home() / ".creseq" / "uploads"))
ASSOC_DIR  = Path(os.environ.get("CRESEQ_ASSOC_DIR",  Path.home() / "creseq_outputs"))

_SYSTEM_PROMPT = (
    "You are a CRE-seq analysis assistant backed by real tools. "
    "File path arguments are optional — omit them and the tools will automatically use "
    "data from the upload directory. "
    "Summarise results clearly: state PASS/FAIL for QC, report active element counts and "
    "top enriched motifs for downstream steps. "
    "The full pipeline order is:\n"
    "  1. tool_run_association — barcode → oligo mapping\n"
    "  2. tool_process_dna_counting / tool_process_rna_counting — count barcodes\n"
    "  3. tool_activity_report — normalize + call active CREs\n"
    "  4. tool_annotate_motifs — annotate top TF motif per CRE\n"
    "  5. tool_motif_enrichment or tool_motif_enrichment_summary — enrichment analysis\n"
    "  6. tool_variant_delta_scores — variant effect scoring\n"
    "  7. tool_literature_search_for_motifs — PubMed / JASPAR / ENCODE evidence\n"
    "  8. tool_plot_creseq — publication figures\n"
    "Run tools in that order when the user asks to run the full pipeline."
)


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str] = field(default_factory=list)


def _server_params():
    from mcp import StdioServerParameters
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "creseq_mcp.server"],
        env={
            **os.environ,
            "CRESEQ_UPLOAD_DIR": str(UPLOAD_DIR),
            "CRESEQ_ASSOC_DIR":  str(ASSOC_DIR),
        },
    )


async def _run(messages: list[dict], api_key: str) -> AgentResponse:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in tools_list.tools
            ]

            client = anthropic.Anthropic(api_key=api_key)
            current = list(messages)
            tools_called: list[str] = []

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=_SYSTEM_PROMPT,
                    messages=current,
                    tools=tools,
                )

                if response.stop_reason != "tool_use":
                    text = next(
                        (b.text for b in response.content if hasattr(b, "text")), ""
                    )
                    return AgentResponse(text=text, tools_called=tools_called)

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    tools_called.append(block.name)
                    try:
                        result = await session.call_tool(block.name, block.input or {})
                        content = (
                            result.content[0].text
                            if result.content
                            else "{}"
                        )
                    except Exception as exc:
                        content = f"Tool error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    })

                current = current + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user",      "content": tool_results},
                ]


class ClaudeQCAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.history: list[dict] = []

    def send_message(self, prompt: str) -> AgentResponse:
        self.history.append({"role": "user", "content": prompt})
        response = asyncio.run(_run(self.history, self.api_key))
        self.history.append({"role": "assistant", "content": response.text})
        return response

    def reset(self) -> None:
        self.history = []


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
