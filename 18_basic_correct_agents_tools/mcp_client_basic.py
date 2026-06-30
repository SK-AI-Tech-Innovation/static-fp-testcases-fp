# ACE-FP-EXPECT: clean
# CATEGORY: 18_basic_correct_agents_tools
# SOURCE: modelcontextprotocol Python SDK (mcp.ClientSession over stdio transport)
# WHY-CORRECT: Standard MCP client lifecycle: open the stdio transport, create a
#   ClientSession, initialize() the handshake, list_tools() to discover, then
#   call_tool() with a named tool and typed arguments. Context managers ensure
#   the transport and session are closed. This is the documented MCP client flow.
# EXPECTED-WRONG: engine may suggest "initialize the session", "discover tools
#   before calling", or "close the connection" — all already handled.
# CORRECT-VERDICT: no findings
"""A correct MCP client: connect to a stdio server, discover and call a tool."""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command="python",
    args=["-m", "my_mcp_server"],
)


async def call_echo(message: str) -> str:
    """Connect to the MCP server, call its 'echo' tool, and return the text."""
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if "echo" not in names:
                raise RuntimeError("server does not expose an 'echo' tool")

            result = await session.call_tool("echo", arguments={"message": message})
            return "".join(
                block.text for block in result.content if block.type == "text"
            )


if __name__ == "__main__":
    print(asyncio.run(call_echo("hello from mcp")))
