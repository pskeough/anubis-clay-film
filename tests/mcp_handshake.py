"""Start the MCP server over stdio, list its tools, and call the read-only ones.

    python tests/mcp_handshake.py
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clay_film import schema  # noqa: E402


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=["-m", "clay_film.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("tools:", tools)
            g = await s.call_tool("guide", {"location": "lakeside-pine-path"})
            print("guide chars:", len(g.content[0].text))
            ok = await s.call_tool("validate_plan", {"plan_json": json.dumps(schema.example())})
            print("validate example:", ok.content[0].text)
            bad = schema.example()
            bad["shots"][0]["camera"]["lens"] = 40
            er = await s.call_tool("validate_plan", {"plan_json": json.dumps(bad)})
            print("validate broken:", er.content[0].text)
            loc = await s.call_tool("location", {"location_id": "dorm-room"})
            print("dorm-room marks:", json.loads(loc.content[0].text)["marks"])
    return 0


sys.exit(asyncio.run(main()))
