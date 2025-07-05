# client.py
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    command: str = "mcp"
    args: list[str] = None
    env: dict[str, str] | None = None
    timeout: int = 30

    def __post_init__(self):
        if self.args is None:
            self.args = ["run", "server.py"]


class MCPClient:
    def __init__(self, config: ClientConfig = None):
        self.config = config or ClientConfig()
        self.server_params = StdioServerParameters(
            command=self.config.command, args=self.config.args, env=self.config.env
        )
        logger.info(f"Initialized MCP client with command: {self.config.command}")

    async def get_session(self):
        return stdio_client(self.server_params)

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Generic tool calling method."""
        try:
            async with await self.get_session() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        tool_name, arguments=arguments or {}
                    )
                    logger.info(f"Tool '{tool_name}' call completed")
                    return json.loads(result.content[0].text) if result.content else {}
        except Exception as e:
            logger.error(f"Tool '{tool_name}' call failed: {e}")
            return {"error": str(e)}

    async def list_available_tools(self):
        """List all available tools."""
        try:
            async with await self.get_session() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_names = [tool.name for tool in tools.tools]
                    logger.info(f"Found {len(tool_names)} tools")
                    return tool_names
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    # Tool-specific methods
    async def call_statistics(self, numbers: list[float]):
        return await self._call_tool("statistics", {"numbers": numbers})

    async def call_system_info(self):
        return await self._call_tool("get_system_info")

    async def call_analyze_file(self, file_path: str) -> dict[str, Any]:
        """Analyze a file."""
        return await self._call_tool("analyze_file", {"file_path": file_path})

    async def call_clear_cache(self) -> dict[str, Any]:
        """Clear analysis cache."""
        return await self._call_tool("clear_analysis_cache")

    async def call_search_in_file(
        self, file_path: str, search_term: str, case_sensitive: bool = False
    ):
        return await self._call_tool(
            "search_in_file",
            {
                "file_path": file_path,
                "search_term": search_term,
                "case_sensitive": case_sensitive,
            },
        )


async def main():
    import random

    client = MCPClient()
    tools = await client.list_available_tools()
    print(f"Available tools: {tools}")
    numbers = [random.randint(0, 1000) for _ in range(100)]
    stats_results = await client.call_statistics(numbers)
    print(f"Statistics: {json.dumps(stats_results, indent=2)}")

    system_info = await client.call_system_info()
    print(f"System info: {json.dumps(system_info, indent = 2)}")

    test_file = "/home/olande/Desktop/Hands on Large Language Models.pdf"
    if Path(test_file).exists():
        analysis_result = await client.call_analyze_file(test_file)
        print(f"File Analysis: {json.dumps(analysis_result, indent=2)}")

        search_result = await client.call_search_in_file(test_file, "model")
        print(f"Search results: {json.dumps(search_result, indent=2)}")

    cache_result = await client.call_clear_cache()
    print(f"Cache Clear: {json.dumps(cache_result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
