import asyncio
import json
import os
from typing import Any, List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "llama3.2")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external tools and providing informative answers. Engage in a natural, friendly manner while using available tools for real-time information retrieval.

# Tools
{tools}

# Notes 
- Responses should be based on the latest available data.
- Maintain an engaging and friendly tone.
- Highlight the usefulness of tools in assisting users comprehensively."""

class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self):
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        result = await self.session.list_tools()
        
        # Debug: Let's see what the actual structure is
        print(f"Tools result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"Tuple length: {len(result)}")
            for i, item in enumerate(result):
                print(f"Item {i} type: {type(item)}")
                
        return result

    def call_tool(self, tool_name: str) -> Any:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable

async def agent_loop(query: str, tools: dict, messages: List[dict] = None):
    messages = messages or [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            tools="\n- ".join(
                [f"{t['name']}: {t['schema']['function']['description']}" for t in tools.values()]
            )
        )}
    ]
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[t["schema"] for t in tools.values()] if tools else None,
        max_tokens=4096,
        temperature=0,
    )

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            tool_result = await tools[tool_call.function.name]["callable"](**arguments)
            messages.extend([
                response.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": json.dumps(tool_result)},
            ])
        response = await client.chat.completions.create(model=MODEL, messages=messages)

    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content, messages

async def main():
    """
    Main function that sets up the MCP server, initializes tools, and runs the interactive loop.
    The server is run in a Docker container to ensure isolation and consistency.
    """
    try:
        # Configure Docker-based MCP server for SQLite
        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",  # Remove container after exit
                "-i",  # Interactive mode
                "-v",  # Mount volume
                "mcp-test:/mcp",  # Map local volume to container path
                "mcp/sqlite",  # Use SQLite MCP image
                "--db-path",
                "/mcp/test.db",  # Database file path inside container
            ],
            env=None,
        )

        # Start MCP client and create interactive session
        async with MCPClient(server_params) as mcp_client:
            try:
                # Get available database tools
                tools_data = await mcp_client.get_available_tools()
                
                # Process tools based on the debug output
                tools = {}
                
                # We'll adapt this part based on the debug output
                if isinstance(tools_data, tuple) and len(tools_data) >= 2:
                    tools_list = tools_data[1]  # Assuming the second element contains the tools
                    
                    # Debug: Let's see what's in the tools list
                    print(f"Tools list type: {type(tools_list)}")
                    if tools_list:
                        first_tool = tools_list[0] if tools_list else None
                        print(f"First tool type: {type(first_tool)}")
                        print(f"First tool dir: {dir(first_tool)}")
                    
                    # Now build the tools dictionary
                    for tool in tools_list:
                        try:
                            tools[tool.name] = {
                                "name": tool.name,
                                "callable": mcp_client.call_tool(tool.name),
                                "schema": {
                                    "type": "function",
                                    "function": {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "parameters": tool.inputSchema,
                                    },
                                },
                            }
                        except AttributeError as e:
                            print(f"Error with tool: {tool}")
                            print(f"Tool attributes: {dir(tool) if hasattr(tool, '__dir__') else 'No dir'}")
                            raise
                
                # Start interactive prompt loop for user queries
                messages = None
                while True:
                    try:
                        # Get user input and check for exit commands
                        user_input = input("\nEnter your prompt (or 'quit' to exit): ")
                        if user_input.lower() in ["quit", "exit", "q"]:
                            break

                        # Process the prompt and run agent loop
                        response, messages = await agent_loop(user_input, tools, messages)
                        print("\nResponse:", response)
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                    except Exception as e:
                        print(f"\nError occurred in prompt loop: {e}")
            except Exception as e:
                print(f"\nError setting up tools: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"\nError in main function: {e}")

if __name__ == "__main__":
    asyncio.run(main())
