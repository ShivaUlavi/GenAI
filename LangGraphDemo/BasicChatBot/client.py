from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "math" : {
                "command":"python",
                "args":["mathserver.py"], 
                "transport":"stdio",
            },
            "weather" : {
                "url" : "http://127.0.0.1:8000/mcp ",
                "transport":"streamable-http",
            }

            
        }
    )
    import os
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

    tools = await client.get_tools()
    model = ChatGroq(model="qwen/qwen3.8-27b")
    agent = create_react_agent(
    model,tools)

    math_respone = await agent.invoke({"messages":[{"role":"user", "content": "what is (3+5) * 12"}]})
    print("Math Response: ",math_respone['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
