"""Проверка MCP-сервера: список инструментов и два тестовых вызова."""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


def as_text(result):
    """Инструмент MCP возвращает список блоков, берём из них текст."""
    if isinstance(result, list):
        return "\n".join(block.get("text", "") for block in result)
    return str(result)


async def main():
    client = MultiServerMCPClient({
        "cyberleninka": {"url": "http://localhost:8001/mcp", "transport": "streamable_http"},
    })
    tools = {t.name: t for t in await client.get_tools()}
    print("Инструменты:", list(tools))

    found = await tools["search_articles"].ainvoke({"query": "ИИ-агенты", "size": 3})
    print("\nПоиск:\n" + as_text(found))

    url = input("\nВставьте ссылку на одну из статей: ").strip()
    article = await tools["get_article"].ainvoke({"url": url})
    print("\nСтатья:\n" + as_text(article))


asyncio.run(main())
