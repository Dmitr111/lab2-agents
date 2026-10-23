"""MCP-сервер «КиберЛенинка»: поиск научных статей и разбор страницы статьи."""
import html
import os
import re

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

BASE = os.getenv("CYBERLENINKA_URL", "https://cyberleninka.ru")
HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    "origin": "https://cyberleninka.ru",
    "referer": "https://cyberleninka.ru/search",
}

mcp = FastMCP("cyberleninka", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "8001")))


@mcp.tool()
async def search_articles(query: str, size: int = 10) -> list[dict]:
    """Ищет научные статьи на КиберЛенинке по теме.

    Возвращает список статей: название, ссылка, авторы, год, журнал.
    """
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        resp = await client.post(
            f"{BASE}/api/search",
            json={"mode": "articles", "q": query, "size": size, "from": 0},
        )
        resp.raise_for_status()

    articles = []
    for a in resp.json().get("articles", []):
        articles.append({
            "title": html.unescape(re.sub(r"<[^>]+>", "", a.get("name", ""))),  # убираем <b> и &quot;
            "url": "https://cyberleninka.ru" + a.get("link", ""),
            "authors": ", ".join(a.get("authors") or []),
            "year": a.get("year"),
            "journal": a.get("journal", ""),
        })
    return articles


@mcp.tool()
async def get_article(url: str) -> dict:
    """Загружает страницу статьи с КиберЛенинки и возвращает ключевые слова и полную аннотацию."""
    if not url.startswith("https://cyberleninka.ru/article/"):
        raise ValueError("Нужна ссылка вида https://cyberleninka.ru/article/...")

    page_url = url.replace("https://cyberleninka.ru", BASE, 1)
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        resp = await client.get(page_url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    keywords = [s.get_text(strip=True) for s in soup.select('i[itemprop="keywords"] span')]
    meta = soup.select_one('meta[name="description"]')
    abstract = meta.get("content", "").strip() if meta else ""
    return {"url": url, "keywords": keywords, "abstract": abstract}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    