"""Агент-сборщик: ищет статьи через MCP-сервер, оценивает их релевантность и пишет в Qdrant."""
import asyncio
import json
import os
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MCP_URL = os.getenv("MCP_URL", "http://localhost:8001/mcp")
COLLECTION = "articles_lab2"
THRESHOLD = 6  # статьи с оценкой ниже не сохраняем

SYSTEM_PROMPT = """Ты строгий научный референт. Оцени, насколько статья соответствует теме поиска, по шкале от 0 до 10.
9–10: статья целиком посвящена теме.
6–8: тема — одна из главных в статье.
3–5: тема затронута частично или как пример.
0–2: тема упоминается вскользь или статья о другом.
Не завышай оценку: большинство статей из поисковой выдачи заслуживают 3–7.
Напиши краткое резюме статьи на русском языке в 2–3 предложениях."""


class Assessment(BaseModel):
    score: int = Field(description="Релевантность теме поиска от 0 до 10")
    summary: str = Field(description="Краткое резюме статьи в 2–3 предложениях")


llm = ChatOllama(model="qwen3:8b", base_url=OLLAMA_URL, temperature=0, reasoning=False, num_ctx=8192)
judge = llm.with_structured_output(Assessment, method="json_schema")
embeddings = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_URL)
qdrant = QdrantClient(url=QDRANT_URL)


def parse(result) -> list:
    """Ответ инструмента MCP — список текстовых блоков с JSON внутри."""
    return [json.loads(block["text"]) for block in result if block.get("type") == "text"]


def document_text(card: dict) -> str:
    """Текст, по которому строится вектор статьи."""
    return (f"{card['title']} ({card['year']}). {card['journal']}\n"
            f"Ключевые слова: {card['keywords']}\n"
            f"Аннотация: {card['abstract']}\n"
            f"Резюме: {card['summary']}")


def ensure_collection():
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )


async def collect(topic: str, size: int = 10, log=print) -> dict:
    """Собирает статьи по теме и сохраняет релевантные в векторную базу."""
    client = MultiServerMCPClient({"cyberleninka": {"url": MCP_URL, "transport": "streamable_http"}})
    tools = {t.name: t for t in await client.get_tools()}

    found = parse(await tools["search_articles"].ainvoke({"query": topic, "size": size}))
    log(f"Найдено статей: {len(found)}")

    kept = []
    for i, article in enumerate(found, 1):
        try:
            page = parse(await tools["get_article"].ainvoke({"url": article["url"]}))[0]
        except Exception as e:
            log(f"[{i}/{len(found)}] не удалось загрузить страницу: {e}")
            continue

        card = {**article, "keywords": ", ".join(page["keywords"]), "abstract": page["abstract"], "topic": topic}
        result = await judge.ainvoke([
            ("system", SYSTEM_PROMPT),
            ("human", f"Тема поиска: {topic}\n\nНазвание: {card['title']}\n"
                      f"Ключевые слова: {card['keywords']}\nАннотация: {card['abstract']}"),
        ])
        log(f"[{i}/{len(found)}] {result.score}/10  {card['title'][:80]}")
        if result.score >= THRESHOLD:
            kept.append({**card, "score": result.score, "summary": result.summary})
        await asyncio.sleep(1)  # не нагружаем сайт

    if kept:
        ensure_collection()
        texts = [document_text(c) for c in kept]
        vectors = await embeddings.aembed_documents(texts)
        qdrant.upsert(COLLECTION, points=[
            # id из ссылки: повторный сбор той же статьи перезапишет её, а не создаст дубль
            models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["url"])), vector=v, payload={**c, "text": t})
            for c, t, v in zip(kept, texts, vectors)
        ])

    log(f"Сохранено в «{COLLECTION}»: {len(kept)} из {len(found)}")
    return {"topic": topic, "found": len(found), "saved": len(kept), "titles": [c["title"] for c in kept]}


if __name__ == "__main__":
    topic = input("Тема: ").strip()
    size = int(input("Сколько статей взять (по умолчанию 10): ").strip() or 10)
    asyncio.run(collect(topic, size))
 