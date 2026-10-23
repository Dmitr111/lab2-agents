"""Агент-исследователь: RAG по коллекции статей в Qdrant."""
import asyncio
import os
import re

from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "articles_lab2"
TOP_K = 5
MIN_SIMILARITY = 0.45  # статьи дальше этого порога считаем не относящимися к вопросу
NO_DATA = ("В базе пока нет статей по этой теме. Чтобы они появились, попросите их собрать, "
           "например: «собери статьи про ...».")

SYSTEM_PROMPT = """Ты научный ассистент. Отвечаешь на вопросы по статьям из базы знаний, собранной с КиберЛенинки.

Правила:
- Отвечай только по фрагментам статей, которые даны в сообщении. Не придумывай статьи, авторов и цифры.
- Фрагменты пронумерованы. После каждого утверждения ставь в квадратных скобках номер фрагмента, который его подтверждает. Пример: «Блокчейн применяют для хранения медицинских карт [2], а также для учёта поставок лекарств [1, 4].»
- Поиск всегда возвращает ближайшие фрагменты, даже если они не по теме. Используй только те, что действительно относятся к вопросу.
- Если ни один фрагмент не относится к вопросу или нужных сведений в них нет, прямо так и скажи, не ставь ссылок и укажи found = false. Если ответ во фрагментах есть, укажи found = true.
- Пиши только о том, что есть во фрагментах по вопросу. Не упоминай фрагменты, которые не используешь, и не пиши слово «фрагмент» в ответе.
- Список источников в конце не пиши: его добавит система.
- Если переданы замечания рецензента, исправь ответ с их учётом и выдай его целиком заново.

Тон деловой, на «вы», без эмодзи. Объём до 200 слов, а если просят подробнее — до 400."""


class Answer(BaseModel):
    found: bool = Field(description="true — во фрагментах есть ответ на вопрос, false — нужных сведений нет")
    text: str = Field(description="Текст ответа со ссылками [n] на фрагменты")


llm = ChatOllama(model="qwen3:8b", base_url=OLLAMA_URL, temperature=0, reasoning=False, num_ctx=8192)
writer = llm.with_structured_output(Answer, method="json_schema")
embeddings = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_URL)
qdrant = QdrantClient(url=QDRANT_URL)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Векторный поиск: k статей, ближайших к запросу."""
    if not qdrant.collection_exists(COLLECTION):  # статьи ещё ни разу не собирали
        return []
    vector = embeddings.embed_query(query)
    hits = qdrant.query_points(COLLECTION, query=vector, limit=k, with_payload=True).points
    return [{**hit.payload, "similarity": round(hit.score, 3)} for hit in hits]


def format_fragments(fragments: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {f['title']} ({f['year']})\n{f['text']}" for i, f in enumerate(fragments, 1))


def attach_sources(text: str, fragments: list[dict]) -> tuple[str, list[dict]]:
    """Перенумеровывает ссылки по порядку упоминания и добавляет список источников из метаданных."""
    order = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        for n in map(int, re.split(r"\s*,\s*", group)):
            if 1 <= n <= len(fragments) and n not in order:
                order.append(n)
    new_number = {old: new for new, old in enumerate(order, 1)}

    def renumber(match):
        nums = sorted({new_number[n] for n in map(int, re.split(r"\s*,\s*", match.group(1))) if n in new_number})
        return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", renumber, text)
    text = re.sub(r"\s+([.,;:])", r"\1", text).strip()  # убираем пробел, оставшийся от удалённой ссылки
    if not order:
        return text, []
    sources = [fragments[n - 1] for n in order]
    lines = [f"{i}. {s['title']} ({s['year']}) — {s['url']}" for i, s in enumerate(sources, 1)]
    return text + "\n\nИсточники:\n" + "\n".join(lines), sources


async def research(question: str, query: str | None = None, draft: str = "", feedback: str = "",
                   exclude: list[str] | None = None) -> dict:
    """Ищет фрагменты в базе и пишет ответ со ссылками.

    query — поисковый запрос, если он отличается от вопроса (например, для уточняющего вопроса).
    draft и feedback — прошлый ответ и замечания рецензента при доработке.
    exclude — ссылки на статьи, которые рецензент признал неподходящими: при доработке их не показываем модели.
    """
    fragments = await asyncio.to_thread(retrieve, query or question)
    fragments = [f for f in fragments if f["similarity"] >= MIN_SIMILARITY and f["url"] not in (exclude or [])]
    if not fragments:  # в базе нет ничего близкого: модель даже не вызываем
        return {"answer": NO_DATA, "found": False, "sources": [], "fragments": []}

    task = f"Фрагменты статей:\n\n{format_fragments(fragments)}\n\nВопрос: {question}"
    if feedback:
        task += f"\n\nТвой прошлый ответ:\n{draft}\n\nЗамечания рецензента:\n{feedback}"
    reply = await writer.ainvoke([("system", SYSTEM_PROMPT), ("human", task)])
    text, sources = attach_sources(reply.text, fragments)
    return {"answer": text, "found": reply.found, "sources": sources, "fragments": fragments}


async def main():
    while True:
        q = input("\nВопрос (пустая строка — выход): ").strip()
        if not q:
            break
        result = await research(q)
        print("\nНайдено:", *[f"  {f['similarity']}  {f['title']}" for f in result["fragments"]], sep="\n")
        print("\n" + result["answer"])


if __name__ == "__main__":
    asyncio.run(main())
