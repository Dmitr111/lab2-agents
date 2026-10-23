"""Агент-рецензент: проверяет ответ исследователя по найденным фрагментам."""
import asyncio
import os
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

SYSTEM_PROMPT = """Ты рецензент научного ассистента. Проверяешь ответ другого агента, прежде чем он уйдёт пользователю.
У тебя нет собственных знаний о статьях, проверяй только по переданным фрагментам.

Проверь:
1. Каждое утверждение со ссылкой [n] подтверждается фрагментом с этим номером.
2. В ответе нет фактов, цифр, авторов и статей, которых нет во фрагментах.
3. Ответ отвечает именно на заданный вопрос.
4. Если ответ говорит, что нужных сведений нет, во фрагментах их действительно нет.

Если всё верно, поставь verdict "approved" и оставь comments пустым.
Если есть нарушения, поставь verdict "revise" и коротко перечисли в comments, что именно исправить.
В unsupported перечисли номера фрагментов, на которые ответ ссылается без оснований: фрагмент не подтверждает сказанное или не относится к вопросу."""


class Review(BaseModel):
    verdict: Literal["approved", "revise"] = Field(description="approved — ответ верен, revise — нужна доработка")
    comments: str = Field(description="Конкретные замечания для исследователя; пусто, если ответ одобрен")
    unsupported: list[int] = Field(default_factory=list,
                                   description="Номера фрагментов, на которые ответ ссылается без оснований")


llm = ChatOllama(model="qwen3:8b", base_url=OLLAMA_URL, temperature=0, reasoning=False, num_ctx=8192)
checker = llm.with_structured_output(Review, method="json_schema")


async def review(question: str, answer: str, fragments: list[dict]) -> dict:
    """Возвращает {"verdict": ..., "comments": ..., "unsupported": [...]}."""
    numbered = "\n\n".join(f"[{i}] {f['title']} ({f['year']})\n{f['text']}" for i, f in enumerate(fragments, 1))
    result = await checker.ainvoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Вопрос пользователя: {question}\n\nФрагменты:\n\n{numbered or 'нет'}\n\nОтвет для проверки:\n{answer}"),
    ])
    return result.model_dump()


async def main():
    from researcher import research  # для проверки гоняем связку исследователь → рецензент

    while True:
        q = input("\nВопрос (пустая строка — выход): ").strip()
        if not q:
            break
        result = await research(q)
        print("\nОтвет исследователя:\n" + result["answer"])
        if not result["fragments"]:
            print("\n(в базе ничего не найдено, рецензент не нужен)")
            continue
        # рецензент видит те же номера, что в ответе: источники, а если их нет — все фрагменты
        verdict = await review(q, result["answer"], result["sources"] or result["fragments"])
        print("\nРецензент:", verdict)


if __name__ == "__main__":
    asyncio.run(main())
