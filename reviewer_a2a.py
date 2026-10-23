"""Рецензент как отдельный сервис: принимает задания и отдаёт вердикт по протоколу A2A."""
import os

import uvicorn
from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from reviewer import review

PORT = int(os.getenv("A2A_PORT", "8002"))
PUBLIC_URL = os.getenv("A2A_PUBLIC_URL", f"http://localhost:{PORT}/")

# Карточка агента: по ней другие агенты узнают, кто это, что он умеет и куда слать задания
card = AgentCard(
    name="Рецензент научного ассистента",
    description="Проверяет ответ по найденным фрагментам статей и возвращает вердикт approved или revise с замечаниями.",
    version="1.0.0",
    supported_interfaces=[AgentInterface(url=PUBLIC_URL, protocol_binding="JSONRPC", protocol_version="1.0")],
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[AgentSkill(
        id="review_answer",
        name="Проверка ответа",
        description="Вход: question, answer, fragments. Выход: verdict и comments.",
        tags=["review", "rag", "fact-check"],
    )],
)


class ReviewerExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = get_data_parts(context.message.parts)[0]
        for f in task["fragments"]:  # JSON в A2A передаёт числа как дробные: 2025 → 2025.0
            if isinstance(f.get("year"), float):
                f["year"] = int(f["year"])
        print(f"Получено задание: «{task['question']}», фрагментов {len(task['fragments'])}")
        result = await review(task["question"], task["answer"], task["fragments"])
        print(f"Вердикт: {result['verdict']} {result['comments']}".rstrip())
        await event_queue.enqueue_event(new_data_message(result, context_id=context.context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Отмена не поддерживается")


handler = DefaultRequestHandler(agent_executor=ReviewerExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
