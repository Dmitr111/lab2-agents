"""Клиент A2A: оркестратор отправляет задание рецензенту и получает вердикт."""
import os

import httpx

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_part, new_message
from a2a.types import Role, SendMessageRequest

REVIEWER_URL = os.getenv("REVIEWER_URL", "http://localhost:8002")
TIMEOUT = 300  # секунд: модель рецензента отвечает долго, а по умолчанию httpx ждёт всего 5 секунд


async def review_remote(question: str, answer: str, fragments: list[dict]) -> dict:
    """То же, что reviewer.review, но через сеть по протоколу A2A."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        client = await create_client(REVIEWER_URL, ClientConfig(streaming=False, httpx_client=http))
        payload = {"question": question, "answer": answer, "fragments": fragments}
        request = SendMessageRequest(message=new_message([new_data_part(payload)], role=Role.ROLE_USER))
        async for response in client.send_message(request):
            if response.HasField("message"):
                return get_data_parts(response.message.parts)[0]
    raise RuntimeError("Рецензент не вернул ответ")
