import os
import logging
from typing import Optional, Literal, List

from openai import AsyncOpenAI

try:
    # Попробуем взять общий логгер из основного файла
    from tg_manager_bot_dynamic import logger  # type: ignore
except Exception:
    logger = logging.getLogger(__name__)


OpenAIModel = Literal[
    "gpt-5",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
]

DEFAULT_SYSTEM_PROMPT = ""
DEFAULT_USER_PROMPT = ""

_openai_client: Optional[AsyncOpenAI] = None
_cached_api_key: Optional[str] = None


def get_openai_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """
    Возвращает singleton AsyncOpenAI-клиент.
    Ключ берётся либо из параметра, либо из переменной окружения OPENAI_API_KEY.
    """
    global _openai_client, _cached_api_key

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key не найден. "
            "Установите переменную окружения OPENAI_API_KEY "
            "или передайте api_key в get_openai_client()."
        )

    if _openai_client is None or _cached_api_key != api_key:
        _openai_client = AsyncOpenAI(api_key=api_key)
        _cached_api_key = api_key
        logger.info("Создан новый AsyncOpenAI клиент.")

    return _openai_client


async def gpt(
    model: OpenAIModel,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Базовая обёртка над Chat Completions API.
    Возвращает текст одного ответа.
    """
    client = get_openai_client(api_key)

    messages = []
    o1_models = {"o1", "o1-mini"}

    # Для большинства моделей используем system_prompt как обычно
    if model not in o1_models:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    else:
        if system_prompt:
            logger.warning(
                "Для модели %s system_prompt будет проигнорирован "
                "из-за ограничений модели.",
                model,
            )

    messages.append({"role": "user", "content": user_prompt})

    kwargs = {
        "model": model,
        "messages": messages,
    }

    # Не все модели поддерживают temperature (например, o1, o1-mini, gpt-5)
    if model not in {"o1", "o1-mini", "gpt-5"} and temperature is not None:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.debug(
        "Отправка запроса к OpenAI: model=%s, len(messages)=%s",
        model,
        len(messages),
    )

    response = await client.chat.completions.create(**kwargs)

    if not response.choices:
        raise RuntimeError("Пустой ответ от OpenAI (choices=[])")

    msg = response.choices[0].message
    if not msg or msg.content is None:
        raise RuntimeError("Нет содержимого в первом ответе OpenAI.")

    text = msg.content
    logger.debug("Ответ OpenAI (обрезано до 200 символов): %r", text[:200])
    return text


async def gpt_multi(
    model: OpenAIModel,
    system_prompt: str,
    user_prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    n: int = 3,
) -> List[str]:
    """
    Обёртка над Chat Completions API.
    Возвращает список из n вариантов ответа.
    """
    client = get_openai_client(api_key)

    messages = []
    o1_models = {"o1", "o1-mini"}

    if model not in o1_models:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    else:
        if system_prompt:
            logger.warning(
                "Для модели %s system_prompt будет проигнорирован "
                "из-за ограничений модели.",
                model,
            )

    messages.append({"role": "user", "content": user_prompt})

    n = max(1, int(n or 1))

    kwargs = {
        "model": model,
        "messages": messages,
        "n": n,
    }

    # Не все модели поддерживают temperature (например, o1, o1-mini, gpt-5)
    if model not in {"o1", "o1-mini", "gpt-5"} and temperature is not None:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.debug(
        "Отправка multi-запроса к OpenAI: model=%s, len(messages)=%s, n=%s",
        model,
        len(messages),
        n,
    )

    response = await client.chat.completions.create(**kwargs)

    if not response.choices:
        raise RuntimeError("Пустой ответ от OpenAI (choices=[])")

    texts: List[str] = []
    for i, ch in enumerate(response.choices):
        msg = getattr(ch, "message", None)
        if not msg or msg.content is None:
            logger.warning("Пустой message в choice #%s", i)
            continue
        t = str(msg.content).strip()
        if not t:
            continue
        texts.append(t)

    if not texts:
        raise RuntimeError("Все варианты от OpenAI пустые.")

    logger.debug(
        "Multi-ответ OpenAI (первый обрезан до 200 символов): %r",
        texts[0][:200],
    )
    return texts


async def gpt_answer(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: Optional[str] = None,
    model: OpenAIModel = "gpt-5",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Упрощённый вызов gpt(): только текст промпта и опциональный system_prompt.
    """
    return await gpt(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def gpt_answer_variants(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: Optional[str] = None,
    model: OpenAIModel = "gpt-5",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    n: int = 3,
) -> List[str]:
    """
    То же, что gpt_answer, но возвращает сразу несколько вариантов ответа.
    """
    return await gpt_multi(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )
