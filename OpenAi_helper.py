import os
import logging
import re
from datetime import datetime
from typing import Optional, Literal, List, Sequence, Tuple

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


def _strip_quotes(text: str) -> str:
    """Удаляет кавычки из текста."""
    if not text:
        return text
    return "".join(ch for ch in text if ch not in {'"', "'"})


def _is_negative_reply(text: str) -> bool:
    """Проверяет, является ли ответ отрицанием "нет" без знаков препинания."""
    normalized = (text or "").lower().strip()
    for symbol in (".", "!", "?", '"', "'"):
        normalized = normalized.replace(symbol, "")
    return normalized == "нет"


async def generate_dating_ai_variants(
    text: str,
    *,
    history_lines: Sequence[str],
    history_texts: Sequence[str],
    profile: Optional[str] = None,
    api_key: Optional[str] = None,
    model: OpenAIModel = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    n: int = 3,
) -> List[str]:
    """Генерирует варианты ответов в стиле анкеты знакомств."""

    now = datetime.now()
    logger.info("Генерация AI-ответов для входящего сообщения: %s", text)

    sys_prompt = (
        "Ты, слегка туповатый и простодушный человек в чате, ЖЕНСКОГО пола. "
        "Не выдавай, что ты нейросеть или искусственный интеллект. Отвечай просто, "
        "коротко, иногда делай ошибки, как реальный человек."
        "Также, если собеседник пишет не на русском языке, дай понять, что ты не понимаешь, что он хочет.\n"
        "Не проявляй инициативу в разговоре, отвечай только на конкретные сообщения, не пытайся поддерживать разговор.\n"
        "Дайвинчик (ДВ, ЛДВ, Давинчик) — бот для знакомств, где создают анкеты, ставят лайки и общаются при взаимном лайке. "
        "Простой сервис для поиска совпадений и общения, в этом сервисе, ты оставляла анкету там и общаешься с своим потенциальным парнем\n"
        "Во всех остальных случаях веди себя как обычный, простой собеседник девушка. "
    )

    history_block = "\n".join(history_lines).strip()
    if history_block:
        history_block = f"{history_block}\n"

    prompt = (
        f"сейчас {now.day:02d} число, {now.month:02d} месяца, {now.year} года\n"
        f"\nТвое описание: \n{(profile or 'Описание профиля не указано.').strip()}\n\n"
        "История сообщений ниже\n'Я:'- Твои сообщения, 'Он:' -  сообщения собеседника \n"
        f"{history_block}"
        f"Последнее сообщение пользователя: {text}"
    )

    raw_variants = await gpt_multi(
        model=model,
        system_prompt=sys_prompt,
        user_prompt=prompt,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
    )

    history_values_repr = repr(list(history_texts))
    processed: List[str] = []

    for variant in raw_variants:
        candidate = (variant or "").strip()
        if not candidate:
            continue

        if _is_negative_reply(candidate):
            logger.debug("Обнаружен отрицательный ответ, выполняю переформулировку")
            rephrase_prompt = (
                "Отвечай только на русском языке.\n"
                "Придумай текст, который предполагает не знание темы, вот текст, тема которого должна быть определена "
                f"'{text}'"
                "Пример ответа:\"Я не поняла про что ты говоришь\"\n"
            )
            if history_values_repr and history_values_repr != "[]":
                rephrase_prompt += (
                    "Ответ не должен совпадать из каких либо сообщений из данных:\n"
                    f"{history_values_repr}"
                )

            candidate = await gpt(
                model=model,
                system_prompt="Переформулируй текст",
                user_prompt=rephrase_prompt,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            candidate = (candidate or "").strip()

        if not candidate:
            continue

        grammar_prompt = (
            "Добавь грамматические ошибки, но они должны быть неявными и разными по типу.\n "
            "Используй скобочку только, если это требуется, не нужно создавать скобочку,')'. \n"
            "Скобки ставятся только если действительно нужно пояснить или уточнить что-то в тексте, а не просто чтобы сделать предложение «живее» или «эмоциональнее». Если контекст не требует, скобки не нужны\n"
            "Если скобочка не нужна и это вопрос, поставь знак вопроса. \n"
            "Если в сообщении несколько предложений, все, кроме последнего, должны заканчиваться точкой. \n"
            "Если в сообщении одно предложение, точку не ставить.\n\n"
            "Когда ставить запятые:\n"
            "— перед союзами а, но, однако, зато\n"
            "— при однородных членах, если нет повторяющихся союзов \n"
            "— при вводных словах \n"
            "— при деепричастных оборотах \n"
            "— при причастных оборотах после определяемого слова \n"
            "— между частями сложносочинённого предложения\n"
            "— между главным и придаточным в сложноподчинённом предложении \n"
            "Когда не ставить запятые:\n"
            "— между подлежащим и сказуемым \n"
            "— между двумя сказуемыми без союза\n"
            "— при одиночном деепричастии, ставшем наречием \n"
            "— если причастный оборот стоит перед определяемым словом \n"
            "— при устойчивых выражениях — зависит от контекста\n"
            "Не используй заглавную букву как ошибку\n"
            f"ТЕКСТ:\n\"{candidate}\""
        )

        candidate = await gpt(
            model=model,
            system_prompt="добавь ошибки в текст не меняя его структуру",
            user_prompt=grammar_prompt,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        candidate = _strip_quotes((candidate or "").strip())

        if candidate:
            processed.append(candidate)

    return processed


async def recommend_dating_ai_variant(
    *,
    incoming_text: str,
    variants: Sequence[str],
    history_lines: Sequence[str],
    profile: Optional[str] = None,
    api_key: Optional[str] = None,
    model: OpenAIModel = "gpt-4o",
    temperature: float = 0.4,
    max_tokens: Optional[int] = 250,
) -> Tuple[Optional[int], Optional[str]]:
    """Определяет, какой из трёх вариантов ответа выглядит наиболее подходящим."""

    clean_variants = [str(v).strip() for v in variants if str(v).strip()]
    if len(clean_variants) < 1:
        return None, None

    history_block = "\n".join(history_lines).strip()
    if history_block:
        history_block = (
            "История переписки (от старых сообщений к новым):\n"
            f"{history_block}\n\n"
        )

    variants_lines = [f"{i}) {v}" for i, v in enumerate(clean_variants, start=1)]
    variants_block = "\n".join(variants_lines)

    profile_text = (profile or "Описание профиля не указано.").strip()

    user_prompt = (
        "Ты помогаешь оператору переписки девушки выбрать лучший ответ на сообщение потенциального парня.\n"
        f"Описание девушки: {profile_text}\n"
        f"Входящее сообщение от собеседника: {incoming_text}\n\n"
        f"{history_block}"
        "Варианты ответов девушки:\n"
        f"{variants_block}\n\n"
        "Рекомендации должны быть такие, чтобы максимально близиться с человеком, для того, чтобы получить его доверие максимально быстро, максимально скорее, более того, рассуждения нужны либо после одного большого предложения со смыслом, либо после пары предложений, ну тоже как бы со смыслом.\n"
        "Ответь на русском языке. Сформулируй рекомендацию в одном или двух предложениях. В первом предложении однозначно укажи номер рекомендуемого варианта в формате «Рекомендую вариант 2 …»."
    )

    recommendation_text = await gpt(
        model=model,
        system_prompt=(
            "Ты — эмпатичный помощник, который выбирает самый располагающий вариант ответа."
        ),
        user_prompt=user_prompt,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    recommendation_text = (recommendation_text or "").strip()
    if not recommendation_text:
        return None, None

    match = re.search(r"вариант\s*№?\s*([123])", recommendation_text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b([123])\b", recommendation_text)

    if match:
        idx = int(match.group(1)) - 1
    else:
        idx = None

    if idx is not None and (idx < 0 or idx >= len(clean_variants)):
        idx = None

    return idx, recommendation_text