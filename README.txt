TG Manager Bot (Telethon) — static private SOCKS5 + multi-API
====================================================

1) Установка
-----------
На Windows достаточно запустить `run_bot.bat` — скрипт сам найдёт Python,
создаст виртуальное окружение и установит зависимости перед запуском.

Для ручной установки:
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

2) Настройка
------------
- Открой tg_manager_bot_dynamic.py и заполни:
  * API_KEYS  — до 5 пар {"api_id": ..., "api_hash": "..."}
  * BOT_TOKEN — токен от @BotFather
  * ADMIN_ID  — твой user id (например 8099997426)
  * PRIVATE_PROXY — оставь enabled=True и свои параметры статичного SOCKS5; или поставь False

3) Запуск
--------
python tg_manager_bot_dynamic.py

В Telegram открой своего бота:
/start -> Добавить аккаунт -> номер -> код (и 2FA при необходимости).

Логи пишутся в bot.log. Окно не закроется при ошибке — ждёт Enter.
