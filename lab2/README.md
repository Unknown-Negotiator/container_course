# ЛР2. Docker Compose для immunogen-сервиса

## Что разворачиваем
- `db` — PostgreSQL 14, хранит данные (volume `db_data`).
- `init-db` — одноразовый init, ждет `db` и создает таблицу `healthcheck`.
- `api` — приложение из ЛР1 (сборка из `../lab1/Dockerfile.good`), кеши антиберти в volume `cache_data`, слушает порт `${API_PORT}`.

Общая сеть: `labnet`. Явные имена контейнеров (`lab2-*`). Env вынесены в `.env`.

## Как запустить
```bash
cd container_course/lab2
docker compose up --build
```
Проверка:
```bash
curl localhost:${API_PORT}/health
curl -X POST localhost:${API_PORT}/predict -H "Content-Type: application/json" -d '{"heavy_fasta":">h1\nCARDRSTW\n"}'
```

## Ответы на вопросы
- Лимиты ресурсов: в `docker-compose.yml` для обычного `docker compose` можно задавать `mem_limit`, `cpus` (compose v2) или `deploy.resources` — но `deploy` применяется только в swarm mode. Для чисто локального compose надежнее использовать `mem_limit`/`cpus` (или запуск через swarm/stack, если нужен `deploy`).
- Запуск только одного сервиса: `docker compose up api` (или любое имя сервиса), другие не стартуют. Можно аналогично `docker compose run --rm init-db` для одноразовых задач.
