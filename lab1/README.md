# ЛР1. Dockerfile для immunogen CLI/API

Приложение: CPU-only сервис для предсказания иммуногенности антител (AntiBERTy эмбеддинги + ансамбль LogisticRegression). Есть CLI и FastAPI.

## Структура
- `app/` — код (`immunogen_service/cli.py`, `api.py`, `inference.py`), `requirements.txt`
- `models/marks2021_ensemble.joblib` — сохраненный ансамбль
- `Dockerfile.good` — хороший образ
- `Dockerfile.bad` — намеренно плохой образ

## Как пользоваться
Сборка хорошего образа:
```bash
docker build -f lab1/Dockerfile.good -t immunogen-api:good lab1
```

Запуск API (с кешем весов вне контейнера):
```bash
mkdir -p cache
docker run --rm -p 8000:8000 -v $(pwd)/cache:/cache immunogen-api:good
# healthcheck
curl localhost:8000/health
# инференс: heavy-only
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"heavy_fasta": ">h1\nCARDRSTW\n"}'
```

Запуск CLI:
```bash
docker run --rm -v $(pwd)/data:/data immunogen-api:good \
  python -m immunogen_service.cli --heavy /data/heavy.fa --out /data/out.json
```
- Если передаётся только heavy, light-фичи забиваются нулями.
- AntiBERTy скачает веса при первом запуске и сохранит в `/cache`; оффлайн нужен заранее прогретый кеш.

## Плохие практики в `Dockerfile.bad`
1. `COPY . .` втаскивает весь репозиторий (много лишнего, включая кеши и секреты).
2. Нет очистки `apt` и cache pip → раздутый образ.
3. Запуск от root и установка лишних пакетов (vim, nano, curl, git, build-essential) без нужды.
Дополнительно: непинованные зависимости `torch torchvision torchaudio` → дрейф билдов и размер.

## Как исправлено в `Dockerfile.good`
- Узкая база `python:3.10-slim`, явный список пакетов, очистка `apt`.
- Пинованные зависимости, отдельный слой для `requirements.txt`.
- Кеши моделей вынесены в volume (`/cache`), не пихаем данные в образ.
- Не-root пользователь, `WORKDIR /app`, только нужные файлы копируются.

## Плохие практики использования контейнера (примерно)
- Хранить секреты (ключи/токены) внутри образа или bake-ить их в слой — утечка при любом пуше.
- Давать контейнеру полный root/`--privileged`/`--net=host` без необходимости — риск безопасности и утечка хоста.

## Когда лучше не использовать контейнеры
- Очень stateful нагрузки с тяжёлым диском/FS, где проще прямой сервис на bare metal (высокий IOPS, специализированные драйверы).
- Сценарии, требующие эксклюзивного доступа к железу/ядру (нестандартные kernel-модули, специализированные GPU-драйверы), где изоляция только мешает.
