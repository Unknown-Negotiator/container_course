# ЛР3. Kubernetes: Postgres + Nextcloud (minikube)

## Шаги
```bash
cd container_course/lab3
# создать объекты
kubectl apply -f pg-configmap.yml
kubectl apply -f pg-secret.yml
kubectl apply -f pg-service.yml
kubectl apply -f pg-deployment.yml
kubectl apply -f nextcloud-configmap.yml
kubectl apply -f nextcloud-secret.yml
kubectl apply -f nextcloud-deployment.yml
```
Проверка:
```bash
kubectl get pods
kubectl get svc postgres-service
kubectl describe deployment postgres
kubectl describe deployment nextcloud
kubectl logs deploy/nextcloud
```

Подключение к UI (через NodePort):
```bash
kubectl expose deployment nextcloud --type=NodePort --port=80
minikube service nextcloud
```
Логин/пароль берутся из config/secret (`NEXTCLOUD_ADMIN_USER` и `NEXTCLOUD_ADMIN_PASSWORD`).

## Что сделано по заданию
- Postgres: ConfigMap (`pg-configmap.yml`), Secret (`pg-secret.yml` для user/password), Service NodePort (`pg-service.yml`), Deployment (`pg-deployment.yml`).
- Nextcloud: ConfigMap (`nextcloud-configmap.yml`), Secret (`nextcloud-secret.yml`), Deployment (`nextcloud-deployment.yml`) с readiness и liveness probe.
- Вынесено: все переменные Nextcloud в ConfigMap, секреты (БД и админ-пароль) в Secret.

## Ответы на вопросы
- Порядок применения манифестов имеет значение: сервис/конфиг/секрет должны появиться до Deployment, иначе под не стартует (не найдёт зависимости). В apply порядок можно задать явно, либо использовать `kubectl apply -f .` (kubectl сам повторит, но лучше явный порядок).
- Что будет при scale postgres до 0 и обратно: данные останутся только если есть persistent volume; без PVC под пересоздастся с пустыми данными, Nextcloud потребует инициализации заново. Даже с PVC под может быть недоступен временно, Nextcloud даст ошибки подключения к БД до восстановления.
