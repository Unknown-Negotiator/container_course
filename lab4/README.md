# ЛР4. Own сервис в Kubernetes (minikube)

## Обзор
- Кастомный образ `immunogen-api:lab4` (Dockerfile.api здесь, сборка из `../lab1`).
- Deployment `immunogen-api` c init-контейнером `warm-cache`, emptyDir volume `/cache`, readiness/liveness пробы, ConfigMap+Secret.
- Deployment `notifier` (второй сервис), читает Secret, логирует heartbeat.
- Service `immunogen-api` (NodePort) для доступа к API.

## Сборка образа (локально, minikube)
```bash
cd container_course/lab4
eval $(minikube docker-env)   # чтобы образ был виден в minikube
docker build -f Dockerfile.api -t immunogen-api:lab4 ../lab1
```

## Применение манифестов
```bash
kubectl apply -f lab4-configmap.yml
kubectl apply -f lab4-secret.yml
kubectl apply -f api-deployment.yml
kubectl apply -f api-service.yml
kubectl apply -f worker-deployment.yml
```

Проверка:
```bash
kubectl get pods,svc
kubectl describe deploy immunogen-api
kubectl logs deploy/immunogen-api
kubectl logs deploy/notifier
```

Доступ к API (NodePort):
```bash
minikube service immunogen-api --url
curl $(minikube service immunogen-api --url)/health
```

## Соответствие требованиям
- ≥2 Deployment: `immunogen-api` + `notifier`.
- Кастомный образ из своего Dockerfile: `Dockerfile.api` -> `immunogen-api:lab4`.
- Init-контейнер: `warm-cache` в `immunogen-api`.
- Volume: `emptyDir` `cache-volume` монтируется в API.
- ConfigMap/Secret: `lab4-configmap.yml` и `lab4-secret.yml`, используются в обоих деплойментах.
- Service: `api-service.yml`.
- Пробы: readiness/liveness для `immunogen-api`.
- Лейблы: `app: ...` на всех объектах.
