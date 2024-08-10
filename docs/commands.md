todo: clean
# Introduction
This page lists some commands often used during development. Some of these will be added as VSCode tasks.

# Docker
## Compose up
- Spin up on dev server: `docker-compose -f docker/development.yml --env-file ./.docker-env -p bunqynab up --build`

## Build
- Build: `docker build -t auckebos/bunq-ynab-connect -f docker/Dockerfile .`
- Build with BuildX: 
    - Create builder: `docker buildx create --name bunq-ynab-connect-builder`
    - Build: `docker buildx use bunq-ynab-connect-builder`
    - Build & push for both required architectures: `docker buildx build --platform linux/amd64,linux/arm64 -t auckebos/bunq-ynab-connect --push -f docker/Dockerfile .`
    - Build for one architecture: `docker buildx build --platform linux/arm64 -f docker/Dockerfile .`

# Development
- `[Untill fixed with ruff]` Remove unused imports: `autoflake --in-place --remove-unused-variables --recursive .`
- Delete old mlfow runs: `mlflow gc --backend-store-uri sqlite:////mlflow/mlflow.db --older-than 30d`. Delete runs manually first


