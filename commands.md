# Docker
## Compose up:
docker-compose -f docker/docker-compose.yml --env-file ./.docker-env -p bunqynab up

## Build:
docker build -t auckebos/bunq-ynab-connect -f docker/Dockerfile .

## BuildX:
- docker buildx create --name bunq-ynab-connect-builder
- docker buildx use bunq-ynab-connect-builder
- docker buildx build --platform linux/amd64,linux/arm64 -t auckebos/bunq-ynab-connect --push -f docker/Dockerfile .

# Dev
## Remove unused imports:
autoflake --in-place --remove-unused-variables --recursive .

## Compse up on dev
docker-compose -f docker/development.yml --env-file ./.docker-env -p bunqynab up