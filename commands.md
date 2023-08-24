# Docker
Compose up:
docker-compose --env-file .env -f docker/docker-compose.yml -p bunqynab  up

# Dev
Remove unused imports:
autoflake --in-place --remove-unused-variables --recursive .