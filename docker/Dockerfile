FROM python:3.8-slim
LABEL maintainer="Aucke Bos"

# Set user
ARG USERNAME=bunqynab
ARG UID=1000
ARG GID=$UID
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID bunqynab && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" bunqynab && \
    echo 'bunqynab ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN apt install -y gcc
RUN chmod -R 755 /home/$USERNAME

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false 
# Copy files
COPY --chown=bunqynab:bunqynab ./pyproject.toml ./poetry.lock ./docker/entrypoint.sh ./README.md /home/bunqynab/
RUN chmod +x /home/bunqynab/entrypoint.sh

# Install deps
WORKDIR /home/bunqynab
RUN poetry install --no-interaction --no-ansi --no-dev --no-root
# Install own package
COPY --chown=bunqynab:bunqynab ./bunq_ynab_connect /home/bunqynab/bunq_ynab_connect    
RUN poetry install --no-interaction --no-ansi --only-root

USER $USERNAME
ENV USERNAME=bunqynab
ENTRYPOINT ["/bin/sh", "-c", "/home/bunqynab/entrypoint.sh"]