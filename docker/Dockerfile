FROM python:3.8-slim
LABEL maintainer="Aucke Bos"

# Set user
ARG USERNAME=bunqynab
ARG UID=1000
ARG GID=$UID
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID ${USERNAME} && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} && \
    echo '${USERNAME} ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN apt install -y gcc
RUN chmod -R 755 /home/$USERNAME

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false 
# Copy files
COPY --chown=${USERNAME}:${USERNAME} ./pyproject.toml ./poetry.lock ./docker/entrypoint.sh ./README.md /home/${USERNAME}/
RUN chmod +x /home/${USERNAME}/entrypoint.sh

# Install deps
WORKDIR /home/${USERNAME}
RUN poetry install --no-interaction --no-ansi --no-dev --no-root
# Install own package
COPY --chown=${USERNAME}:${USERNAME} ./bunq_ynab_connect /home/${USERNAME}/bunq_ynab_connect    
RUN poetry install --no-interaction --no-ansi --only-root

USER $USERNAME
ENV USERNAME=${USERNAME}
ENTRYPOINT ["/bin/sh", "-c", "/home/${USERNAME}/entrypoint.sh"]