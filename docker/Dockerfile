FROM python:3.11-slim-bullseye
LABEL maintainer="Aucke Bos"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Set user
ARG USERNAME=bunqynab
ARG UID=1000
ARG GID=$UID
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID bunqynab && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" bunqynab && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN apt-get install -y --no-install-recommends build-essential
RUN apt install -y gcc
RUN chmod -R 755 /home/$USERNAME

# Copy files
COPY --chown=$USERNAME:$USERNAME ./pyproject.toml ./requirements.lock ./docker/*.sh ./README.md /home/$USERNAME/
RUN chmod +x /home/$USERNAME/*.sh
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# Install deps
WORKDIR /home/$USERNAME
RUN uv sync

ENV PATH="/home/$USERNAME/.venv/bin:$PATH"

# Install self
# COPY --chown=$USERNAME:$USERNAME ./bunq_ynab_connect /home/$USERNAME/bunq_ynab_connect
# RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -e .

ENV USERNAME=$USERNAME
USER $USERNAME
CMD ["/bin/bash", "./entrypoint.sh"]

