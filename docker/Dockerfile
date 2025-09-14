ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bullseye
LABEL maintainer="Aucke Bos"

# Install uv by copying uv docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up user without password
ARG USERNAME=bunqynab
ARG UID=1000
ARG GID=$UID
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID $USERNAME && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" $USERNAME && \
    echo '$USERNAME ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers    

# Copy files
COPY --chown=$USERNAME:$USERNAME ./pyproject.toml ./requirements.lock ./docker/*.sh ./README.md /home/$USERNAME/
RUN chmod +x /home/$USERNAME/*.sh
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# Install deps
ENV USERNAME=$USERNAME
USER $USERNAME
WORKDIR /home/$USERNAME
RUN uv sync
ENV PATH="/home/$USERNAME/.venv/bin:$PATH"
# Install self
# COPY --chown=$USERNAME:$USERNAME ./bunq_ynab_connect /home/$USERNAME/bunq_ynab_connect
# RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -e .


CMD ["/bin/bash", "./entrypoint.sh"]

