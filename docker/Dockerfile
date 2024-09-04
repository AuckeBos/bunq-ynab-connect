FROM python:3.12-bookworm
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
RUN apt-get install -y --no-install-recommends build-essential
RUN apt install -y gcc
RUN chmod -R 755 /home/$USERNAME
# Copy files
COPY --chown=bunqynab:bunqynab ./pyproject.toml ./requirements.lock ./docker/*.sh ./README.md /home/bunqynab/
RUN chmod +x /home/bunqynab/*.sh
RUN chown -R bunqynab:bunqynab /home/bunqynab

# Install deps
WORKDIR /home/bunqynab
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r requirements.lock

# Install self
COPY --chown=bunqynab:bunqynab ./bunq_ynab_connect /home/bunqynab/bunq_ynab_connect
RUN pip install -e .

ENV USERNAME=bunqynab
USER $USERNAME
CMD ["/bin/bash", "./entrypoint.sh"]
