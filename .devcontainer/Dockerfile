ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bullseye
LABEL maintainer="Aucke Bos"

# Install uv by copying uv docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Set up user without password
ARG USERNAME=bunqynab
ARG UID=1000
ARG GID=$UID
RUN apt update && \
    addgroup --gid $GID $USERNAME && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" $USERNAME 

RUN chmod -R 755 /home/$USERNAME    

USER $USERNAME
ENTRYPOINT [ "tail", "-f", "/dev/null" ]
