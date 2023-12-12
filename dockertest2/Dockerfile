FROM python:3.8-slim
LABEL maintainer="Aucke Bos"


# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false 
COPY ./pyproject.toml /
RUN poetry install --no-interaction --no-ansi --only-root

# RUN pip install yarl==1.9.2