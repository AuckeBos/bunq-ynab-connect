import sys
from logging import LoggerAdapter
from typing import TypeVar

from kink import di
from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, select

T = TypeVar("T", bound="Table")


class Table(SQLModel):
    """Base class for SqlModel tables.

    Should be used if we should be able to instantiate a class from Json data.
    Instantiation should be done using the from_dict method.

    Assumes a schema exists for the table. The schema should be a Pydantic
    BaseModel class. Schema is instantiated using model_validate. Then
    it is converted to the SqlModel using model_validate. The SqlModel
    is returned.

    This structure resolves an issue with SqlModel, where relationships are not
    instantiated when the class is created from JSON.

    """

    @classmethod
    def get_schema_class(cls) -> type[BaseModel]:
        try:
            return getattr(sys.modules[cls.__module__], f"{cls.__name__}Schema")
        except AttributeError as e:
            msg = f"Make sure a class  {cls.__name__}Schema exists in {cls.__module__}"
            di[LoggerAdapter].exception(msg)
            raise AttributeError(msg) from e

    @classmethod
    def from_dict(cls: type[T], dict_: dict) -> type[T]:
        schema: BaseModel = cls.get_schema_class().model_validate(dict_)
        return cls.model_validate(schema)

    @classmethod
    def insert(cls: type[T], items: list[T]) -> None:
        database: Engine = di["database"]
        with Session(database) as session:
            session.add_all(items)
            session.commit()

    @classmethod
    def upsert(cls: type[T], items: list[T]) -> None:
        database: Engine = di["database"]
        with Session(database) as session:
            statement = select(cls).where(cls.id in [item.id for item in items])
            existing_items = session.exec(statement).all()
            session.delete(existing_items)
            session.commit()
        cls.insert(items)

    @classmethod
    def overwrite(cls: type[T], items: list[T]) -> None:
        database: Engine = di["database"]
        with Session(database) as session:
            statement = select(cls)
            existing_items = session.exec(statement).all()
            for item in existing_items:
                session.delete(item)
            session.commit()
        cls.insert(items)
