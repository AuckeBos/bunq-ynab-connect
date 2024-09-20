from typing import TypeVar

from pydantic import BaseModel
from sqlmodel import SQLModel

T = TypeVar("T", bound="Table")


class Table(SQLModel):
    """Base class for schemas.

    An implementation should be named MyTableSchema, and
    a corresponding Sq;<pde; should be named MyTable.

    The schema exists such that we can instantiate a class by Json data
    (from a client). The schema is converted into the table using
    the to_sqlmodel method. The SqlModel can then be saved to the database.

    We should only use the schema when instantiating a class from Json data,
    In all other cases, we should use the SqlModel.

    """

    @classmethod
    def schema(cls) -> type[BaseModel]:
        return getattr(__import__(cls.__module__), f"{cls.__name__}Schema")

    @classmethod
    def from_dict(cls: type[T], dict_: dict) -> type[T]:
        schema: BaseModel = cls.schema().model_validate(dict_)
        return cls.model_validate(schema)
