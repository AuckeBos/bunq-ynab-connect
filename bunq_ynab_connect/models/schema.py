from pydantic import BaseModel
from sqlmodel import SQLModel


class Schema(BaseModel):
    """Base class for schemas.

    An implementation should be named MyTableSchema, and
    a corresponding Sq;<pde; should be named MyTable.

    The schema exists such that we can instantiate a class by Json data
    (from a client). The schema is converted into the table using
    the to_sqlmodel method. The SqlModel can then be saved to the database.

    We should only use the schema when instantiating a class from Json data,
    In all other cases, we should use the SqlModel.

    """

    def to_sqlmodel(self, cls: type[SQLModel]) -> SQLModel:
        return cls.model_validate(self)

    @classmethod
    def model_validate(cls, *args, **kwargs) -> SQLModel:
        pydantic_model = super().model_validate(*args, **kwargs)
        return cls.to_sqlmodel(pydantic_model)
