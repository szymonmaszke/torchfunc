import abc


class Base(abc.ABC):
    def __str__(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def __repr__(self) -> str:
        parameters = ", ".join(
            f"{key}={value}"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )
        return f"{self}({parameters})"
