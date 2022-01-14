from abc import abstractmethod, abstractproperty, ABCMeta


class AbstractState(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractproperty
    def board(self):
        pass
    