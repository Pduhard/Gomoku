from abc import abstractmethod, abstractproperty, ABCMeta


class AbstractState(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass
        # raise NotImplementedError

    @abstractproperty
    def state(self):
        pass
        # raise NotImplementedError
    