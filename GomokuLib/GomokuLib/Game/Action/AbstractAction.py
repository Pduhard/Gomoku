from abc import abstractmethod, abstractproperty, ABCMeta


class AbstractAction(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractproperty
    def action(self):
        pass