from abc import abstractmethod, abstractproperty, ABCMeta


class AbstractAction(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractproperty
    def action(self):
        pass

    # @abstractmethod
    # def __eq__(self, __o: object) -> bool:
    #     pass
