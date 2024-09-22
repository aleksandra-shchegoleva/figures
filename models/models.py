from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import NoReturn, List, Tuple
import numpy as np
import seaborn as sns

class AbstractFactory(ABC):
    @abstractmethod
    def create_model(self):
        pass

class AbstractModel(ABC):
    @abstractmethod
    def set_parameters(self):
        pass

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    def plot(self):
        pass