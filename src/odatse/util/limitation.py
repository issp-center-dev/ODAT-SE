from abc import ABCMeta, abstractmethod

import numpy as np

from .read_matrix import read_matrix, read_vector


class LimitationBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, is_limitary: bool):
        self.is_limitary = is_limitary

    @abstractmethod
    def judge(self, x: np.ndarray) -> bool:
        raise NotImplementedError

class Unlimited(LimitationBase):
    def __init__(self):
        super().__init__(False)
    def judge(self, x: np.ndarray) -> bool:
        return True

class Inequality(LimitationBase):
    def __init__(self, a: np.ndarray, b: np.ndarray, is_limitary: bool):
        super().__init__(is_limitary)
        if self.is_limitary:
            self.a = np.array(a)
            self.b = np.array(b)
            self.minusb = -np.array(b)
            self.n_formula = a.shape[0]
            self.ndim = a.shape[1]

    def judge(self, x: np.ndarray) -> bool:
        if self.is_limitary:
            Ax_b = np.dot(self.a, x) + self.b
            judge_result = np.all(Ax_b > 0)
        else:
            judge_result = True
        return judge_result

    @classmethod
    def from_dict(cls, d):
        co_a: np.ndarray = read_matrix(d.get("co_a", []))
        co_b: np.ndarray = read_matrix(d.get("co_b", []))

        # is_set_co_a = (co_a.size > 0 and co_a.ndim == 2 and co_a.shape[1] == dimension)
        # is_set_co_b = (co_b.size > 0 and co_b.ndim == 2 and co_b.shape == (co_a.shape[0], 1))

        if co_a.size == 0:
            is_set_co_a = False
        else:
            if co_a.ndim == 2:
                is_set_co_a = True
            else:
                raise ValueError("co_a should be a matrix of size equal to number of constraints times dimension")

        if co_b.size == 0:
            is_set_co_b = False
        else:
            if co_b.ndim == 2 and co_b.shape == (co_a.shape[0], 1):
                is_set_co_b = True
            else:
                raise ValueError("co_b should be a column vector of size equal to number of constraints")

        if is_set_co_a and is_set_co_b:
            is_limitary = True
        elif (not is_set_co_a) and (not is_set_co_b):
            is_limitary = False
        else:
            msg = "ERROR: Both co_a and co_b must be defined."
            raise ValueError(msg)

        return cls(co_a, co_b.reshape(-1), is_limitary)
