import cv2
import matplotlib.pyplot as plt
import numpy as np
import typing


class Vector2D(object):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def x(self) -> int:
        return self.x

    @property
    def y(self) -> int:
        return self.y

    def __add__(self, value: typing.Self) -> typing.Self:
        return Vector2D(self.x + value.x, self.y + value.y)

class RTable(object):
    def __init__(self):
        self.content: dict[float, list[Vector2D]] = {}
    
    def exists(self, theta: float) -> bool:
        return self.content.get(theta)

    def add(self, theta: float, vector: Vector2D):
        if not self.exists(theta=theta):
            self.content[theta] = []
        self.content[theta].append(vector)

    def get(self, theta: float) -> list[Vector2D]:
        return self.content.get(theta)

class GHT(object):
    def __init__(self):
        pass

    def _get_gradients(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return None

    def _build_r_table(self, image: cv2.typing.MatLike) -> dict[float, list[tuple[int, int]]]:
        pass

    def detect(self, template: cv2.typing.MatLike, reference: cv2.typing.MatLike):
        pass

