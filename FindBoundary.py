from typing import Any, Callable, Tuple, TypeAlias
import numpy as np

# Alias for hinting, points should be 1D np.ndarrays with two elements, x and y coordinates
Point: TypeAlias = np.ndarray
# Alias for norm function
norm = np.linalg.norm


class BoundaryTracer:
    def __init__(
        self,
        In0: Point,
        Out0: Point,
        isInside: Callable,
        StepWidth: float,
        StepLength: float,
        PairTol: float,
    ) -> None:
        self.StepWidth = StepWidth
        self.StepLength = StepLength
        self.PairTol = PairTol
        self.isInside = isInside  # Checks if a point is inside the set

        In0, Out0 = self.TightenPair(In0, Out0)
        self.In = [In0]  # List of points inside the set
        self.Out = [Out0]  # List of points outside the set
        # Initial direction is perpendicular to the pair (to the right when looking outside)
        self.Direction = Out0 - In0
        self.Direction = np.array(
            [self.Direction[1], -self.Direction[0]]
        ) / np.linalg.norm(self.Direction)

    def TakeStep(self, Tighten=True) -> None:
        StepAlongLine = self.StepLength * self.Direction
        StepOutside = (
            self.StepWidth * np.array([-self.Direction[1], self.Direction[0]]) / 2
        )

        Middle = (self.In[-1] + self.Out[-1]) / 2
        newIn = Middle + StepAlongLine - StepOutside
        if not self.isInside(newIn):
            # Use newIn as newOut
            newOut = newIn
            newIn = self.In[-1]
            # self.Direction = newOut - Middle
            # self.Direction = self.Direction / norm(self.Direction)
        else:
            newOut = Middle + StepAlongLine + StepOutside
            if self.isInside(newOut):
                # Use newOut as newIn
                newIn = newOut
                newOut = self.Out[-1]
                # self.Direction = newOut - Middle
                # self.Direction = self.Direction / norm(self.Direction)

        # Save new pair
        self.In.append(newIn)
        self.Out.append(newOut)
        if Tighten:
            self.TightenNewestPair()
            self.UpdateDirection()

    def UpdateDirection(self):
        oldM = (self.In[-2] + self.Out[-2]) / 2
        newM = (self.In[-1] + self.Out[-1]) / 2
        Diff = newM - oldM
        n = norm(Diff)
        k = 0.5
        if n != 0:
            self.Direction = self.Direction * k + (1 - k) * Diff / norm(Diff)

    def TightenNewestPair(self):
        # Tighten the newest pair
        self.In[-1], self.Out[-1] = self.TightenPair(self.In[-1], self.Out[-1])

    def TightenPair(self, In: Point, Out: Point) -> Tuple[Point, Point]:

        while np.linalg.norm(In - Out) > self.PairTol:
            Middle = (In + Out) / 2
            if self.isInside(Middle):
                In = Middle
            else:
                Out = Middle

        return (In, Out)

    def GetNewestMiddle(self) -> Point:
        return (self.In[-1] + self.Out[-1]) / 2
