"""Mass model for the OGBar bar: a massless shell carrying two point masses.

The bar's shell (the 4 x 4 x 24 cm box) has no mass. Two point masses sit on
its long axis (local x) at x = -d (W_L) and x = +d (W_R), with
W_L + W_R = total_mass always. Only the split changes between episodes, so the
centre of mass moves along the bar while the total mass stays fixed:

    c = (W_R - W_L) * d / total_mass        (centre-of-mass offset along local x)

Each point mass is modelled as a solid ball of radius r (inside the bar), not a
true point: MuJoCo needs strictly positive rotational inertia on every axis, and
two true points on the centreline give zero inertia about the long axis (and,
at an 80/0 split, zero inertia about the centre of mass).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarMass:
    w_left: float          # kg, point mass at local x = -d
    w_right: float         # kg, point mass at local x = +d
    d: float               # m, distance of each point mass from the bar centre
    r: float               # m, radius of the ball each point mass is modelled as

    @property
    def total(self) -> float:
        return self.w_left + self.w_right

    @property
    def com_offset(self) -> float:
        """c: centre of mass along the bar's local x axis, measured from the geometric centre."""
        return (self.w_right - self.w_left) * self.d / self.total

    @property
    def inertia_diag(self) -> np.ndarray:
        """Principal moments (Ixx, Iyy, Izz) about the centre of mass, in the bar frame.

        Both masses lie on the local x axis, so the bar frame is already the principal frame.
        """
        c = self.com_offset
        ball = 0.4 * self.total * self.r**2   # (2/5) m r^2, summed over both balls
        ixx = ball
        iyy = self.w_left * (self.d + c) ** 2 + self.w_right * (self.d - c) ** 2 + ball
        return np.array([ixx, iyy, iyy])


def sample_w_right(cfg: dict, rng: np.random.Generator) -> float:
    """Draw W_R in kg. cfg is the `mass` block of the data config.

    Both end masses are strictly positive: W_R ~ U[w_min, total - w_min], so W_L = total - W_R >= w_min too.
    """
    total, w_min = cfg['total_mass'], cfg['w_min']
    if cfg['split'] == 'uniform':
        return float(rng.uniform(w_min, total - w_min))
    if cfg['split'] == 'discrete':
        return float(rng.choice(np.asarray(cfg['w_right_values'], dtype=np.float64)))
    raise ValueError(f"unknown mass.split: {cfg['split']}")


def make_mass(cfg: dict, w_right: float) -> BarMass:
    """Build the mass model for a given W_R (kg); W_L = total_mass - W_R."""
    total, w_min = cfg['total_mass'], cfg['w_min']
    if not w_min - 1e-12 <= w_right <= total - w_min + 1e-12:
        raise ValueError(f'W_R={w_right} outside [{w_min}, {total - w_min}]: both end masses must be >= w_min')
    return BarMass(w_left=total - w_right, w_right=w_right, d=cfg['offset_d'], r=cfg['ball_radius'])
