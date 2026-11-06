"""What a force does to a printed part (D-124): three-dimensional linear elasticity on a prism.

A part is clamped at x = 0 on 20-node bricks, which don't lock in bending as 8-node bricks do.  It is linear, so its
response to a unit load is solved once per shape, and scaled by whatever force the simulator reads.  Stress at the root
is extrapolated along the outer fibre from the first sections past it: a clamp's corner is a singularity of the model, not
of a part.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

E = 1.808e9  # Pa, printed PLA, as sea.mech.members
NU = 0.35
_C = E / ((1 + NU) * (1 - 2 * NU)) * np.array(
    [[1 - NU, NU, NU, 0, 0, 0], [NU, 1 - NU, NU, 0, 0, 0], [NU, NU, 1 - NU, 0, 0, 0],
     [0, 0, 0, (1 - 2 * NU) / 2, 0, 0], [0, 0, 0, 0, (1 - 2 * NU) / 2, 0], [0, 0, 0, 0, 0, (1 - 2 * NU) / 2]])
# the 20 nodes in natural coordinates
_NAT = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                 if sum(c != 0 for c in (x, y, z)) >= 2], float)
_G = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W = np.array([5 / 9, 8 / 9, 5 / 9])
_GAUSS = [((a, b, c), wa * wb * wc) for a, wa in zip(_G, _W, strict=True) for b, wb in zip(_G, _W, strict=True) for c, wc in zip(_G, _W, strict=True)]


def _shape_grad(p: tuple[float, float, float]) -> np.ndarray:
    """dN/d(xi, eta, zeta) of the 20 shape functions at ``p``: (20, 3)."""
    x, y, z = p
    out = np.empty((20, 3))
    for i, (a, b, c) in enumerate(_NAT):
        if a and b and c:  # a corner
            f = (1 + a * x) * (1 + b * y) * (1 + c * z) / 8
            s = a * x + b * y + c * z - 2
            out[i] = (a * (1 + b * y) * (1 + c * z) / 8 * s + f * a, b * (1 + a * x) * (1 + c * z) / 8 * s + f * b,
                      c * (1 + a * x) * (1 + b * y) / 8 * s + f * c)
        elif not a:
            out[i] = (-2 * x * (1 + b * y) * (1 + c * z) / 4, (1 - x * x) * b * (1 + c * z) / 4, (1 - x * x) * (1 + b * y) * c / 4)
        elif not b:
            out[i] = (a * (1 - y * y) * (1 + c * z) / 4, -2 * y * (1 + a * x) * (1 + c * z) / 4, (1 + a * x) * (1 - y * y) * c / 4)
        else:
            out[i] = (a * (1 + b * y) * (1 - z * z) / 4, (1 + a * x) * b * (1 - z * z) / 4, -2 * z * (1 + a * x) * (1 + b * y) / 4)
    return out


_GRADS = [(_shape_grad(p), w) for p, w in _GAUSS]


def _b(grad: np.ndarray) -> np.ndarray:
    """Strain from the element's displacements, with ``grad`` dN/d(x, y, z): (6, 60)."""
    b = np.zeros((6, 60))
    b[0, 0::3], b[1, 1::3], b[2, 2::3] = grad[:, 0], grad[:, 1], grad[:, 2]
    b[3, 0::3], b[3, 1::3] = grad[:, 1], grad[:, 0]
    b[4, 1::3], b[4, 2::3] = grad[:, 2], grad[:, 1]
    b[5, 0::3], b[5, 2::3] = grad[:, 2], grad[:, 0]
    return b


class Prism:
    """A part along x of length ``length``, width ``w`` in y and depth ``h`` in z, ``tip`` of its depth at the far end."""

    def __init__(self, length: float, w: float, h: float, tip: float = 1.0, nx: int | None = None, nz: int = 2):
        self.length, self.w, self.h, self.tip = length, w, h, tip
        self.nx = nx or int(np.clip(np.ceil(1.5 * length / h), 6, 24))
        self.nz = nz
        ids: dict[tuple[int, int, int], int] = {}
        pts: list[tuple[float, float, float]] = []
        self.elems = []
        for ex in range(self.nx):
            for ez in range(nz):
                row = []
                for a, b, c in _NAT:
                    key = (2 * ex + 1 + int(a), int(b) + 1, 2 * ez + 1 + int(c))
                    if key not in ids:
                        ids[key] = len(pts)
                        x = key[0] / (2 * self.nx) * length
                        depth = h * (1 - (1 - tip) * x / length)
                        pts.append((x, (key[1] - 1) / 2 * w, (key[2] / (2 * nz) - 0.5) * depth))
                    row.append(ids[key])
                self.elems.append(row)
        self.pts = np.array(pts)
        self.n = 3 * len(pts)
        self._assemble()

    def _assemble(self) -> None:
        rows, cols, vals, grows, gcols, gvals = [], [], [], [], [], []
        for el in self.elems:
            xyz = self.pts[el]
            dof = np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in el])
            k, g = np.zeros((60, 60)), np.zeros((20, 20))
            for grad, wt in _GRADS:
                jac = grad.T @ xyz
                det = np.linalg.det(jac)
                dn = grad @ np.linalg.inv(jac).T
                b = _b(dn)
                k += b.T @ _C @ b * det * wt
                g += np.outer(dn[:, 0], dn[:, 0]) * det * wt / self._area_at(xyz, grad)  # a unit axial load's stress
            rows.append(np.repeat(dof, 60))
            cols.append(np.tile(dof, 60))
            vals.append(k.ravel())
            kg = np.zeros((60, 60))
            for axis in range(3):
                kg[axis::3, axis::3] = g
            grows.append(np.repeat(dof, 60))
            gcols.append(np.tile(dof, 60))
            gvals.append(kg.ravel())
        self.k = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        self.kg = sp.csc_matrix((np.concatenate(gvals), (np.concatenate(grows), np.concatenate(gcols))), shape=(self.n, self.n))

    def _area_at(self, xyz: np.ndarray, grad: np.ndarray) -> float:
        return self.w * (xyz[:, 2].max() - xyz[:, 2].min()) * self.nz

    def _free(self, fix: np.ndarray) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n), fix)

    def _clamped(self) -> np.ndarray:
        return np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in np.nonzero(self.pts[:, 0] < 1e-9 * self.length)[0]])

    def _solve(self, f: np.ndarray) -> np.ndarray:
        free = self._free(self._clamped())
        u = np.zeros(self.n)
        u[free] = spl.spsolve(self.k[free][:, free], f[free])
        return u

    def _tip(self, axis: int, at_z: float | None = None) -> np.ndarray:
        """A unit force on the far end along ``axis``, spread over its nodes as their tributary areas would."""
        ends = np.nonzero(self.pts[:, 0] > (1 - 1e-9) * self.length)[0]
        if at_z is not None:
            ends = ends[np.argsort(np.abs(self.pts[ends, 2] - at_z))[:3]]
        f = np.zeros(self.n)
        f[3 * ends + axis] = 1.0 / ends.size
        return f

    def _stress(self, u: np.ndarray, elem: int, xi: float, zeta: float) -> float:
        """Von Mises at a point on the outer fibre of element ``elem``."""
        el = self.elems[elem]
        grad = _shape_grad((xi, 0.0, zeta))
        dn = grad @ np.linalg.inv(grad.T @ self.pts[el]).T
        dof = np.ravel([[3 * i, 3 * i + 1, 3 * i + 2] for i in el])
        s = _C @ _b(dn) @ u[dof]
        return float(np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) + 3 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)))

    def _root_stress(self, u: np.ndarray) -> float:
        """The outer fibre, extrapolated to the root from one and two sections past it (the clamp's corner aside)."""
        top = self.nz - 1
        one = max(self._stress(u, top + self.nz, -1.0, 1.0), self._stress(u, self.nz, -1.0, -1.0))
        two = max(self._stress(u, top + 2 * self.nz, -1.0, 1.0), self._stress(u, 2 * self.nz, -1.0, -1.0))
        return max(2 * one - two, one)

    def bend(self) -> tuple[float, float]:
        """With a unit force across the far end, in z: the stress at the root (Pa) and the bend there (m)."""
        u = self._solve(self._tip(2))
        ends = np.nonzero(self.pts[:, 0] > (1 - 1e-9) * self.length)[0]
        return self._root_stress(u), float(u[3 * ends + 2].mean())

    def pull(self) -> float:
        """With a unit force along it: the stress at mid-length (Pa)."""
        u = self._solve(self._tip(0))
        return self._stress(u, (self.nx // 2) * self.nz, 0.0, 0.0)

    def buckle(self) -> float:
        """The load (N) along it at which this part, clamped at its root and free at its far end, buckles: the elastic and
        geometric stiffness' smallest eigenvalue.  Half a part pinned at both ends buckles just so."""
        free = self._free(self._clamped())
        val = spl.eigsh(self.kg[free][:, free], k=1, M=self.k[free][:, free], which="LA", return_eigenvectors=False)
        return float(1.0 / val[0])


def _r(x: float) -> float:
    return round(float(x), 4)  # to 0.1 mm: search shares its shapes


@lru_cache(maxsize=4096)
def _bend(length: float, w: float, h: float) -> tuple[float, float]:
    return Prism(length, w, h).bend()


@lru_cache(maxsize=4096)
def _buckle(length: float, w: float, h: float) -> float:
    weak, strong = min(w, h), max(w, h)  # z the thin way: it buckles across its thickness
    return Prism(length / 2, strong, weak, nx=int(np.clip(np.ceil(0.75 * length / weak), 8, 24))).buckle()


def bend(length: float, w: float, h: float) -> tuple[float, float]:
    """A part clamped at one end, ``h`` deep in the direction of the force: stress at the root per newton across its far
    end (Pa/N), and how far that end moves per newton (m/N)."""
    if length <= 1e-4:
        return 0.0, 0.0
    return _bend(_r(length), _r(w), _r(h))


def buckle(length: float, w: float, h: float) -> float:
    """The load (N) a part pinned at both ends buckles at."""
    return _buckle(_r(length), _r(w), _r(h)) if length > 1e-4 else float("inf")


def bend_near(length: float, w: float, h: float) -> tuple[float, float]:
    """``bend``, for a length that changes at every step: solved at the nearest millimetre, carried to this length as a
    cantilever's stress and bend go with its length."""
    if length <= 1e-4:
        return 0.0, 0.0
    near = max(round(length, 3), 1e-3)
    stress, move = _bend(near, _r(w), _r(h))
    return stress * length / near, move * (length / near) ** 3


def buckle_near(length: float, w: float, h: float) -> float:
    """``buckle``, for a length that changes at every step: solved at the nearest millimetre, carried as 1 / length^2."""
    if length <= 1e-4:
        return float("inf")
    near = max(round(length, 3), 1e-3)
    return _buckle(near, _r(w), _r(h)) * (near / length) ** 2
