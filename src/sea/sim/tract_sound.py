"""The talking robot's sound, simulated in time (D-121): air pressure and flow stepped along the tract.

The tract becomes M cells on a staggered grid, each posture's cells its own length so its tube keeps its own length.  A
cell holds air (compliance, from the integral of area); a face between two cells passes it (inertance, from the integral
of 1/area between their centres), so a narrowing shorter than a cell keeps its effect.  Then
    C_i dp_i/dt = U_i - U_{i+1},    L_f dU_f/dt = p_up - p_down - R_f(U_f) U_f + p_noise,f
At the lips, the radiation load of a piston in a baffle, resistance and inertance in parallel, stepped implicitly.  At the
glottis either a flow pulse, whose response gives the formants, or lungs behind an open glottis.

With flow, a face loses what a real jet loses: Borda-Carnot across an expansion and Poiseuille along a slit.  Where a jet
leaves a narrowing above the critical Reynolds number, a noise pressure source sits at its exit (Flanagan and Cherry 1969).
"""

from __future__ import annotations

import numpy as np

C, RHO = 35000.0, 1.14e-3  # cm/s, g/cm3
NU = 0.15  # cm2/s, air's kinematic viscosity
M_PER_CM = 1 / 0.6
CFL = 0.7
BETA = 2 * 0.0078 * C  # 1/s: the wall and viscous loss, at 1 kHz
PS = 8000.0  # dyn/cm2: 800 Pa of lung pressure
A_GLOTTIS = 0.25  # cm2, the glottis held open
RE_C = 1800.0  # a jet turns turbulent above this
NOISE_GAIN = 2e-6  # dyn/cm2 per unit of Re^2 - Re_c^2
A_SEAL = 2e-3  # cm2: a section below this passes no air


class Tube:
    """Postures (B sections of areas and lengths, glottis first) as a batch of M cells."""

    def __init__(self, areas: np.ndarray, lengths: np.ndarray):
        areas, lengths = np.atleast_2d(areas), np.atleast_2d(lengths)
        b = areas.shape[0]
        total = lengths.sum(1)
        m = int(np.ceil(total.max() * M_PER_CM))
        self.dx = total / m
        self.dt = CFL * self.dx.min() / C
        a = np.maximum(areas, A_SEAL)
        edges = np.c_[np.zeros(b), np.cumsum(lengths, 1)]
        vol = np.c_[np.zeros(b), np.cumsum(a * lengths, 1)]
        inv = np.c_[np.zeros(b), np.cumsum(lengths / a, 1)]
        xs = (np.arange(2 * m + 1) / (2 * m))[None] * total[:, None]  # cell edges and centres
        v = np.stack([np.interp(xs[i], edges[i], vol[i]) for i in range(b)])
        w = np.stack([np.interp(xs[i], edges[i], inv[i]) for i in range(b)])
        self.c = (v[:, 2::2] - v[:, :-2:2]) / (RHO * C**2)  # each cell's compliance
        self.l = RHO * (w[:, 3::2] - w[:, 1:-2:2])  # each inner face's inertance, centre to centre
        self.area = (v[:, 2::2] - v[:, :-2:2]) / self.dx[:, None]  # each cell's mean area
        self.lip = a[:, -1]
        self.l_lip = RHO * (w[:, -1] - w[:, -2])
        self.b, self.m = b, m
        # the narrowest area a face's span crosses, for what a jet there does
        idx = [np.searchsorted(edges[i], xs[i, 1:-2:2]) for i in range(b)]
        idx2 = [np.searchsorted(edges[i], xs[i, 3::2]) for i in range(b)]
        self.throat = np.empty((b, m - 1))
        for i in range(b):
            for f in range(m - 1):
                lo, hi = max(idx[i][f] - 1, 0), min(idx2[i][f], a.shape[1])
                self.throat[i, f] = a[i, lo:max(hi, lo + 1)].min()
        self.seal = self.throat <= A_SEAL * 1.0001

    def load(self):
        return _Lips(self)


class _Lips:
    """Radiation: the last half cell, then a resistance and an inertance in parallel, implicit together."""

    def __init__(self, t: Tube):
        self.r = 128 * RHO * C / (9 * np.pi**2 * t.lip)
        self.lr = 8 * RHO * np.sqrt(t.lip / np.pi) / (3 * np.pi * t.lip)
        self.k = t.dt / t.l_lip
        self.q = t.dt / self.lr
        self.u = np.zeros(t.b)
        self.ul = np.zeros(t.b)

    def step(self, p: np.ndarray) -> np.ndarray:
        pl = self.r * (self.u - self.ul + self.k * p) / (1 + self.r * (self.k + self.q))
        self.u = self.u + self.k * (p - pl)
        self.ul = self.ul + self.q * pl
        return self.u


def impulse(t: Tube, seconds: float = 0.04) -> np.ndarray:
    """The flow at the lips (B, T) after a short flow pulse at the glottis."""
    n = int(seconds / t.dt)
    kp, ku = t.dt / t.c, t.dt / t.l
    open_ = ~t.seal
    damp = 1 / (1 + BETA * t.dt)
    p, u = np.zeros((t.b, t.m)), np.zeros((t.b, t.m - 1))
    src = np.exp(-0.5 * ((np.arange(n) * t.dt - 3e-4) / 5e-5) ** 2)
    lips = t.load()
    out = np.zeros((t.b, n))
    for i in range(n):
        u = damp * open_ * (u - ku * (p[:, 1:] - p[:, :-1]))
        ue = damp * lips.step(p[:, -1])
        lips.u = ue
        p[:, 0] -= kp[:, 0] * (u[:, 0] - src[i])
        p[:, 1:-1] -= kp[:, 1:-1] * (u[:, 1:] - u[:, :-1])
        p[:, -1] -= kp[:, -1] * (ue - u[:, -1])
        out[:, i] = ue
    return out


def spectrum(x: np.ndarray, dt: float, freq: np.ndarray, pad: int = 8) -> np.ndarray:
    n = x.shape[-1]
    w = np.fft.rfft(x, n=pad * n, axis=-1)
    fs = np.fft.rfftfreq(pad * n, dt)
    mag = 20 * np.log10(np.abs(w) + 1e-12)
    i = np.minimum(np.searchsorted(fs, freq), fs.size - 1)
    return mag[..., i]


def peaks(mag: np.ndarray, freq: np.ndarray, n: int = 3) -> np.ndarray:
    """The first ``n`` peaks of each row, Hz, (B, n); nan where fewer."""
    peak = (mag[:, 1:-1] > mag[:, :-2]) & (mag[:, 1:-1] >= mag[:, 2:])
    out = np.full((mag.shape[0], n), np.nan)
    rows, cols = np.nonzero(peak)
    cols = cols + 1
    rank = np.arange(rows.size) - np.searchsorted(rows, rows)
    keep = rank < n
    rows, cols, rank = rows[keep], cols[keep], rank[keep]
    y0, y1, y2 = mag[rows, cols - 1], mag[rows, cols], mag[rows, cols + 1]
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    out[rows, rank] = freq[cols] + off * (freq[1] - freq[0])
    return out


def formants(areas: np.ndarray, lengths: np.ndarray, n: int = 3, freq: np.ndarray | None = None) -> np.ndarray:
    freq = np.arange(40.0, 5000.0, 10.0) if freq is None else freq
    t = Tube(areas, lengths)
    return peaks(spectrum(impulse(t), t.dt, freq), freq, n)


def blow(t: Tube, seed: int = 0, seconds: float = 0.05) -> dict[str, np.ndarray]:
    """Lungs behind an open glottis.  The lips' flow (B, T) over the settled part, the steady flow and the strongest
    Reynolds number at a jet (B), and the pressure behind the narrowest place."""
    n = int(seconds / t.dt)
    start = n // 3
    rng = np.random.default_rng(seed)
    kp, ku = t.dt / t.c, t.dt / t.l
    open_ = ~t.seal
    damp = 1 / (1 + BETA * t.dt)
    p, u = np.zeros((t.b, t.m)), np.zeros((t.b, t.m - 1))
    lips = t.load()
    a_f = t.throat
    down = t.area[:, 1:]
    # the jet's exit area and where a forward flow expands
    bc = 0.5 * RHO * np.maximum(1 / a_f - 1 / np.maximum(down, a_f), 0.0) ** 2
    pois = 8 * np.pi * NU * RHO * t.dx[:, None] / a_f**2
    jet = down > 1.5 * a_f  # a jet leaves here
    h = np.sqrt(a_f)
    ug = np.zeros(t.b)
    out = np.zeros((t.b, n - start))
    flow, re_max, behind = np.zeros(t.b), np.zeros(t.b), np.zeros(t.b)
    rows, narrow = np.arange(t.b), np.argmin(a_f, 1)
    for i in range(n):
        re = np.abs(u) * h / (a_f * NU)
        src = jet * NOISE_GAIN * np.maximum(re**2 - RE_C**2, 0.0) * rng.standard_normal(u.shape)
        loss = 1 + t.dt * BETA + ku * (bc * np.abs(u) + pois)
        u = open_ * (u - ku * (p[:, 1:] - p[:, :-1] - src)) / loss
        ue = damp * lips.step(p[:, -1])
        lips.u = ue
        drop = PS - p[:, 0]
        ug = np.sign(drop) * A_GLOTTIS * np.sqrt(2 * np.abs(drop) / RHO)
        p[:, 0] -= kp[:, 0] * (u[:, 0] - ug)
        p[:, 1:-1] -= kp[:, 1:-1] * (u[:, 1:] - u[:, :-1])
        p[:, -1] -= kp[:, -1] * (ue - u[:, -1])
        if i >= start:
            out[:, i - start] = ue
            flow += ue / (n - start)
            behind += p[rows, narrow] / (n - start)
            re_max = np.maximum(re_max, (re * jet).max(1))
    return {"lips": out, "flow": flow, "re": re_max, "behind": behind}


def noise_peak(lips: np.ndarray, dt: float, freq: np.ndarray, segs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Peak (Hz) and power (dB) of the fluctuation of ``lips`` (B, T) by averaging spectra over segments."""
    x = lips - lips.mean(1, keepdims=True)
    n = x.shape[1] // segs
    win = np.hanning(n)
    mag = np.zeros((x.shape[0], freq.size))
    for s in range(segs):
        mag += 10 ** (spectrum(x[:, s * n : (s + 1) * n] * win, dt, freq, 4) / 10)
    db = 10 * np.log10(mag / segs + 1e-12)
    return freq[db.argmax(1)], db.max(1)


def hiss(t: Tube, seconds: float = 0.03, freq: np.ndarray | None = None) -> np.ndarray:
    """Where a jet's noise peaks (Hz, B): a pressure pulse at the exit of each posture's narrowest place, the tube's
    response at the lips.  White noise there has this spectrum as its expected power."""
    freq = np.arange(500.0, 10000.0, 25.0) if freq is None else freq
    n = int(seconds / t.dt)
    kp, ku = t.dt / t.c, t.dt / t.l
    open_ = ~t.seal
    damp = 1 / (1 + BETA * t.dt)
    p, u = np.zeros((t.b, t.m)), np.zeros((t.b, t.m - 1))
    rows = np.arange(t.b)
    exit_ = np.argmin(t.throat, 1)
    pulse = np.exp(-0.5 * ((np.arange(n) * t.dt - 1e-4) / 2e-5) ** 2)
    lips = t.load()
    out = np.zeros((t.b, n))
    for i in range(n):
        force = p[:, 1:] - p[:, :-1]
        force[rows, exit_] -= pulse[i]
        u = damp * open_ * (u - ku * force)
        ue = damp * lips.step(p[:, -1])
        lips.u = ue
        p[:, 0] -= kp[:, 0] * u[:, 0]
        p[:, 1:-1] -= kp[:, 1:-1] * (u[:, 1:] - u[:, :-1])
        p[:, -1] -= kp[:, -1] * (ue - u[:, -1])
        out[:, i] = ue
    mag = spectrum(out, t.dt, freq, 2)
    return freq[mag.argmax(1)]


def speak(areas: np.ndarray, lengths: np.ndarray, frame_s: float, seed: int = 0) -> dict[str, np.ndarray]:
    """Lungs behind an open glottis while the tract moves: areas (F, N) a frame every ``frame_s``.  The flow at the lips
    (T), the pressure behind the narrowest place of each frame (F), the time step."""
    t = Tube(areas, np.broadcast_to(lengths, areas.shape))
    n = int(areas.shape[0] * frame_s / t.dt)
    rng = np.random.default_rng(seed)
    p, u = np.zeros(t.m), np.zeros(t.m - 1)
    r = 128 * RHO * C / (9 * np.pi**2 * t.lip)
    lr = 8 * RHO * np.sqrt(t.lip / np.pi) / (3 * np.pi * t.lip)
    ue, ul = 0.0, 0.0
    out = np.zeros(n)
    behind = np.zeros(t.b)
    a_f, down = t.throat, t.area[:, 1:]
    bc = 0.5 * RHO * np.maximum(1 / a_f - 1 / np.maximum(down, a_f), 0.0) ** 2
    pois = 8 * np.pi * NU * RHO * t.dx[:, None] / a_f**2
    jet, h = down > 1.5 * a_f, np.sqrt(a_f)
    narrow = np.argmin(a_f, 1)
    kp, ku = t.dt / t.c, t.dt / t.l
    open_ = ~t.seal
    k_l, q = t.dt / t.l_lip, t.dt / lr
    for i in range(n):
        f = min(int(i * t.dt / frame_s), t.b - 1)
        re = np.abs(u) * h[f] / (a_f[f] * NU)
        src = jet[f] * NOISE_GAIN * np.maximum(re**2 - RE_C**2, 0.0) * rng.standard_normal(u.shape)
        loss = 1 + t.dt * BETA + ku[f] * (bc[f] * np.abs(u) + pois[f])
        u = open_[f] * (u - ku[f] * (p[1:] - p[:-1] - src)) / loss
        pl = r[f] * (ue - ul + k_l[f] * p[-1]) / (1 + r[f] * (k_l[f] + q[f]))
        ue, ul = (ue + k_l[f] * (p[-1] - pl)) / (1 + BETA * t.dt), ul + q[f] * pl
        drop = PS - p[0]
        ug = np.sign(drop) * A_GLOTTIS * np.sqrt(2 * abs(drop) / RHO)
        p[0] -= kp[f, 0] * (u[0] - ug)
        p[1:-1] -= kp[f, 1:-1] * (u[1:] - u[:-1])
        p[-1] -= kp[f, -1] * (ue - u[-1])
        out[i] = ue
        behind[f] += p[max(narrow[f], 0)] / max(int(frame_s / t.dt), 1)
    return {"lips": out, "behind": behind, "dt": np.array(t.dt)}
