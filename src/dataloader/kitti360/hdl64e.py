"""KITTI-360 Velodyne HDL-64E laser geometry, recovered from the released scans.

The public ``.bin`` scans hold only float32 XYZI, but each return was produced
by one of the 64 lasers through the vendor calibration model (Velodyne driver:
``z = d*sin(delta_k) + V_k`` with horizontal range ``d*cos(delta_k)`` and a
horizontal offset). The returns of laser ``k`` therefore lie on a cone
around the sensor z axis, ``z = rho*tan(delta_k) + V_k`` with
``rho = hypot(x, y)``; ``V_k`` is the height where the laser's rays cross that
axis (the cone apex). The horizontal offset is +-26.0 mm, alternating in sign
between neighbouring lasers. In the stored points it only turns a return's
azimuth by asin(h/rho) (0.74 deg at 2 m). The points stay on the cones, so a
ray from the apex in the measured azimuth passes through the return.

The table below was fitted per laser on 722 scans from all nine Train/Val
drives. Lasers were first separated by scan order (each laser's sweep starts
at the forward direction, see ``scan_order_lasers``). The maximum residual was
3 micrometres, the float32 storage limit, and a fit on only the 40 LiDAR4D
targets agrees to 5e-7. ``assign_lasers`` uses the table directly, so it does
not depend on scan order.

Lasers are ordered top (+1.94 deg) to bottom (-25.00 deg). Lasers 0-31 form
the upper block (apex 0.198-0.209 m above the sensor origin) and 32-63 the
lower block (0.116-0.126 m), matching the HDL-64E's two emitter blocks.
"""
from __future__ import annotations

import numpy as np

KITTI360_HDL64E_ELEVATION_DEG = np.array([
    1.936700, 1.573966, 1.304757, 0.871566, 0.578810, 0.180617,
    -0.088762, -0.451829, -0.803150, -1.201239, -1.493880, -1.833245,
    -2.207566, -2.546633, -2.873838, -3.235882, -3.539334, -3.935853,
    -4.215524, -4.588103, -4.913792, -5.250778, -5.610601, -5.958395,
    -6.328889, -6.675746, -6.999036, -7.287312, -7.678770, -8.058025,
    -8.310470, -8.711414, -9.026018, -9.573515, -10.062491, -10.470730,
    -10.956945, -11.598997, -12.115005, -12.562095, -13.040989, -13.484813,
    -14.048301, -14.598064, -15.188734, -15.656734, -16.176634, -16.554010,
    -17.186821, -17.730370, -18.323430, -18.797076, -19.320236, -19.736371,
    -20.222575, -20.787690, -21.318126, -21.935510, -22.437605, -22.856577,
    -23.322386, -23.971016, -24.506606, -24.999202,
])
KITTI360_HDL64E_APEX_HEIGHT_M = np.array([
    0.209124, 0.208628, 0.208265, 0.207691, 0.207310, 0.206801,
    0.206462, 0.206012, 0.205585, 0.205110, 0.204768, 0.204376,
    0.203953, 0.203576, 0.203219, 0.202831, 0.202512, 0.202103,
    0.201819, 0.201448, 0.201130, 0.200808, 0.200470, 0.200150,
    0.199816, 0.199510, 0.199230, 0.198985, 0.198660, 0.198352,
    0.198151, 0.197838, 0.125838, 0.125416, 0.125047, 0.124743,
    0.124388, 0.123927, 0.123564, 0.123254, 0.122928, 0.122630,
    0.122258, 0.121902, 0.121526, 0.121233, 0.120912, 0.120682,
    0.120303, 0.119982, 0.119638, 0.119366, 0.119070, 0.118838,
    0.118569, 0.118260, 0.117973, 0.117643, 0.117377, 0.117156,
    0.116913, 0.116576, 0.116300, 0.116048,
])
# Emitter blocks as (first laser, end laser). Each block is rendered from one
# origin on the sensor z axis: the midrange of its apex heights, which bounds
# the apex mismatch by 5.6 mm (upper) and 4.9 mm (lower).
KITTI360_HDL64E_BLOCKS = ((0, 32), (32, 64))
KITTI360_HDL64E_BLOCK_ORIGIN_Z_M = tuple(
    float(KITTI360_HDL64E_APEX_HEIGHT_M[start:end].min()
          + KITTI360_HDL64E_APEX_HEIGHT_M[start:end].max()) / 2
    for start, end in KITTI360_HDL64E_BLOCKS
)
# A return that misses every laser cone by more than this is not from this unit.
MAX_CONE_RESIDUAL_M = 0.005
COLUMN_ASSIGNMENTS = ("firing_slot", "nearest")


def assign_lasers(xyz, chunk=32768):
    """Return each point's laser index (0 = top) and its cone residual in metres."""
    xyz = np.asarray(xyz, dtype=np.float64)
    tangent = np.tan(np.radians(KITTI360_HDL64E_ELEVATION_DEG))
    lasers = np.empty(len(xyz), dtype=np.int64)
    residual = np.empty(len(xyz), dtype=np.float64)
    for start in range(0, len(xyz), chunk):
        part = xyz[start:start + chunk]
        rho = np.hypot(part[:, 0], part[:, 1])
        error = np.abs(part[:, 2:3] - (rho[:, None] * tangent + KITTI360_HDL64E_APEX_HEIGHT_M))
        lasers[start:start + chunk] = error.argmin(axis=1)
        residual[start:start + chunk] = error.min(axis=1)
    return lasers, residual


def block_of_lasers(lasers):
    """Block index for each laser index."""
    lasers = np.asarray(lasers)
    block = np.zeros(lasers.shape, dtype=np.int64)
    for index, (start, end) in enumerate(KITTI360_HDL64E_BLOCKS):
        block[(lasers >= start) & (lasers < end)] = index
    return block


def _isotonic_increasing(values):
    """L2 isotonic (non-decreasing) regression by pool-adjacent-violators."""
    means, weights, counts = [], [], []
    for value in values:
        means.append(float(value))
        weights.append(1.0)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            weights[-2] = weight
            counts[-2] += counts[-1]
            means.pop()
            weights.pop()
            counts.pop()
    return np.repeat(means, counts)


def firing_slot_columns(position, width, max_shift=1.0):
    """Distinct, order-preserving columns for one laser's returns.

    ``position`` is each return's fractional column. A laser fires about once
    per column (48 us cycle, ~2177 firings per revolution at 9.57 Hz), but the
    stored azimuths are quantized to the 0.09 deg encoder grid (near returns
    are also turned by asin(+-26 mm / rho)), so rounding
    puts two consecutive firings into one 0.165 deg column for ~7.5% of
    returns. Sorting returns by azimuth and fitting strictly increasing
    integers (isotonic regression of ``position - rank``) gives each firing
    its own column. Returns that would move more than ``max_shift`` columns
    keep their rounded column. Returns ``(columns, shift)``.
    """
    position = np.asarray(position, dtype=np.float64)
    rounded = np.rint(position).astype(np.int64) % width
    if len(position) < 2:
        return rounded, np.rint(position) - position
    order = np.argsort(position, kind="stable")
    ordered = position[order]
    # Unwrap the 360-degree seam at the largest azimuth gap.
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + width]))
    start = (int(np.argmax(gaps)) + 1) % len(ordered)
    sequence = np.roll(np.arange(len(ordered)), -start)
    unwrapped = ordered[sequence].copy()
    unwrapped[unwrapped < unwrapped[0]] += width
    rank = np.arange(len(unwrapped))
    slots = rank + np.rint(_isotonic_increasing(unwrapped - rank)).astype(np.int64)
    columns = np.empty(len(position), dtype=np.int64)
    shift = np.empty(len(position), dtype=np.float64)
    columns[order[sequence]] = slots % width
    shift[order[sequence]] = slots - unwrapped
    far = np.abs(shift) > max_shift
    columns[far] = rounded[far]
    shift[far] = np.rint(position[far]) - position[far]
    return columns, shift


def project_hdl64e_panorama(points, width, max_depth=80.0, column_assignment="firing_slot"):
    """Laser-indexed GT panorama: one row per laser, ranges from block origins.

    Row ``k`` holds laser ``k``. Each return's range is measured from its
    block's render origin ``(0, 0, KITTI360_HDL64E_BLOCK_ORIGIN_Z_M[g])`` in
    the sensor frame, and its column comes from the azimuth around the sensor
    z axis (column W/2 faces +x, columns increase toward -y, the seam wraps).
    Pixels shared after column assignment keep the nearest return.
    """
    if column_assignment not in COLUMN_ASSIGNMENTS:
        raise ValueError(f"column_assignment must be one of {COLUMN_ASSIGNMENTS}")
    points = np.asarray(points, dtype=np.float32)
    width = int(width)
    height = len(KITTI360_HDL64E_ELEVATION_DEG)
    finite = np.isfinite(points).all(axis=1)
    xyz = np.where(finite[:, None], points[:, :3].astype(np.float64), 0.0)
    lasers, residual = assign_lasers(xyz)
    block = block_of_lasers(lasers)
    origin_z = np.asarray(KITTI360_HDL64E_BLOCK_ORIGIN_Z_M)[block]
    ranges = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + (xyz[:, 2] - origin_z) ** 2)
    valid = finite & (np.linalg.norm(xyz, axis=1) > 0) & (ranges < max_depth)
    worst = float(residual[valid].max()) if valid.any() else 0.0
    if worst > MAX_CONE_RESIDUAL_M:
        raise ValueError(f"Return misses every KITTI-360 HDL-64E laser cone by {worst:.4f} m")
    position = (np.pi - np.arctan2(xyz[:, 1], xyz[:, 0])) / (2 * np.pi) * width
    columns = np.rint(position).astype(np.int64) % width
    shift = np.zeros(len(points), dtype=np.float64)
    if column_assignment == "firing_slot":
        for laser in range(height):
            members = np.flatnonzero(valid & (lasers == laser))
            if len(members):
                columns[members], shift[members] = firing_slot_columns(position[members], width)
    ids = np.flatnonzero(valid)
    pixels = lasers[ids] * width + columns[ids]
    order = np.argsort(ranges[ids], kind="stable")
    _, first = np.unique(pixels[order], return_index=True)
    chosen = ids[order[first]]
    depth = np.zeros(height * width, dtype=np.float32)
    intensity = np.zeros_like(depth)
    depth[lasers[chosen] * width + columns[chosen]] = ranges[chosen]
    intensity[lasers[chosen] * width + columns[chosen]] = points[chosen, 3]
    stats = {
        "raw_points": int(len(points)),
        "valid_points": int(len(ids)),
        "gt_pixels": int(len(chosen)),
        "collision_points": int(len(ids) - len(chosen)),
        "max_cone_residual_m": worst,
        "max_column_shift": float(np.abs(shift[ids]).max()) if len(ids) else 0.0,
    }
    return depth.reshape(1, height, width), intensity.reshape(1, height, width), stats


def scan_order_lasers(xyz):
    """Laser index from scan order: a new laser starts when the sweep re-crosses +x.

    Used only to derive the calibration table; 3 of 1,102 checked scans split
    one laser in two, so inference uses ``assign_lasers`` instead.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    seam_azimuth = -np.arctan2(xyz[:, 1], -xyz[:, 0])
    return np.concatenate([[0], np.cumsum(np.diff(seam_azimuth) < 0)])


def fit_laser_cones(rho, z, iterations=5):
    """Robust line fit ``z = rho*tan(delta) + V``; returns ``(delta_deg, V)``."""
    rho, z = np.asarray(rho, dtype=np.float64), np.asarray(z, dtype=np.float64)
    design = np.stack([rho, np.ones_like(rho)], axis=1)
    keep = np.ones(len(z), dtype=bool)
    for _ in range(iterations):
        coefficient, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        error = np.abs(z - design @ coefficient)
        scale = 1.4826 * np.median(error[keep])
        keep = error < max(3 * scale, 1e-4)
    return float(np.degrees(np.arctan(coefficient[0]))), float(coefficient[1])
