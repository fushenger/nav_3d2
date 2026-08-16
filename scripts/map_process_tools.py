#!/usr/bin/env python3
"""
Utility tools for map_process experiments.

This file intentionally does not modify the existing ROS nodes.  It provides:

1. level
   Estimate the dominant floor plane from a PCD map, rotate the map so that the
   plane normal aligns with +Z, and apply the same transform to the trajectory.

2. obstacle-filter
   Remove traversable points that have nearby points in the original/leveled map
   above them.  This compensates for cases where the ground extractor outputs a
   floor patch under/near low obstacles.

3. repair-traversable
   Rebuild large planar traversable surfaces on a regular grid, filling only
   bounded small gaps.  Remaining stair treads are fitted as bounded, continuous
   ramps that are checked against the adjacent planar surfaces.

Supported PCD DATA encodings: binary and ascii.  The current project data is
binary PCD, which is the primary path.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.transform import Rotation


PCD_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
}


XYZI_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
    ]
)


def _parse_pcd_header(path: Path) -> Tuple[List[str], Dict[str, List[str]], int]:
    header: List[str] = []
    meta: Dict[str, List[str]] = {}

    with path.open("rb") as fh:
        while True:
            line_bytes = fh.readline()
            if not line_bytes:
                raise ValueError(f"{path}: missing DATA line in PCD header")
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            header.append(line)
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                parts = stripped.split()
                meta[parts[0].upper()] = parts[1:]
                if parts[0].upper() == "DATA":
                    return header, meta, fh.tell()


def _pcd_dtype(meta: Dict[str, List[str]]) -> np.dtype:
    fields = meta["FIELDS"]
    sizes = [int(x) for x in meta["SIZE"]]
    types = meta["TYPE"]
    counts = [int(x) for x in meta.get("COUNT", ["1"] * len(fields))]

    dtype_fields = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        np_type = PCD_TYPE_MAP.get((typ.upper(), size))
        if np_type is None:
            raise ValueError(f"Unsupported PCD field type: {name} TYPE={typ} SIZE={size}")
        if count == 1:
            dtype_fields.append((name, np_type))
        else:
            dtype_fields.append((name, np_type, (count,)))
    return np.dtype(dtype_fields)


def load_pcd(path: Path) -> Tuple[List[str], Dict[str, List[str]], np.ndarray]:
    header, meta, offset = _parse_pcd_header(path)
    dtype = _pcd_dtype(meta)
    data_kind = meta["DATA"][0].lower()
    points = int(meta["POINTS"][0])

    if data_kind == "binary":
        raw = path.read_bytes()[offset:]
        cloud = np.frombuffer(raw, dtype=dtype, count=points).copy()
    elif data_kind == "ascii":
        raw = path.read_bytes()[offset:]
        arr = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        fields = meta["FIELDS"]
        if arr.shape[1] != len(fields):
            raise ValueError(f"{path}: ascii column count mismatch")
        cloud = np.empty(arr.shape[0], dtype=dtype)
        for i, name in enumerate(fields):
            cloud[name] = arr[:, i].astype(cloud.dtype[name])
    else:
        raise ValueError(f"{path}: unsupported PCD DATA encoding: {data_kind}")

    for required in ("x", "y", "z"):
        if required not in cloud.dtype.names:
            raise ValueError(f"{path}: missing required PCD field '{required}'")
    return header, meta, cloud


def _updated_header(header: List[str], meta: Dict[str, List[str]], npoints: int) -> bytes:
    out: List[str] = []
    for line in header:
        key = line.strip().split(maxsplit=1)[0].upper() if line.strip() else ""
        if key == "WIDTH":
            out.append(f"WIDTH {npoints}")
        elif key == "HEIGHT":
            out.append("HEIGHT 1")
        elif key == "POINTS":
            out.append(f"POINTS {npoints}")
        else:
            out.append(line)
    return ("\n".join(out) + "\n").encode("utf-8")


def write_pcd(path: Path, header: List[str], meta: Dict[str, List[str]], cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_kind = meta["DATA"][0].lower()
    with path.open("wb") as fh:
        fh.write(_updated_header(header, meta, len(cloud)))
        if data_kind == "binary":
            cloud.tofile(fh)
        elif data_kind == "ascii":
            fields = meta["FIELDS"]
            arr = np.column_stack([cloud[name] for name in fields])
            np.savetxt(fh, arr, fmt="%.9g")
        else:
            raise ValueError(f"Unsupported PCD DATA encoding: {data_kind}")


def xyzi_pcd_header() -> Tuple[List[str], Dict[str, List[str]]]:
    """Return a planner-compatible binary PointXYZI PCD header."""
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z intensity",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        "WIDTH 0",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 0",
        "DATA binary",
    ]
    return header, {"DATA": ["binary"]}


def to_xyzi(cloud: np.ndarray) -> np.ndarray:
    """Convert any supported input PCD fields into the planner's PointXYZI layout."""
    output = np.empty(len(cloud), dtype=XYZI_DTYPE)
    output["x"] = cloud["x"].astype(np.float32, copy=False)
    output["y"] = cloud["y"].astype(np.float32, copy=False)
    output["z"] = cloud["z"].astype(np.float32, copy=False)

    names = cloud.dtype.names or ()
    if "intensity" in names:
        output["intensity"] = cloud["intensity"].astype(np.float32, copy=False)
    elif "scalar_intensity" in names:
        output["intensity"] = cloud["scalar_intensity"].astype(np.float32, copy=False)
    else:
        output["intensity"] = 0.0
    return output


def xyz_view(cloud: np.ndarray) -> np.ndarray:
    return np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float64, copy=False)


def transform_xyz_in_place(cloud: np.ndarray, rotation: np.ndarray, pivot: np.ndarray) -> None:
    pts = xyz_view(cloud)
    transformed = (pts - pivot) @ rotation.T + pivot
    cloud["x"] = transformed[:, 0].astype(cloud["x"].dtype, copy=False)
    cloud["y"] = transformed[:, 1].astype(cloud["y"].dtype, copy=False)
    cloud["z"] = transformed[:, 2].astype(cloud["z"].dtype, copy=False)

    names = cloud.dtype.names or ()
    if {"normal_x", "normal_y", "normal_z"}.issubset(names):
        normals = np.column_stack((cloud["normal_x"], cloud["normal_y"], cloud["normal_z"])).astype(
            np.float64, copy=False
        )
        rotated_normals = normals @ rotation.T
        cloud["normal_x"] = rotated_normals[:, 0].astype(cloud["normal_x"].dtype, copy=False)
        cloud["normal_y"] = rotated_normals[:, 1].astype(cloud["normal_y"].dtype, copy=False)
        cloud["normal_z"] = rotated_normals[:, 2].astype(cloud["normal_z"].dtype, copy=False)


def rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    norm_cross = np.linalg.norm(cross)
    if norm_cross < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - src * np.dot(axis, src)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi).as_matrix()
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (norm_cross**2))


def lower_surface_candidates(
    points: np.ndarray,
    grid_size: float,
    percentile: float,
    min_bin_points: int,
) -> np.ndarray:
    xy = points[:, :2]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    xy = xy[finite]
    min_xy = xy.min(axis=0)
    ij = np.floor((xy - min_xy) / grid_size).astype(np.int64)
    keys = ij[:, 0] * 4_294_967_291 + ij[:, 1]
    order = np.argsort(keys)
    keys_sorted = keys[order]
    change = np.r_[True, keys_sorted[1:] != keys_sorted[:-1], True]
    bounds = np.flatnonzero(change)

    selected: List[int] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start < min_bin_points:
            continue
        idxs = order[start:end]
        z_values = points[idxs, 2]
        z_ref = np.percentile(z_values, percentile)
        selected.append(int(idxs[np.argmin(np.abs(z_values - z_ref))]))

    if not selected:
        raise ValueError("No lower-surface candidates found; reduce --min-bin-points or increase sample size")
    return points[np.asarray(selected, dtype=np.int64)]


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int,
    threshold: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(iterations):
        ids = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        distances = np.abs((points - p0) @ normal)
        inliers = distances < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        raise ValueError("RANSAC failed to find a valid plane")

    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    return normal, centroid, best_count


def transform_trajectory_format1(parts: List[str], rotation: np.ndarray, pivot: np.ndarray) -> List[str]:
    pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    new_pos = rotation @ (pos - pivot) + pivot
    quat = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    new_quat = (Rotation.from_matrix(rotation) * Rotation.from_quat(quat)).as_quat()
    return [
        parts[0],
        f"{new_pos[0]:.9f}",
        f"{new_pos[1]:.9f}",
        f"{new_pos[2]:.9f}",
        f"{new_quat[0]:.9f}",
        f"{new_quat[1]:.9f}",
        f"{new_quat[2]:.9f}",
        f"{new_quat[3]:.9f}",
        *parts[8:],
    ]


def transform_trajectory_line(line: str, fmt: int, rotation: np.ndarray, pivot: np.ndarray) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line.rstrip("\n")
    parts = stripped.split()
    try:
        if fmt == 1:
            if len(parts) < 8:
                return line.rstrip("\n")
            return " ".join(transform_trajectory_format1(parts, rotation, pivot))
        if fmt == 2:
            if len(parts) < 3:
                return line.rstrip("\n")
            idxs = (0, 1, 2)
        elif fmt == 3:
            if len(parts) < 7:
                return line.rstrip("\n")
            idxs = (4, 5, 6)
        else:
            raise ValueError(f"Unsupported trajectory format: {fmt}")

        pos = np.array([float(parts[idxs[0]]), float(parts[idxs[1]]), float(parts[idxs[2]])])
        new_pos = rotation @ (pos - pivot) + pivot
        for idx, value in zip(idxs, new_pos):
            parts[idx] = f"{value:.9f}"
        return " ".join(parts)
    except ValueError:
        return line.rstrip("\n")


def transform_trajectory(
    input_path: Path,
    output_path: Path,
    fmt: int,
    rotation: np.ndarray,
    pivot: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            dst.write(transform_trajectory_line(line, fmt, rotation, pivot) + "\n")


def command_level(args: argparse.Namespace) -> None:
    header, meta, cloud = load_pcd(args.input_pcd)
    points = xyz_view(cloud)

    rng = np.random.default_rng(args.seed)
    if len(points) > args.max_sample_points:
        sample_idx = rng.choice(len(points), size=args.max_sample_points, replace=False)
        sample = points[sample_idx]
    else:
        sample = points

    candidates = lower_surface_candidates(
        sample,
        grid_size=args.grid_size,
        percentile=args.ground_percentile,
        min_bin_points=args.min_bin_points,
    )
    normal, pivot, inlier_count = fit_plane_ransac(
        candidates,
        iterations=args.ransac_iterations,
        threshold=args.ransac_threshold,
        rng=rng,
    )
    tilt_deg = math.degrees(math.acos(float(np.clip(normal @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))))
    if tilt_deg > args.max_correction_deg:
        raise RuntimeError(
            f"Estimated tilt is {tilt_deg:.2f} deg, above --max-correction-deg={args.max_correction_deg}. "
            "This usually means the plane fit locked onto a wall or ramp. Increase sample quality or limit input."
        )

    rotation = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
    transform_xyz_in_place(cloud, rotation, pivot)
    write_pcd(args.output_pcd, header, meta, cloud)
    transform_trajectory(args.input_trajectory, args.output_trajectory, args.trajectory_format, rotation, pivot)

    print(f"Input PCD:        {args.input_pcd}")
    print(f"Output PCD:       {args.output_pcd}")
    print(f"Input trajectory: {args.input_trajectory}")
    print(f"Output trajectory:{args.output_trajectory}")
    print(f"Estimated ground normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"Correction tilt: {tilt_deg:.3f} deg")
    print(f"Plane support: {inlier_count}/{len(candidates)} lower-surface candidates")
    print(f"Rotation matrix:\n{rotation}")
    print(f"Pivot: [{pivot[0]:.6f}, {pivot[1]:.6f}, {pivot[2]:.6f}]")


def command_obstacle_filter(args: argparse.Namespace) -> None:
    ref_header, ref_meta, ref_cloud = load_pcd(args.reference_pcd)
    trav_header, trav_meta, trav_cloud = load_pcd(args.input_traversable_pcd)
    del ref_header, ref_meta

    ref_xyz = xyz_view(ref_cloud)
    trav_xyz = xyz_view(trav_cloud)
    finite_ref = np.isfinite(ref_xyz).all(axis=1)
    ref_xyz = ref_xyz[finite_ref]

    tree = cKDTree(ref_xyz[:, :2])
    keep = np.ones(len(trav_xyz), dtype=bool)

    for start in range(0, len(trav_xyz), args.chunk_size):
        end = min(start + args.chunk_size, len(trav_xyz))
        neighbors = tree.query_ball_point(trav_xyz[start:end, :2], r=args.obstacle_radius)
        for local_i, ids in enumerate(neighbors):
            if not ids:
                continue
            global_i = start + local_i
            dz = ref_xyz[np.asarray(ids, dtype=np.int64), 2] - trav_xyz[global_i, 2]
            if np.any((dz >= args.min_obstacle_height) & (dz <= args.max_obstacle_height)):
                keep[global_i] = False

    filtered = trav_cloud[keep].copy()
    write_pcd(args.output_pcd, trav_header, trav_meta, filtered)
    removed = len(trav_cloud) - len(filtered)
    pct = 100.0 * removed / max(1, len(trav_cloud))
    print(f"Reference PCD:    {args.reference_pcd}")
    print(f"Input traversable:{args.input_traversable_pcd}")
    print(f"Output PCD:       {args.output_pcd}")
    print(f"Removed {removed}/{len(trav_cloud)} traversable points ({pct:.2f}%)")
    print(
        "Obstacle rule: remove traversable point if reference map has a nearby point "
        f"within {args.obstacle_radius:.2f} m XY and "
        f"{args.min_obstacle_height:.2f}..{args.max_obstacle_height:.2f} m above it"
    )


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one least-squares plane to points and return its upward-facing normal."""
    if len(points) < 3:
        raise ValueError("Need at least three points to fit a plane")
    pivot = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - pivot, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    return normal, pivot


def plane_residuals(points: np.ndarray, normal: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    return np.abs((points - pivot) @ normal)


def voxel_downsample(cloud: np.ndarray, leaf_size: float) -> np.ndarray:
    if leaf_size <= 0.0 or len(cloud) <= 1:
        return cloud
    xyz = xyz_view(cloud)
    voxel_indices = np.floor(xyz / leaf_size).astype(np.int64)
    _, keep = np.unique(voxel_indices, axis=0, return_index=True)
    return cloud[np.sort(keep)]


def _disk(radius: int) -> np.ndarray:
    """Return a compact 2-D circular structuring element."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _plane_grid(
    points: np.ndarray,
    normal: np.ndarray,
    pivot: np.ndarray,
    grid_size: float,
    fill_gap: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Project a nearly horizontal surface and fill only interior small gaps."""
    if len(points) == 0 or normal[2] <= 1e-5:
        return np.empty((0, 3), dtype=np.float64), {"input_cells": 0, "output_cells": 0}

    indices = np.floor(points[:, :2] / grid_size).astype(np.int64)
    lower = indices.min(axis=0)
    upper = indices.max(axis=0)
    shape = tuple((upper - lower + 1).tolist())
    occupied = np.zeros(shape, dtype=bool)
    occupied[indices[:, 0] - lower[0], indices[:, 1] - lower[1]] = True

    fill_radius = max(0, int(round(fill_gap / grid_size)))
    if fill_radius:
        closed = ndimage.binary_closing(occupied, structure=_disk(fill_radius))
    else:
        closed = occupied
    completed = ndimage.binary_fill_holes(closed)

    # Preserve the measured boundary.  New samples are permitted only when they
    # remain close to an accepted source cell, which prevents expansion into a
    # neighbouring room or through an obstacle-sized gap.
    if fill_radius:
        distance = ndimage.distance_transform_edt(~occupied)
        completed &= distance <= fill_radius
    completed |= occupied

    ii, jj = np.nonzero(completed)
    xy = (np.column_stack((ii, jj)) + lower + 0.5) * grid_size
    z = pivot[2] - (
        normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
    ) / normal[2]
    samples = np.column_stack((xy, z))
    return samples, {
        "input_cells": int(occupied.sum()),
        "output_cells": int(completed.sum()),
        "filled_cells": int(completed.sum() - occupied.sum()),
    }


def _fit_horizontal_surfaces(
    points: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[str, object]]]:
    """Extract substantial, close-to-level planes and return their grid samples."""
    active = np.ones(len(points), dtype=bool)
    samples: List[np.ndarray] = []
    report: List[Dict[str, object]] = []

    for plane_index in range(args.max_planar_surfaces):
        remaining = points[active]
        if len(remaining) < args.min_planar_cells:
            break
        fit_points = remaining
        if len(fit_points) > args.max_fit_points:
            fit_points = fit_points[rng.choice(len(fit_points), args.max_fit_points, replace=False)]
        candidate = _best_plane_with_slope(
            fit_points,
            args.planar_ransac_iterations,
            args.planar_distance,
            0.0,
            args.max_planar_slope_deg,
            rng,
        )
        if candidate is None:
            break
        normal, pivot = candidate

        residuals = plane_residuals(remaining, normal, pivot)
        inliers = residuals <= args.planar_distance
        if int(inliers.sum()) < args.min_planar_cells:
            break

        # Refit once after evaluating all remaining cells, then use this plane
        # to rebuild the surface rather than retaining noisy source heights.
        normal, pivot = fit_plane_svd(remaining[inliers])
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        residuals = plane_residuals(remaining, normal, pivot)
        inliers = residuals <= args.planar_distance
        if slope > args.max_planar_slope_deg or int(inliers.sum()) < args.min_planar_cells:
            break

        grid, grid_stats = _plane_grid(
            remaining[inliers], normal, pivot, args.grid_size, args.planar_fill_gap
        )
        if len(grid) < args.min_planar_cells:
            break
        samples.append(grid)
        report.append(
            {
                "index": plane_index + 1,
                "source_cells": int(inliers.sum()),
                "slope_deg": slope,
                "normal": [float(value) for value in normal],
                "point_on_plane": [float(value) for value in pivot],
                **grid_stats,
            }
        )
        active_indices = np.flatnonzero(active)
        active[active_indices[inliers]] = False

    return samples, points[active], report


def _best_plane_with_slope(
    points: np.ndarray,
    iterations: int,
    distance: float,
    min_slope: float,
    max_slope: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """RANSAC constrained by surface slope, so stair planes do not block floors."""
    best_normal: np.ndarray | None = None
    best_pivot: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        ids = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        magnitude = np.linalg.norm(normal)
        if magnitude < 1e-9:
            continue
        normal /= magnitude
        if normal[2] < 0.0:
            normal = -normal
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        if not min_slope <= slope <= max_slope:
            continue
        count = int((np.abs((points - p0) @ normal) <= distance).sum())
        if count > best_count:
            best_normal, best_pivot, best_count = normal, p0, count
    if best_normal is None or best_pivot is None:
        return None
    inliers = plane_residuals(points, best_normal, best_pivot) <= distance
    if int(inliers.sum()) < 3:
        return None
    return fit_plane_svd(points[inliers])


def _best_stair_plane(
    points: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find the strongest bounded inclined surface among residual tread cells."""
    if len(points) < 3:
        return None
    fit_points = points
    if len(fit_points) > args.max_fit_points:
        fit_points = fit_points[rng.choice(len(fit_points), args.max_fit_points, replace=False)]

    candidate = _best_plane_with_slope(
        fit_points,
        args.stair_ransac_iterations,
        args.stair_distance,
        args.min_stair_slope_deg,
        args.max_stair_slope_deg,
        rng,
    )
    if candidate is None:
        return None
    best_normal, best_pivot = candidate
    mask = plane_residuals(points, best_normal, best_pivot) <= args.stair_distance
    if int(mask.sum()) < args.min_stair_cells:
        return None
    normal, pivot = fit_plane_svd(points[mask])
    slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    if not args.min_stair_slope_deg <= slope <= args.max_stair_slope_deg:
        return None
    mask = plane_residuals(points, normal, pivot) <= args.stair_distance
    return normal, pivot, mask


def _stair_grid(
    points: np.ndarray,
    normal: np.ndarray,
    pivot: np.ndarray,
    grid_size: float,
    maximum_gap: float,
    minimum_length: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate a continuous ramp inside the observed stair footprint."""
    horizontal_normal = normal[:2]
    horizontal_norm = float(np.linalg.norm(horizontal_normal))
    if horizontal_norm < 1e-6:
        return np.empty((0, 3), dtype=np.float64), {}
    uphill = -np.array([normal[0], normal[1], 0.0]) / horizontal_norm
    crosswise = np.array([-uphill[1], uphill[0], 0.0])
    offsets = points - pivot
    longitudinal = offsets @ uphill
    lateral = offsets @ crosswise
    lo_bin = int(np.floor(longitudinal.min() / grid_size))
    hi_bin = int(np.floor(longitudinal.max() / grid_size))
    bin_count = hi_bin - lo_bin + 1
    if bin_count < 2:
        return np.empty((0, 3), dtype=np.float64), {}

    present = np.zeros(bin_count, dtype=bool)
    low = np.full(bin_count, np.nan)
    high = np.full(bin_count, np.nan)
    u_bins = np.floor(longitudinal / grid_size).astype(np.int64) - lo_bin
    for bin_index in np.unique(u_bins):
        values = lateral[u_bins == bin_index]
        present[bin_index] = True
        low[bin_index] = values.min()
        high[bin_index] = values.max()

    gap_bins = max(1, int(round(maximum_gap / grid_size)))
    connected = ndimage.binary_closing(present, structure=np.ones(gap_bins * 2 + 1, dtype=bool))
    connected &= ndimage.distance_transform_edt(~present) <= gap_bins
    labels, count = ndimage.label(connected)
    if count == 0:
        return np.empty((0, 3), dtype=np.float64), {}
    sizes = np.bincount(labels)[1:]
    selected_label = int(np.argmax(sizes)) + 1
    selected = labels == selected_label
    if selected.sum() * grid_size < minimum_length:
        return np.empty((0, 3), dtype=np.float64), {}

    valid_indices = np.flatnonzero(present)
    all_indices = np.arange(bin_count)
    low = np.interp(all_indices, valid_indices, low[present])
    high = np.interp(all_indices, valid_indices, high[present])
    samples: List[np.ndarray] = []
    for bin_index in np.flatnonzero(selected):
        u = (lo_bin + bin_index + 0.5) * grid_size
        v_lo = int(np.floor(low[bin_index] / grid_size))
        v_hi = int(np.ceil(high[bin_index] / grid_size))
        v = (np.arange(v_lo, v_hi + 1) + 0.5) * grid_size
        xy = pivot[:2] + u * uphill[:2] + v[:, None] * crosswise[:2]
        z = pivot[2] - (
            normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
        ) / normal[2]
        samples.append(np.column_stack((xy, z)))
    ramp = np.concatenate(samples) if samples else np.empty((0, 3), dtype=np.float64)
    u_range = np.flatnonzero(selected)
    return ramp, {
        "source_cells": int(len(points)),
        "ramp_cells": int(len(ramp)),
        "length_m": float(selected.sum() * grid_size),
        "width_median_m": float(np.median(high[selected] - low[selected])),
        "longitudinal_spacing_m": grid_size,
        "endpoint_u": [
            float((lo_bin + u_range[0] + 0.5) * grid_size),
            float((lo_bin + u_range[-1] + 0.5) * grid_size),
        ],
        "uphill_direction_xy": [float(uphill[0]), float(uphill[1])],
    }


def _ramp_connection_distances(
    ramp: np.ndarray,
    normal: np.ndarray,
    planar_samples: List[np.ndarray],
    grid_size: float,
) -> List[float]:
    """Measure the shortest links from each ramp end to rebuilt flat surfaces."""
    if not planar_samples or len(ramp) == 0:
        return []
    horizontal = normal[:2]
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        return []
    uphill = -np.array([normal[0], normal[1], 0.0]) / norm
    longitudinal = ramp @ uphill
    tree = cKDTree(np.concatenate(planar_samples))
    distances: List[float] = []
    for endpoint in (longitudinal.min(), longitudinal.max()):
        end_cells = ramp[np.abs(longitudinal - endpoint) <= grid_size]
        distance, _ = tree.query(end_cells, k=1)
        distances.append(float(np.min(distance)))
    return distances


def _xyzi_from_xyz(points: np.ndarray) -> np.ndarray:
    cloud = np.empty(len(points), dtype=XYZI_DTYPE)
    cloud["x"] = points[:, 0].astype(np.float32, copy=False)
    cloud["y"] = points[:, 1].astype(np.float32, copy=False)
    cloud["z"] = points[:, 2].astype(np.float32, copy=False)
    cloud["intensity"] = 1.0
    return cloud


def command_repair_traversable(args: argparse.Namespace) -> None:
    if args.grid_size <= 0.0:
        raise ValueError("--grid-size must be positive")
    if args.processing_grid_size <= 0.0:
        raise ValueError("--processing-grid-size must be positive")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    _, _, input_cloud = load_pcd(args.input_pcd)
    source = to_xyzi(input_cloud)
    source_xyz = xyz_view(source)
    source_xyz = source_xyz[np.isfinite(source_xyz).all(axis=1)]
    # Use a stable coarse proxy to identify surfaces, then render the accepted
    # fits at the requested output density.  Fine input cells make sparse stair
    # treads fragment during RANSAC, while they are ideal for visual completeness.
    source_grid = voxel_downsample(_xyzi_from_xyz(source_xyz), args.processing_grid_size)
    grid_xyz = xyz_view(source_grid)
    if len(grid_xyz) < args.min_planar_cells:
        raise RuntimeError("Input traversable cloud has too few valid grid cells")

    rng = np.random.default_rng(args.seed)
    planar_samples, residual, planar_report = _fit_horizontal_surfaces(grid_xyz, args, rng)
    stair_samples: List[np.ndarray] = []
    stair_report: List[Dict[str, object]] = []
    rejected_stairs: List[Dict[str, object]] = []
    remaining = residual
    for stair_index in range(args.max_stairs):
        result = _best_stair_plane(remaining, args, rng)
        if result is None:
            break
        normal, pivot, inliers = result
        ramp, stats = _stair_grid(
            remaining[inliers], normal, pivot, args.grid_size,
            args.stair_fill_gap, args.min_stair_length,
        )
        if len(ramp) < args.min_stair_cells:
            break
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        connections = _ramp_connection_distances(ramp, normal, planar_samples, args.grid_size)
        if len(connections) != 2 or max(connections) > args.max_stair_platform_gap:
            # This is not a proven platform-to-platform connector.  Preserve
            # its original cells in the residual rather than fabricating a ramp.
            rejected_stairs.append(
                {
                    "index": stair_index + 1,
                    "slope_deg": slope,
                    "platform_connection_distances_m": connections,
                    "reason": "not_connected_to_two_fitted_platforms",
                }
            )
            break
        stair_samples.append(ramp)
        stair_report.append(
            {
                "index": stair_index + 1,
                "slope_deg": slope,
                "normal": [float(value) for value in normal],
                "point_on_plane": [float(value) for value in pivot],
                "platform_connection_distances_m": connections,
                **stats,
            }
        )
        remaining = remaining[~inliers]

    if not stair_samples:
        raise RuntimeError("No connected stair ramp was fitted; no output was written")

    # Preserve any small accepted fragments not represented by a major plane or
    # a fitted stair.  They were already deemed traversable by the source stage.
    output_parts = planar_samples + stair_samples + [remaining]
    output_xyz = np.concatenate([part for part in output_parts if len(part)])
    output = voxel_downsample(_xyzi_from_xyz(output_xyz), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    report = {
        "input_pcd": str(args.input_pcd),
        "output_pcd": str(args.output_pcd),
        "grid_size_m": args.grid_size,
        "processing_grid_size_m": args.processing_grid_size,
        "input_points": int(len(source)),
        "input_grid_cells": int(len(grid_xyz)),
        "output_grid_cells": int(len(output)),
        "unmodelled_residual_cells": int(len(remaining)),
        "planes": planar_report,
        "stairs": stair_report,
        "rejected_stair_candidates": rejected_stairs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Output cells: {len(output)} at {args.grid_size:.3f} m")
    print(f"Fitted planes: {len(planar_report)}; connected stair ramps: {len(stair_report)}")
    print(f"Fit report: {args.report}")


def command_fill_plane_rectangle(args: argparse.Namespace) -> None:
    """Replace a confirmed local platform with a complete fitted plane grid."""
    if args.grid_size <= 0.0 or args.replace_distance <= 0.0:
        raise ValueError("--grid-size and --replace-distance must be positive")
    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("Rectangle bounds must satisfy min < max")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    normal = np.asarray(args.normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    if normal[2] <= 1e-5:
        raise ValueError("Local fitted plane must not be vertical")
    pivot = np.asarray(args.point_on_plane, dtype=np.float64)

    _, _, raw_input = load_pcd(args.input_pcd)
    input_cloud = to_xyzi(raw_input)
    input_xyz = xyz_view(input_cloud)
    local = (
        (input_xyz[:, 0] >= args.x_min)
        & (input_xyz[:, 0] <= args.x_max)
        & (input_xyz[:, 1] >= args.y_min)
        & (input_xyz[:, 1] <= args.y_max)
        & (plane_residuals(input_xyz, normal, pivot) <= args.replace_distance)
    )

    x_indices = np.arange(
        int(math.ceil(args.x_min / args.grid_size)),
        int(math.ceil(args.x_max / args.grid_size)),
        dtype=np.int64,
    )
    y_indices = np.arange(
        int(math.ceil(args.y_min / args.grid_size)),
        int(math.ceil(args.y_max / args.grid_size)),
        dtype=np.int64,
    )
    xx, yy = np.meshgrid((x_indices + 0.5) * args.grid_size,
                         (y_indices + 0.5) * args.grid_size, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    z = pivot[2] - (
        normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
    ) / normal[2]
    fitted = _xyzi_from_xyz(np.column_stack((xy, z)))
    output = voxel_downsample(np.concatenate((input_cloud[~local], fitted)), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    report = {
        "input_pcd": str(args.input_pcd),
        "output_pcd": str(args.output_pcd),
        "grid_size_m": args.grid_size,
        "replaced_source_points": int(local.sum()),
        "fitted_plane_points": int(len(fitted)),
        "output_points": int(len(output)),
        "bounds_xy": [args.x_min, args.x_max, args.y_min, args.y_max],
        "normal": [float(value) for value in normal],
        "point_on_plane": [float(value) for value in pivot],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Replaced local source points: {int(local.sum())}")
    print(f"Fitted rectangle cells: {len(fitted)}")
    print(f"Fit report: {args.report}")


def command_fill_plane_hull(args: argparse.Namespace) -> None:
    """Fill a local planar region at high density inside its observed XY hull."""
    if args.grid_size <= 0.0 or args.contour_distance <= 0.0 or args.replace_distance <= 0.0:
        raise ValueError("Grid size and distance thresholds must be positive")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    normal = np.asarray(args.normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    if normal[2] <= 1e-5:
        raise ValueError("Local fitted plane must not be vertical")
    pivot = np.asarray(args.point_on_plane, dtype=np.float64)

    _, _, contour_raw = load_pcd(args.contour_pcd)
    contour_xyz = xyz_view(to_xyzi(contour_raw))
    contour_mask = (
        (contour_xyz[:, 0] >= args.x_min)
        & (contour_xyz[:, 0] <= args.x_max)
        & (contour_xyz[:, 1] >= args.y_min)
        & (contour_xyz[:, 1] <= args.y_max)
        & (plane_residuals(contour_xyz, normal, pivot) <= args.contour_distance)
    )
    support = contour_xyz[contour_mask]
    if len(support) < 3:
        raise RuntimeError("No sufficient original contour points found for the requested local plane")
    hull = ConvexHull(support[:, :2])
    polygon = support[hull.vertices, :2]
    triangulation = Delaunay(polygon)

    x_indices = np.arange(
        int(math.floor(polygon[:, 0].min() / args.grid_size)),
        int(math.ceil(polygon[:, 0].max() / args.grid_size)),
        dtype=np.int64,
    )
    y_indices = np.arange(
        int(math.floor(polygon[:, 1].min() / args.grid_size)),
        int(math.ceil(polygon[:, 1].max() / args.grid_size)),
        dtype=np.int64,
    )
    xx, yy = np.meshgrid((x_indices + 0.5) * args.grid_size,
                         (y_indices + 0.5) * args.grid_size, indexing="ij")
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    xy = candidates[triangulation.find_simplex(candidates) >= 0]
    z = pivot[2] - (
        normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
    ) / normal[2]
    fitted = _xyzi_from_xyz(np.column_stack((xy, z)))

    _, _, input_raw = load_pcd(args.input_pcd)
    input_cloud = to_xyzi(input_raw)
    input_xyz = xyz_view(input_cloud)
    input_inside_hull = triangulation.find_simplex(input_xyz[:, :2]) >= 0
    local = input_inside_hull & (plane_residuals(input_xyz, normal, pivot) <= args.replace_distance)
    output = voxel_downsample(np.concatenate((input_cloud[~local], fitted)), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    report = {
        "input_pcd": str(args.input_pcd),
        "contour_pcd": str(args.contour_pcd),
        "output_pcd": str(args.output_pcd),
        "grid_size_m": args.grid_size,
        "contour_support_points": int(len(support)),
        "contour_hull_area_m2": float(hull.volume),
        "replaced_source_points": int(local.sum()),
        "fitted_plane_points": int(len(fitted)),
        "output_points": int(len(output)),
        "bounds_xy": [args.x_min, args.x_max, args.y_min, args.y_max],
        "normal": [float(value) for value in normal],
        "point_on_plane": [float(value) for value in pivot],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Observed hull area: {hull.volume:.3f} m^2")
    print(f"Fitted hull cells: {len(fitted)}")
    print(f"Fit report: {args.report}")


def command_bridge_surfaces(args: argparse.Namespace) -> None:
    """Create a dense planar transition strip between two confirmed surface points."""
    if args.grid_size <= 0.0 or args.width <= 0.0 or args.endpoint_overlap < 0.0:
        raise ValueError("Grid size and bridge width must be positive; overlap must be non-negative")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    start = np.asarray(args.start, dtype=np.float64)
    end = np.asarray(args.end, dtype=np.float64)
    delta_xy = end[:2] - start[:2]
    length = float(np.linalg.norm(delta_xy))
    if length < args.grid_size:
        raise ValueError("Bridge endpoints are too close")
    direction = delta_xy / length
    crosswise = np.array([-direction[1], direction[0]])
    slope_deg = math.degrees(math.atan2(abs(end[2] - start[2]), length))
    if slope_deg > args.max_slope_deg:
        raise RuntimeError(
            f"Bridge slope {slope_deg:.2f} deg exceeds --max-slope-deg={args.max_slope_deg:.2f}"
        )

    longitudinal = np.arange(-args.endpoint_overlap, length + args.endpoint_overlap + 1e-9,
                             args.grid_size)
    lateral = np.arange(-args.width / 2.0, args.width / 2.0 + 1e-9, args.grid_size)
    ss, tt = np.meshgrid(longitudinal, lateral, indexing="ij")
    xy = start[:2] + ss[..., None] * direction + tt[..., None] * crosswise
    z = start[2] + ss * (end[2] - start[2]) / length
    bridge = _xyzi_from_xyz(np.column_stack((xy.reshape(-1, 2), z.ravel())))

    _, _, raw_input = load_pcd(args.input_pcd)
    input_cloud = to_xyzi(raw_input)
    output = voxel_downsample(np.concatenate((input_cloud, bridge)), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    report = {
        "input_pcd": str(args.input_pcd),
        "output_pcd": str(args.output_pcd),
        "grid_size_m": args.grid_size,
        "width_m": args.width,
        "endpoint_overlap_m": args.endpoint_overlap,
        "start": [float(value) for value in start],
        "end": [float(value) for value in end],
        "horizontal_length_m": length,
        "slope_deg": slope_deg,
        "bridge_points": int(len(bridge)),
        "output_points": int(len(output)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Bridge: {length:.3f} m at {slope_deg:.2f} deg, width {args.width:.3f} m")
    print(f"Bridge points: {len(bridge)}")


def command_fill_plane_triangle(args: argparse.Namespace) -> None:
    """Replace one explicit triangular gap with a dense fitted traversable plane."""
    if args.grid_size <= 0.0 or args.replace_distance <= 0.0:
        raise ValueError("--grid-size and --replace-distance must be positive")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    vertices = np.asarray(args.vertices, dtype=np.float64).reshape(3, 3)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    magnitude = float(np.linalg.norm(normal))
    if magnitude < 1e-8:
        raise ValueError("Triangle vertices are collinear")
    normal /= magnitude
    if normal[2] < 0.0:
        normal = -normal
    if normal[2] <= 1e-5:
        raise RuntimeError("Triangle is vertical and cannot be a traversable surface")
    slope_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    if slope_deg > args.max_slope_deg:
        raise RuntimeError(
            f"Triangle slope {slope_deg:.2f} deg exceeds --max-slope-deg={args.max_slope_deg:.2f}"
        )
    triangulation = Delaunay(vertices[:, :2])
    x_indices = np.arange(
        int(math.floor(vertices[:, 0].min() / args.grid_size)),
        int(math.ceil(vertices[:, 0].max() / args.grid_size)),
        dtype=np.int64,
    )
    y_indices = np.arange(
        int(math.floor(vertices[:, 1].min() / args.grid_size)),
        int(math.ceil(vertices[:, 1].max() / args.grid_size)),
        dtype=np.int64,
    )
    xx, yy = np.meshgrid((x_indices + 0.5) * args.grid_size,
                         (y_indices + 0.5) * args.grid_size, indexing="ij")
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    xy = candidates[triangulation.find_simplex(candidates) >= 0]
    pivot = vertices[0]
    z = pivot[2] - (
        normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
    ) / normal[2]
    fitted = _xyzi_from_xyz(np.column_stack((xy, z)))

    _, _, raw_input = load_pcd(args.input_pcd)
    input_cloud = to_xyzi(raw_input)
    input_xyz = xyz_view(input_cloud)
    inside = triangulation.find_simplex(input_xyz[:, :2]) >= 0
    replace = inside & (plane_residuals(input_xyz, normal, pivot) <= args.replace_distance)
    output = voxel_downsample(np.concatenate((input_cloud[~replace], fitted)), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    area = 0.5 * abs(np.cross(vertices[1, :2] - vertices[0, :2],
                              vertices[2, :2] - vertices[0, :2]))
    report = {
        "input_pcd": str(args.input_pcd),
        "output_pcd": str(args.output_pcd),
        "vertices": [[float(value) for value in row] for row in vertices],
        "triangle_area_xy_m2": float(area),
        "normal": [float(value) for value in normal],
        "slope_deg": slope_deg,
        "grid_size_m": args.grid_size,
        "replaced_source_points": int(replace.sum()),
        "fitted_plane_points": int(len(fitted)),
        "output_points": int(len(output)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Triangle: {area:.3f} m^2 at {slope_deg:.2f} deg")
    print(f"Fitted triangle cells: {len(fitted)}")


def command_fill_plane_polygon(args: argparse.Namespace) -> None:
    """Replace a convex multi-corner gap with one dense fitted traversable plane."""
    if len(args.vertices) < 12 or len(args.vertices) % 3:
        raise ValueError("--vertices must contain at least four complete x y z triples")
    if args.grid_size <= 0.0:
        raise ValueError("--grid-size must be positive")
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    vertices = np.asarray(args.vertices, dtype=np.float64).reshape(-1, 3)
    normal, pivot = fit_plane_svd(vertices)
    residual = plane_residuals(vertices, normal, pivot)
    if float(residual.max()) > args.max_vertex_residual:
        raise RuntimeError(
            f"Boundary vertices do not describe one plane: max residual {residual.max():.3f} m "
            f"> --max-vertex-residual={args.max_vertex_residual:.3f} m"
        )
    slope_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    if slope_deg > args.max_slope_deg:
        raise RuntimeError(
            f"Fitted polygon slope {slope_deg:.2f} deg exceeds --max-slope-deg={args.max_slope_deg:.2f}"
        )
    polygon = Delaunay(vertices[:, :2])
    x_indices = np.arange(
        int(math.floor(vertices[:, 0].min() / args.grid_size)),
        int(math.ceil(vertices[:, 0].max() / args.grid_size)),
        dtype=np.int64,
    )
    y_indices = np.arange(
        int(math.floor(vertices[:, 1].min() / args.grid_size)),
        int(math.ceil(vertices[:, 1].max() / args.grid_size)),
        dtype=np.int64,
    )
    xx, yy = np.meshgrid((x_indices + 0.5) * args.grid_size,
                         (y_indices + 0.5) * args.grid_size, indexing="ij")
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    xy = candidates[polygon.find_simplex(candidates) >= 0]
    z = pivot[2] - (
        normal[0] * (xy[:, 0] - pivot[0]) + normal[1] * (xy[:, 1] - pivot[1])
    ) / normal[2]
    # Exact boundary points make the fitted region meet the selected stair and
    # landing points even when a grid-cell centre does not fall on a corner.
    fitted = _xyzi_from_xyz(np.vstack((np.column_stack((xy, z)), vertices)))

    _, _, raw_input = load_pcd(args.input_pcd)
    input_cloud = to_xyzi(raw_input)
    input_xyz = xyz_view(input_cloud)
    inside = polygon.find_simplex(input_xyz[:, :2]) >= 0
    # The caller selected this footprint as one traversable surface.  Replacing
    # all previous layers in it guarantees a single continuous planner surface.
    output = voxel_downsample(np.concatenate((input_cloud[~inside], fitted)), args.grid_size)
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)

    xy_vertices = vertices[:, :2]
    area = 0.5 * abs(np.dot(xy_vertices[:, 0], np.roll(xy_vertices[:, 1], -1))
                     - np.dot(xy_vertices[:, 1], np.roll(xy_vertices[:, 0], -1)))
    report = {
        "input_pcd": str(args.input_pcd),
        "output_pcd": str(args.output_pcd),
        "vertices": [[float(value) for value in row] for row in vertices],
        "normal": [float(value) for value in normal],
        "point_on_plane": [float(value) for value in pivot],
        "slope_deg": slope_deg,
        "max_vertex_residual_m": float(residual.max()),
        "area_xy_m2": float(area),
        "grid_size_m": args.grid_size,
        "removed_source_points": int(inside.sum()),
        "fitted_plane_points": int(len(fitted)),
        "output_points": int(len(output)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Polygon: {area:.3f} m^2 at {slope_deg:.2f} deg")
    print(f"Fitted polygon points: {len(fitted)}")


def command_crop_traversable_height(args: argparse.Namespace) -> None:
    """Keep a selected fitted footprint plus points at or below a map-frame height."""
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    _, _, raw_input = load_pcd(args.input_pcd)
    cloud = to_xyzi(raw_input)
    xyz = xyz_view(cloud)
    keep = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] <= args.max_z)
    polygon_points = 0
    if args.keep_vertices:
        if len(args.keep_vertices) < 9 or len(args.keep_vertices) % 3:
            raise ValueError("--keep-vertices must contain at least three complete x y z triples")
        vertices = np.asarray(args.keep_vertices, dtype=np.float64).reshape(-1, 3)
        polygon = Delaunay(vertices[:, :2])
        inside = polygon.find_simplex(xyz[:, :2]) >= 0
        keep |= inside
        polygon_points = int(inside.sum())
    output = cloud[keep]
    if len(output) == 0:
        raise RuntimeError("Height crop removed every traversable point")
    header, meta = xyzi_pcd_header()
    write_pcd(args.output_pcd, header, meta, output)
    print(f"Output PCD: {args.output_pcd}")
    print(f"Kept {len(output)}/{len(cloud)} points with z <= {args.max_z:.3f} m")
    if args.keep_vertices:
        print(f"Kept all {polygon_points} points inside the selected fitted footprint")


def load_manual_plane_regions(manifest_path: Path) -> List[Dict[str, object]]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc

    regions = raw.get("regions") if isinstance(raw, dict) else raw
    if not isinstance(regions, list) or not regions:
        raise ValueError("Manual-plane manifest must contain a non-empty 'regions' list")

    parsed: List[Dict[str, object]] = []
    for index, raw_region in enumerate(regions):
        if not isinstance(raw_region, dict):
            raise ValueError(f"Region #{index + 1} must be a JSON object")
        pcd_value = raw_region.get("pcd")
        if not isinstance(pcd_value, str) or not pcd_value:
            raise ValueError(f"Region #{index + 1} is missing its 'pcd' path")

        pcd_path = Path(pcd_value)
        if not pcd_path.is_absolute():
            pcd_path = manifest_path.parent / pcd_path
        if not pcd_path.is_file():
            raise ValueError(f"Region #{index + 1}: input PCD does not exist: {pcd_path}")

        fit_method = raw_region.get("fit", "ransac")
        if fit_method not in ("ransac", "svd"):
            raise ValueError(f"Region #{index + 1}: 'fit' must be 'ransac' or 'svd'")
        project_to_plane = raw_region.get("project_to_plane", False)
        if not isinstance(project_to_plane, bool):
            raise ValueError(f"Region #{index + 1}: 'project_to_plane' must be true or false")

        parsed.append(
            {
                "name": str(raw_region.get("name", pcd_path.stem)),
                "pcd_path": pcd_path,
                "fit": fit_method,
                "max_plane_distance": raw_region.get("max_plane_distance"),
                "min_inlier_ratio": raw_region.get("min_inlier_ratio"),
                "max_slope_deg": raw_region.get("max_slope_deg"),
                "project_to_plane": project_to_plane,
            }
        )
    return parsed


def number_setting(region: Dict[str, object], key: str, default: float) -> float:
    value = region.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Region '{region['name']}': '{key}' must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Region '{region['name']}': '{key}' must be a number") from exc


def command_manual_plane_map(args: argparse.Namespace) -> None:
    if args.output_pcd == args.report:
        raise ValueError("--output-pcd and --report must be different files")
    if args.voxel_size < 0.0:
        raise ValueError("--voxel-size must be non-negative")
    if args.max_fit_points < 3:
        raise ValueError("--max-fit-points must be at least 3")
    if args.ransac_iterations < 1:
        raise ValueError("--ransac-iterations must be positive")
    if args.output_pcd.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output_pcd}; pass --force to replace it")
    if args.report.exists() and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.report}; pass --force to replace it")

    regions = load_manual_plane_regions(args.manifest)
    rng = np.random.default_rng(args.seed)
    output_parts: List[np.ndarray] = []
    report_regions: List[Dict[str, object]] = []

    for region in regions:
        input_path = region["pcd_path"]
        assert isinstance(input_path, Path)
        _, _, input_cloud = load_pcd(input_path)
        source_cloud = to_xyzi(input_cloud)
        source_xyz = xyz_view(source_cloud)
        finite = np.isfinite(source_xyz).all(axis=1)
        source_cloud = source_cloud[finite]
        source_xyz = source_xyz[finite]
        if len(source_xyz) < 3:
            raise ValueError(f"Region '{region['name']}': fewer than three finite points")

        max_distance = number_setting(region, "max_plane_distance", args.default_max_plane_distance)
        min_ratio = number_setting(region, "min_inlier_ratio", args.default_min_inlier_ratio)
        max_slope = number_setting(region, "max_slope_deg", args.default_max_slope_deg)
        if max_distance <= 0.0:
            raise ValueError(f"Region '{region['name']}': max_plane_distance must be positive")
        if not 0.0 < min_ratio <= 1.0:
            raise ValueError(f"Region '{region['name']}': min_inlier_ratio must be within (0, 1]")
        if not 0.0 <= max_slope < 90.0:
            raise ValueError(f"Region '{region['name']}': max_slope_deg must be within [0, 90)")

        if len(source_xyz) > args.max_fit_points:
            sample_indices = rng.choice(len(source_xyz), size=args.max_fit_points, replace=False)
            fit_points = source_xyz[sample_indices]
        else:
            fit_points = source_xyz

        if region["fit"] == "ransac":
            normal, pivot, _ = fit_plane_ransac(
                fit_points,
                iterations=args.ransac_iterations,
                threshold=max_distance,
                rng=rng,
            )
        else:
            normal, pivot = fit_plane_svd(fit_points)

        residuals = plane_residuals(source_xyz, normal, pivot)
        inlier_mask = residuals <= max_distance
        inlier_ratio = float(inlier_mask.mean())
        slope_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        centroid = source_xyz.mean(axis=0)
        centroid_projection = centroid - float((centroid - pivot) @ normal) * normal
        if slope_deg > max_slope:
            raise RuntimeError(
                f"Region '{region['name']}': fitted slope {slope_deg:.2f} deg exceeds "
                f"max_slope_deg={max_slope:.2f}; the selection may include a wall"
            )
        if inlier_ratio < min_ratio:
            raise RuntimeError(
                f"Region '{region['name']}': only {inlier_ratio:.1%} of points are within "
                f"{max_distance:.3f} m of the fitted plane (minimum {min_ratio:.1%})"
            )

        accepted = source_cloud[inlier_mask].copy()
        if bool(region["project_to_plane"]):
            accepted_xyz = xyz_view(accepted)
            signed_distances = (accepted_xyz - pivot) @ normal
            projected = accepted_xyz - signed_distances[:, None] * normal
            accepted["x"] = projected[:, 0].astype(np.float32, copy=False)
            accepted["y"] = projected[:, 1].astype(np.float32, copy=False)
            accepted["z"] = projected[:, 2].astype(np.float32, copy=False)

        output_parts.append(accepted)
        report_regions.append(
            {
                "name": region["name"],
                "pcd": str(input_path),
                "fit": region["fit"],
                "input_points": int(len(source_cloud)),
                "accepted_points": int(len(accepted)),
                "inlier_ratio": inlier_ratio,
                "max_plane_distance": max_distance,
                "slope_deg": slope_deg,
                "normal": [float(value) for value in normal],
                "point_on_plane": [float(value) for value in pivot],
                "height_at_region_centroid": float(centroid_projection[2]),
                "project_to_plane": bool(region["project_to_plane"]),
            }
        )
        print(
            f"Region {region['name']}: {len(accepted)}/{len(source_cloud)} points, "
            f"slope {slope_deg:.2f} deg, height at region centroid {centroid_projection[2]:.3f} m"
        )

    merged = np.concatenate(output_parts)
    merged = voxel_downsample(merged, args.voxel_size)
    if len(merged) == 0:
        raise RuntimeError("Manual-plane map is empty after filtering")

    header, meta = xyzi_pcd_header()
    args.output_pcd.parent.mkdir(parents=True, exist_ok=True)
    write_pcd(args.output_pcd, header, meta, merged)
    report = {
        "manifest": str(args.manifest),
        "output_pcd": str(args.output_pcd),
        "voxel_size": args.voxel_size,
        "output_points": int(len(merged)),
        "regions": report_regions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Output PCD: {args.output_pcd}")
    print(f"Output points after {args.voxel_size:.3f} m voxel filter: {len(merged)}")
    print(f"Fit report: {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    level = subparsers.add_parser("level", help="Level a PCD map and matching trajectory")
    level.add_argument("--input-pcd", type=Path, required=True)
    level.add_argument("--input-trajectory", type=Path, required=True)
    level.add_argument("--output-pcd", type=Path, required=True)
    level.add_argument("--output-trajectory", type=Path, required=True)
    level.add_argument("--trajectory-format", type=int, default=1, choices=(1, 2, 3))
    level.add_argument("--max-sample-points", type=int, default=350_000)
    level.add_argument("--grid-size", type=float, default=0.45)
    level.add_argument("--ground-percentile", type=float, default=18.0)
    level.add_argument("--min-bin-points", type=int, default=5)
    level.add_argument("--ransac-iterations", type=int, default=450)
    level.add_argument("--ransac-threshold", type=float, default=0.08)
    level.add_argument("--max-correction-deg", type=float, default=20.0)
    level.add_argument("--seed", type=int, default=7)
    level.set_defaults(func=command_level)

    obstacle = subparsers.add_parser("obstacle-filter", help="Remove traversable points near raised obstacles")
    obstacle.add_argument("--reference-pcd", type=Path, required=True)
    obstacle.add_argument("--input-traversable-pcd", type=Path, required=True)
    obstacle.add_argument("--output-pcd", type=Path, required=True)
    obstacle.add_argument("--obstacle-radius", type=float, default=0.35)
    obstacle.add_argument("--min-obstacle-height", type=float, default=0.12)
    obstacle.add_argument("--max-obstacle-height", type=float, default=1.40)
    obstacle.add_argument("--chunk-size", type=int, default=1500)
    obstacle.set_defaults(func=command_obstacle_filter)

    repair = subparsers.add_parser(
        "repair-traversable",
        help="Fill bounded planar gaps and rebuild connected stairs as smooth ramps",
    )
    repair.add_argument("--input-pcd", type=Path, required=True)
    repair.add_argument("--output-pcd", type=Path, required=True)
    repair.add_argument("--report", type=Path, required=True)
    repair.add_argument("--grid-size", type=float, default=0.05)
    repair.add_argument("--processing-grid-size", type=float, default=0.05)
    repair.add_argument("--planar-distance", type=float, default=0.045)
    repair.add_argument("--max-planar-slope-deg", type=float, default=25.0)
    repair.add_argument("--planar-fill-gap", type=float, default=0.25)
    repair.add_argument("--min-planar-cells", type=int, default=350)
    repair.add_argument("--max-planar-surfaces", type=int, default=16)
    repair.add_argument("--planar-ransac-iterations", type=int, default=450)
    repair.add_argument("--min-stair-slope-deg", type=float, default=10.0)
    repair.add_argument("--max-stair-slope-deg", type=float, default=55.0)
    repair.add_argument("--stair-distance", type=float, default=0.13)
    repair.add_argument("--stair-fill-gap", type=float, default=0.30)
    repair.add_argument("--min-stair-length", type=float, default=0.60)
    repair.add_argument("--min-stair-cells", type=int, default=120)
    repair.add_argument("--max-stairs", type=int, default=4)
    repair.add_argument("--stair-ransac-iterations", type=int, default=1200)
    repair.add_argument("--max-stair-platform-gap", type=float, default=0.18)
    repair.add_argument("--max-fit-points", type=int, default=60_000)
    repair.add_argument("--seed", type=int, default=7)
    repair.add_argument("--force", action="store_true")
    repair.set_defaults(func=command_repair_traversable)

    rectangle = subparsers.add_parser(
        "fill-plane-rectangle",
        help="Replace one confirmed local platform footprint with a complete fitted plane",
    )
    rectangle.add_argument("--input-pcd", type=Path, required=True)
    rectangle.add_argument("--output-pcd", type=Path, required=True)
    rectangle.add_argument("--report", type=Path, required=True)
    rectangle.add_argument("--normal", nargs=3, type=float, required=True, metavar=("NX", "NY", "NZ"))
    rectangle.add_argument(
        "--point-on-plane", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    rectangle.add_argument("--x-min", type=float, required=True)
    rectangle.add_argument("--x-max", type=float, required=True)
    rectangle.add_argument("--y-min", type=float, required=True)
    rectangle.add_argument("--y-max", type=float, required=True)
    rectangle.add_argument("--grid-size", type=float, default=0.025)
    rectangle.add_argument("--replace-distance", type=float, default=0.06)
    rectangle.add_argument("--force", action="store_true")
    rectangle.set_defaults(func=command_fill_plane_rectangle)

    hull = subparsers.add_parser(
        "fill-plane-hull",
        help="Replace a local planar region inside its original observed XY hull",
    )
    hull.add_argument("--input-pcd", type=Path, required=True)
    hull.add_argument("--contour-pcd", type=Path, required=True)
    hull.add_argument("--output-pcd", type=Path, required=True)
    hull.add_argument("--report", type=Path, required=True)
    hull.add_argument("--normal", nargs=3, type=float, required=True, metavar=("NX", "NY", "NZ"))
    hull.add_argument(
        "--point-on-plane", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    hull.add_argument("--x-min", type=float, required=True)
    hull.add_argument("--x-max", type=float, required=True)
    hull.add_argument("--y-min", type=float, required=True)
    hull.add_argument("--y-max", type=float, required=True)
    hull.add_argument("--grid-size", type=float, default=0.025)
    hull.add_argument("--contour-distance", type=float, default=0.045)
    hull.add_argument("--replace-distance", type=float, default=0.06)
    hull.add_argument("--force", action="store_true")
    hull.set_defaults(func=command_fill_plane_hull)

    bridge = subparsers.add_parser(
        "bridge-surfaces",
        help="Create a dense planar transition strip between two confirmed traversable surfaces",
    )
    bridge.add_argument("--input-pcd", type=Path, required=True)
    bridge.add_argument("--output-pcd", type=Path, required=True)
    bridge.add_argument("--report", type=Path, required=True)
    bridge.add_argument("--start", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    bridge.add_argument("--end", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    bridge.add_argument("--width", type=float, default=0.50)
    bridge.add_argument("--endpoint-overlap", type=float, default=0.05)
    bridge.add_argument("--grid-size", type=float, default=0.025)
    bridge.add_argument("--max-slope-deg", type=float, default=60.0)
    bridge.add_argument("--force", action="store_true")
    bridge.set_defaults(func=command_bridge_surfaces)

    triangle = subparsers.add_parser(
        "fill-plane-triangle",
        help="Replace one explicit three-point gap with a dense fitted triangular plane",
    )
    triangle.add_argument("--input-pcd", type=Path, required=True)
    triangle.add_argument("--output-pcd", type=Path, required=True)
    triangle.add_argument("--report", type=Path, required=True)
    triangle.add_argument(
        "--vertices", nargs=9, type=float, required=True,
        metavar=("X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3"),
    )
    triangle.add_argument("--grid-size", type=float, default=0.025)
    triangle.add_argument("--replace-distance", type=float, default=0.10)
    triangle.add_argument("--max-slope-deg", type=float, default=60.0)
    triangle.add_argument("--force", action="store_true")
    triangle.set_defaults(func=command_fill_plane_triangle)

    polygon = subparsers.add_parser(
        "fill-plane-polygon",
        help="Replace a convex multi-point gap with a single dense fitted plane",
    )
    polygon.add_argument("--input-pcd", type=Path, required=True)
    polygon.add_argument("--output-pcd", type=Path, required=True)
    polygon.add_argument("--report", type=Path, required=True)
    polygon.add_argument("--vertices", nargs="+", type=float, required=True, metavar="XYZ")
    polygon.add_argument("--grid-size", type=float, default=0.025)
    polygon.add_argument("--max-vertex-residual", type=float, default=0.03)
    polygon.add_argument("--max-slope-deg", type=float, default=60.0)
    polygon.add_argument("--force", action="store_true")
    polygon.set_defaults(func=command_fill_plane_polygon)

    crop_height = subparsers.add_parser(
        "crop-traversable-height",
        help="Keep only traversable PCD points at or below a map-frame height",
    )
    crop_height.add_argument("--input-pcd", type=Path, required=True)
    crop_height.add_argument("--output-pcd", type=Path, required=True)
    crop_height.add_argument("--max-z", type=float, required=True)
    crop_height.add_argument("--keep-vertices", nargs="*", type=float, default=[])
    crop_height.add_argument("--force", action="store_true")
    crop_height.set_defaults(func=command_crop_traversable_height)

    manual_plane = subparsers.add_parser(
        "manual-plane-map",
        help="Fit separately selected traversable regions into a multi-height planner PCD",
    )
    manual_plane.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON list of manually selected region PCDs and per-region fit settings",
    )
    manual_plane.add_argument("--output-pcd", type=Path, required=True)
    manual_plane.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON report containing the fitted height, slope, and acceptance ratio for every region",
    )
    manual_plane.add_argument("--voxel-size", type=float, default=0.05)
    manual_plane.add_argument("--default-max-plane-distance", type=float, default=0.05)
    manual_plane.add_argument("--default-min-inlier-ratio", type=float, default=0.70)
    manual_plane.add_argument("--default-max-slope-deg", type=float, default=60.0)
    manual_plane.add_argument("--max-fit-points", type=int, default=80_000)
    manual_plane.add_argument("--ransac-iterations", type=int, default=400)
    manual_plane.add_argument("--seed", type=int, default=7)
    manual_plane.add_argument("--force", action="store_true")
    manual_plane.set_defaults(func=command_manual_plane_map)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
