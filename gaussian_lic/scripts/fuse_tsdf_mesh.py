#!/usr/bin/env python3
"""Fuse Gaussian-LIC TSDF export frames into a triangle mesh with Open3D."""

import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d


def read_intrinsics(path: Path):
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        values[key] = float(value)
    return o3d.camera.PinholeCameraIntrinsic(
        int(values["width"]), int(values["height"]), values["fx"], values["fy"], values["cx"], values["cy"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="<result_path>_tsdf_input (sibling directory of the mapping result)")
    parser.add_argument("--output", type=Path, default=None, help="mesh output path (default: input/../tsdf_mesh.ply)")
    parser.add_argument("--voxel", type=float, default=0.04, help="TSDF voxel size in metres")
    parser.add_argument("--truncation", type=float, default=0.16, help="TSDF truncation distance in metres")
    parser.add_argument("--depth-trunc", type=float, default=20.0, help="discard depth beyond this distance in metres")
    parser.add_argument("--min-cluster-triangles", type=int, default=200, help="remove smaller disconnected mesh components; 0 disables")
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output or root.parent / "tsdf_mesh.ply"
    intrinsic = read_intrinsics(root / "intrinsics.txt")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    with (root / "frames.csv").open(newline="") as handle:
        for frame in csv.DictReader(handle):
            color = o3d.io.read_image(str(root / frame["color"]))
            depth = o3d.io.read_image(str(root / frame["depth"]))
            pose_cw = np.loadtxt(root / frame["pose_cw"], dtype=np.float64)
            if pose_cw.shape != (4, 4):
                raise ValueError(f"invalid 4x4 world-to-camera pose: {frame['pose_cw']}")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
            )
            volume.integrate(rgbd, intrinsic, pose_cw)
            integrated += 1

    if not integrated:
        raise RuntimeError("no TSDF frames found; run Gaussian-LIC with tsdf_export: true")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if args.min_cluster_triangles > 0 and len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[labels] < args.min_cluster_triangles)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_normals=True, write_vertex_colors=True):
        raise RuntimeError(f"failed to write {output}")
    print(f"Integrated {integrated} frames; wrote {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles to {output}")


if __name__ == "__main__":
    main()
