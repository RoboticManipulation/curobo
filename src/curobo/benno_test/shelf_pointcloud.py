# Third Party
import numpy as np
import open3d as o3d

def box_surface_pointcloud(center_xyz, size_xyz, step):
    """Create a surface pointcloud for an axis-aligned box."""
    size_xyz = np.asarray(size_xyz, dtype=float).reshape(3)
    center = np.asarray(center_xyz, dtype=float).reshape(1, 3)
    half = 0.5 * size_xyz

    xs = np.arange(-half[0], half[0] + 1e-6, step)
    ys = np.arange(-half[1], half[1] + 1e-6, step)
    zs = np.arange(-half[2], half[2] + 1e-6, step)

    points = []
    # +/- X faces
    ys_grid, zs_grid = np.meshgrid(ys, zs, indexing="xy")
    points.append(
        np.stack((np.full_like(ys_grid, -half[0]), ys_grid, zs_grid), axis=-1).reshape(-1, 3)
    )
    points.append(
        np.stack((np.full_like(ys_grid, half[0]), ys_grid, zs_grid), axis=-1).reshape(-1, 3)
    )
    # +/- Y faces
    xs_grid, zs_grid = np.meshgrid(xs, zs, indexing="xy")
    points.append(
        np.stack((xs_grid, np.full_like(xs_grid, -half[1]), zs_grid), axis=-1).reshape(-1, 3)
    )
    points.append(
        np.stack((xs_grid, np.full_like(xs_grid, half[1]), zs_grid), axis=-1).reshape(-1, 3)
    )
    # +/- Z faces
    xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")
    points.append(
        np.stack((xs_grid, ys_grid, np.full_like(xs_grid, -half[2])), axis=-1).reshape(-1, 3)
    )
    points.append(
        np.stack((xs_grid, ys_grid, np.full_like(xs_grid, half[2])), axis=-1).reshape(-1, 3)
    )

    pc = np.concatenate(points, axis=0)
    return pc + center


def shelf_pointcloud_parts(
    center_xyz=(0.0, 0.0, 0.0),
    width=0.6,
    depth=0.4,
    height=0.6,
    board_thickness=0.02,
    step=0.01,
):
    """Create a shelf pointcloud as top, bottom, left, right boards.

    Coordinate convention:
    - X: shelf width (left/right)
    - Y: shelf depth (front/back)
    - Z: shelf height (up/down)
    """
    if board_thickness * 2.0 >= height:
        raise ValueError("board_thickness too large for shelf height")

    center = np.asarray(center_xyz, dtype=float).reshape(3)
    inner_height = height - 2.0 * board_thickness

    top_center = center + np.array([0.0, 0.0, 0.5 * height - 0.5 * board_thickness])
    bottom_center = center + np.array([0.0, 0.0, -0.5 * height + 0.5 * board_thickness])
    left_center = center + np.array([-0.5 * width + 0.5 * board_thickness, 0.0, 0.0])
    right_center = center + np.array([0.5 * width - 0.5 * board_thickness, 0.0, 0.0])

    top = box_surface_pointcloud(
        top_center, [width, depth, board_thickness], step=step
    )
    bottom = box_surface_pointcloud(
        bottom_center, [width, depth, board_thickness], step=step
    )
    left = box_surface_pointcloud(
        left_center, [board_thickness, depth, inner_height], step=step
    )
    right = box_surface_pointcloud(
        right_center, [board_thickness, depth, inner_height], step=step
    )

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def shelf_pointcloud(
    center_xyz=(0.0, 0.0, 0.0),
    width=0.6,
    depth=0.4,
    height=0.6,
    board_thickness=0.02,
    step=0.01,
):
    parts = shelf_pointcloud_parts(
        center_xyz=center_xyz,
        width=width,
        depth=depth,
        height=height,
        board_thickness=board_thickness,
        step=step,
    )
    return np.concatenate(list(parts.values()), axis=0)


def shelf_mesh_from_pointcloud(pointcloud, pitch=0.02, name="shelf"):
    """Convert the shelf pointcloud into a Mesh for motion planning."""
    from curobo.geom.types import Mesh

    return Mesh.from_pointcloud(pointcloud, pitch=pitch, name=name)


def visualize_pointcloud(pointcloud):
    """Visualize a pointcloud with Open3D."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for visualization") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.astype(float))
    o3d.visualization.draw_geometries([pcd])


def main():
    step = 0.01
    pitch = 0.02

    parts = shelf_pointcloud_parts(
        center_xyz=(0.4, 0.0, 0.3),
        width=0.6,
        depth=0.4,
        height=0.3,
        board_thickness=0.02,
        step=step,
    )
    shelf_pc = np.concatenate(list(parts.values()), axis=0)
    shelf_mesh = shelf_mesh_from_pointcloud(shelf_pc, pitch=pitch, name="shelf")

    print("Pointcloud shape:", shelf_pc.shape)
    print("Mesh:", shelf_mesh)
    visualize_pointcloud(shelf_pc)


if __name__ == "__main__":
    main()
