# Third Party
import numpy as np

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def cube_pointcloud(center_xyz, size, step):
    half = size * 0.5
    coords = np.arange(-half, half + 1e-6, step)
    xs, ys = np.meshgrid(coords, coords, indexing="xy")

    points = []
    # +/- X faces
    points.append(np.stack((np.full_like(xs, -half), xs, ys), axis=-1))
    points.append(np.stack((np.full_like(xs, half), xs, ys), axis=-1))
    # +/- Y faces
    points.append(np.stack((xs, np.full_like(xs, -half), ys), axis=-1))
    points.append(np.stack((xs, np.full_like(xs, half), ys), axis=-1))
    # +/- Z faces
    points.append(np.stack((xs, ys, np.full_like(xs, -half)), axis=-1))
    points.append(np.stack((xs, ys, np.full_like(xs, half)), axis=-1))

    pc = np.concatenate(points, axis=0).reshape(-1, 3)
    return pc + np.asarray(center_xyz).reshape(1, 3)


def main():
    pitch = 0.02
    step = 0.01  # should be <= pitch for a clean marching-cubes mesh

    cube1 = cube_pointcloud([0.4, 0.0, 0.2], size=0.2, step=step)
    cube2 = cube_pointcloud([0.3, -0.3, 0.3], size=0.15, step=step)

    mesh1 = Mesh.from_pointcloud(cube1, pitch=pitch, name="cube1")
    mesh2 = Mesh.from_pointcloud(cube2, pitch=pitch, name="cube2")
    world_cfg = WorldConfig(mesh=[mesh1, mesh2])

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "ur5e.yml",
        world_cfg,
        collision_checker_type=CollisionCheckerType.MESH,
        interpolation_dt=0.01,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    retract_cfg = motion_gen.get_retract_config()
    start_state = JointState.from_position(retract_cfg.view(1, -1).clone())

    goal_pose = Pose.from_list([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
    result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))

    print(result)

    print("Success:", result.success)
    if result.success:
        traj = result.get_interpolated_plan()
        print("Trajectory shape:", traj.position.shape)


if __name__ == "__main__":
    main()
