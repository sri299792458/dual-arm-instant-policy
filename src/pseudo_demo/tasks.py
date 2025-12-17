
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from .se3 import make_T, rot_x, rot_y, rot_z, interp_SE3, bezier_SE3, jitter_T
from .shapes import (
    sample_box_surface, sample_cylinder_surface, sample_ellipsoid_surface,
    sample_table_plane, transform_points, downsample
)

@dataclass
class Workspace:
    x_min: float = 0.05
    x_max: float = 0.45
    y_min: float = -0.25
    y_max: float = 0.25
    z_table: float = 0.82
    z_min: float = 0.825
    z_max: float = 0.88

@dataclass
class ArmStarts:
    # Roughly “in front of table”, facing inward.
    # You can replace these with your robot's nominal home poses.
    T_left_start: np.ndarray
    T_right_start: np.ndarray

def default_arm_starts() -> ArmStarts:
    # Left: approach from +y, Right: approach from -y (mirrored)
    # Orient grippers to face downward.
    T_left = make_T(rot_x(np.pi), np.array([0.30, 0.20, 1.20], dtype=np.float64))
    T_right = make_T(rot_x(np.pi), np.array([0.30,-0.20, 1.20], dtype=np.float64))
    return ArmStarts(T_left_start=T_left, T_right_start=T_right)

def sample_object(rng, shape: Optional[str]=None):
    if shape is None:
        shape = rng.choice(['box', 'cyl', 'ell'])
    if shape == 'box':
        hx = rng.uniform(0.04, 0.10)
        hy = rng.uniform(0.10, 0.18)  # tray-like (long in y)
        hz = rng.uniform(0.003, 0.010)
        pts = sample_box_surface((hx, hy, hz), n=12000, rng=rng)
        props = dict(shape='box', hx=hx, hy=hy, hz=hz)
    elif shape == 'cyl':
        r = rng.uniform(0.03, 0.06)
        h = rng.uniform(0.08, 0.18)
        pts = sample_cylinder_surface(r, h, n=12000, rng=rng)
        props = dict(shape='cyl', r=r, h=h)
    else:
        rx = rng.uniform(0.04, 0.08)
        ry = rng.uniform(0.08, 0.14)
        rz = rng.uniform(0.02, 0.06)
        pts = sample_ellipsoid_surface((rx,ry,rz), n=12000, rng=rng)
        props = dict(shape='ell', rx=rx, ry=ry, rz=rz)
    return pts, props

def sample_object_pose(rng, ws: Workspace):
    x = rng.uniform(ws.x_min, ws.x_max)
    y = rng.uniform(ws.y_min, ws.y_max)
    z = rng.uniform(ws.z_min, ws.z_max)
    yaw = rng.uniform(-np.pi/6, np.pi/6)
    # Keep roll/pitch small so the object sits on the table
    roll = rng.uniform(-np.deg2rad(5), np.deg2rad(5))
    pitch = rng.uniform(-np.deg2rad(5), np.deg2rad(5))
    Rm = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    return make_T(Rm, np.array([x,y,z], dtype=np.float64)), dict(x=x,y=y,z=z,roll=roll,pitch=pitch,yaw=yaw)

def bimanual_grasp_transforms_for_tray(props, rng):
    """Return constant object->gripper transforms for a cooperative 'two-side grasp'.

    We choose grippers to approach from +/-Y in the *object frame* and face downward.
    """
    # “Grip points” slightly inside the object boundary
    if props['shape'] == 'box':
        hy = props['hy']
        inset = rng.uniform(0.03, 0.05)
        y = hy - inset
    else:
        # for non-trays, just pick a symmetric offset
        y = rng.uniform(0.06, 0.10)

    # Gripper orientation: z-axis down. X-axis along object x.
    R_down = rot_x(np.pi)  # flips z
    # Right hand at -y, left hand at +y
    T_obj_left  = make_T(R_down @ rot_z(0.0), np.array([0.0,  y, 0.0], dtype=np.float64))
    T_obj_right = make_T(R_down @ rot_z(np.pi), np.array([0.0, -y, 0.0], dtype=np.float64))
    return T_obj_left, T_obj_right, dict(grasp_y=y)

def make_pregrasp(T_grasp, offset=0.09):
    """Move backwards along the gripper's +Y axis in its local frame (approach direction)."""
    # In world frame, the local +Y is T[:3,:3] @ [0,1,0]
    y_axis = T_grasp[:3,:3] @ np.array([0,1,0], dtype=np.float64)
    T = T_grasp.copy()
    T[:3,3] = T_grasp[:3,3] + offset * y_axis
    return T

def object_trajectory_waypoints(T0, rng, ws: Workspace, n_wp=4):
    """Sample object motion waypoints for cooperative manipulation."""
    wps = [T0]
    for _ in range(n_wp-1):
        # small translation and yaw change (tray-like movements)
        dp = rng.normal(0.0, 0.06, size=(3,))
        dp[2] = abs(rng.normal(0.0, 0.10))  # often lift
        p = np.clip(wps[-1][:3,3] + dp,
                    [ws.x_min, ws.y_min, ws.z_table+0.02],
                    [ws.x_max, ws.y_max, 1.25])
        dyaw = rng.normal(0.0, np.deg2rad(20))
        Rm = R.from_matrix(wps[-1][:3,:3]) * R.from_euler('z', dyaw)
        wps.append(make_T(Rm.as_matrix(), p))
    return wps

def stitch(*chunks):
    return np.concatenate(chunks, axis=0)

def make_grip_signal(T, close_from_idx):
    grips = np.ones((T,), dtype=np.int8)
    grips[close_from_idx:] = 0
    return grips

def cooperative_rigid(rng, ws: Workspace, starts: ArmStarts,
                      n_table=2500, n_obj=3500, T_approach=90, T_pregrasp=15, T_manip=120):
    """Generate a cooperative rigid bimanual demo by sampling an object trajectory first,
    then deriving both gripper trajectories from fixed object->gripper attachments.
    """
    obj_pts, obj_props = sample_object(rng, shape='box')  # tray-like easiest
    T_w_obj0, pose_meta = sample_object_pose(rng, ws)
    T_obj_left, T_obj_right, grasp_meta = bimanual_grasp_transforms_for_tray(obj_props, rng)

    # Object waypoints & trajectory
    wps = object_trajectory_waypoints(T_w_obj0, rng, ws, n_wp=rng.integers(3,6))
    obj_traj = []
    segs = []
    for i in range(len(wps)-1):
        seg = interp_SE3(wps[i], wps[i+1], num=max(10, T_manip//(len(wps)-1)))
        segs.append(seg)
    obj_traj = np.concatenate(segs, axis=0)  # (T_manip,4,4)

    # Grasp poses at start of manipulation
    T_w_left_grasp0  = obj_traj[0] @ T_obj_left
    T_w_right_grasp0 = obj_traj[0] @ T_obj_right

    # Pregrasp and approach arcs
    pre_offset_l = rng.uniform(0.08, 0.10)
    pre_offset_r = rng.uniform(0.08, 0.10)
    T_w_left_pre  = make_pregrasp(T_w_left_grasp0,  offset=pre_offset_l)
    T_w_right_pre = make_pregrasp(T_w_right_grasp0, offset=pre_offset_r)

    bow_l = float(rng.uniform(0.10, 0.30))
    bow_r = float(rng.uniform(0.10, 0.30))
    up_l = rng.normal(0,1,size=(3,)); up_l /= (np.linalg.norm(up_l)+1e-9)
    up_r = rng.normal(0,1,size=(3,)); up_r /= (np.linalg.norm(up_r)+1e-9)

    left_appr  = bezier_SE3(starts.T_left_start,  T_w_left_pre,  num=T_approach, bow=bow_l, up=up_l)
    right_appr = bezier_SE3(starts.T_right_start, T_w_right_pre, num=T_approach, bow=bow_r, up=up_r)

    left_pre2grasp  = interp_SE3(T_w_left_pre,  T_w_left_grasp0,  num=T_pregrasp)
    right_pre2grasp = interp_SE3(T_w_right_pre, T_w_right_grasp0, num=T_pregrasp)

    # During manipulation, grippers are rigidly attached to the object
    left_manip  = obj_traj @ T_obj_left
    right_manip = obj_traj @ T_obj_right

    # Optional small timing mismatch in approach (teach lead/lag)
    lag = int(rng.integers(-12, 13))
    if lag != 0:
        # shift right trajectory in approach by repeating endpoints
        if lag > 0:
            right_appr = np.concatenate([np.repeat(right_appr[:1], lag, axis=0), right_appr], axis=0)[:T_approach]
        else:
            lag = -lag
            left_appr = np.concatenate([np.repeat(left_appr[:1], lag, axis=0), left_appr], axis=0)[:T_approach]

    T_w_left = stitch(left_appr, left_pre2grasp, left_manip)
    T_w_right = stitch(right_appr, right_pre2grasp, right_manip)
    T_total = T_w_left.shape[0]

    # Gripper states: close at start of manipulation (after pre2grasp)
    close_idx = T_approach + T_pregrasp
    grips_left = make_grip_signal(T_total, close_idx)
    grips_right = make_grip_signal(T_total, close_idx)

    # Build point clouds per timestep: table + moving object
    table = sample_table_plane(x_range=(0.0,0.6), y_range=(-0.35,0.35), z=ws.z_table, n=n_table, rng=rng)
    obj_pts_ds = downsample(obj_pts, n=n_obj, rng=rng)
    pcds = np.zeros((T_total, n_table+n_obj, 3), dtype=np.float32)
    for t in range(T_total):
        # object: static during approach+pre2grasp, then follows obj_traj
        if t < close_idx:
            T_w_obj = T_w_obj0
        else:
            T_w_obj = obj_traj[t - close_idx]
        obj_w = transform_points(T_w_obj, obj_pts_ds)
        pts = np.concatenate([table, obj_w], axis=0)
        # add tiny sensor noise
        pts = pts + rng.normal(0.0, 0.001, size=pts.shape)
        pcds[t] = pts.astype(np.float32)

    meta = dict(
        task_type='cooperative_rigid',
        obj=obj_props,
        obj_pose=pose_meta,
        grasp=grasp_meta,
        timing=dict(T_approach=T_approach, T_pregrasp=T_pregrasp, T_manip=int(obj_traj.shape[0]), lag=int(lag)),
    )

    return dict(
        pcds=pcds,
        T_w_es_left=T_w_left.astype(np.float32),
        T_w_es_right=T_w_right.astype(np.float32),
        grips_left=grips_left,
        grips_right=grips_right,
        meta=meta,
        extras=dict(T_w_obj0=T_w_obj0.astype(np.float32), obj_traj=obj_traj.astype(np.float32))
    )

def independent_two_objects(rng, ws: Workspace, starts: ArmStarts,
                           n_table=2500, n_obj=2000, T_total=220):
    """Two unrelated single-arm manipulations (teaches weak coupling)."""
    # Two objects placed left/right in Y
    objA_pts, objA_props = sample_object(rng, shape=rng.choice(['box','ell','cyl']))
    objB_pts, objB_props = sample_object(rng, shape=rng.choice(['box','ell','cyl']))

    # Fix poses with separation
    T_w_objA, metaA = sample_object_pose(rng, ws)
    T_w_objB, metaB = sample_object_pose(rng, ws)
    T_w_objA[:3,3][1] = float(np.clip(metaA['y'] + 0.12, ws.y_min, ws.y_max))
    T_w_objB[:3,3][1] = float(np.clip(metaB['y'] - 0.12, ws.y_min, ws.y_max))

    # Left arm manipulates A, Right manipulates B
    # Build simple approach -> grasp -> lift trajectories
    def single_arm_plan(T_start, T_obj, side_sign):
        # grasp offset on object y side
        y = side_sign * 0.08
        R_down = rot_x(np.pi)
        T_obj_g = make_T(R_down, np.array([0.0, y, 0.0], dtype=np.float64))
        T_grasp = T_obj @ T_obj_g
        T_pre = make_pregrasp(T_grasp, offset=float(rng.uniform(0.08,0.10)))
        appr = bezier_SE3(T_start, T_pre, num=90, bow=float(rng.uniform(0.10,0.30)),
                          up=rng.normal(0,1,size=(3,)))
        pre2 = interp_SE3(T_pre, T_grasp, num=15)
        # lift
        T_end = T_grasp.copy()
        T_end[2,3] = float(rng.uniform(1.05, 1.25))
        manip = interp_SE3(T_grasp, T_end, num=T_total - (90+15))
        T = np.concatenate([appr, pre2, manip], axis=0)
        grips = np.ones((T_total,), dtype=np.int8)
        grips[90+15:] = 0
        return T, grips, T_obj_g

    T_left, g_left, T_objA_g = single_arm_plan(starts.T_left_start,  T_w_objA, side_sign=+1.0)
    T_right,g_right,T_objB_g = single_arm_plan(starts.T_right_start, T_w_objB, side_sign=-1.0)

    table = sample_table_plane(x_range=(0.0,0.6), y_range=(-0.35,0.35), z=ws.z_table, n=n_table, rng=rng)
    objA_ds = downsample(objA_pts, n=n_obj, rng=rng)
    objB_ds = downsample(objB_pts, n=n_obj, rng=rng)

    pcds = np.zeros((T_total, n_table+2*n_obj, 3), dtype=np.float32)
    for t in range(T_total):
        # move each object only after its grasp closes
        if g_left[t] == 0:
            # approximate: object follows left gripper with fixed attachment
            T_w_objA_t = T_left[t] @ np.linalg.inv(T_objA_g)
        else:
            T_w_objA_t = T_w_objA
        if g_right[t] == 0:
            T_w_objB_t = T_right[t] @ np.linalg.inv(T_objB_g)
        else:
            T_w_objB_t = T_w_objB

        A_w = transform_points(T_w_objA_t, objA_ds)
        B_w = transform_points(T_w_objB_t, objB_ds)
        pts = np.concatenate([table, A_w, B_w], axis=0)
        pts = pts + rng.normal(0.0, 0.001, size=pts.shape)
        pcds[t] = pts.astype(np.float32)

    meta = dict(
        task_type='independent',
        objA=objA_props, objA_pose=metaA,
        objB=objB_props, objB_pose=metaB,
        timing=dict(T_total=int(T_total))
    )

    return dict(
        pcds=pcds,
        T_w_es_left=T_left.astype(np.float32),
        T_w_es_right=T_right.astype(np.float32),
        grips_left=g_left,
        grips_right=g_right,
        meta=meta,
        extras=dict()
    )

def handover(rng, ws: Workspace, starts: ArmStarts,
             n_table=2500, n_obj=3000, T_phase=(80, 40, 80), overlap=18):
    """Left picks object and hands to right. Includes a brief overlap where both are closed."""
    obj_pts, obj_props = sample_object(rng, shape=rng.choice(['box','ell','cyl']))
    T_w_obj0, pose_meta = sample_object_pose(rng, ws)

    # Choose a handover point above the table, near center
    p_h = np.array([rng.uniform(ws.x_min, ws.x_max),
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(0.95, 1.10)], dtype=np.float64)
    R_down = rot_x(np.pi)

    # Left grasp on object, then move to handover point
    T_obj_left = make_T(R_down, np.array([0.0, 0.08, 0.0], dtype=np.float64))
    T_left_grasp0 = T_w_obj0 @ T_obj_left
    T_left_pre = make_pregrasp(T_left_grasp0, offset=float(rng.uniform(0.08,0.10)))
    left_appr = bezier_SE3(starts.T_left_start, T_left_pre, num=T_phase[0], bow=float(rng.uniform(0.1,0.3)),
                           up=rng.normal(0,1,size=(3,)))
    left_pre2 = interp_SE3(T_left_pre, T_left_grasp0, num=T_phase[1])
    # Move object (carried by left) to handover point
    T_left_handover = make_T(R_down, p_h)
    left_to_h = interp_SE3(T_left_grasp0, T_left_handover, num=T_phase[2])

    # Right approaches handover point
    T_right_pre = make_pregrasp(T_left_handover @ make_T(rot_z(np.pi), np.zeros(3)), offset=float(rng.uniform(0.08,0.10)))
    right_appr = bezier_SE3(starts.T_right_start, T_right_pre, num=T_phase[0], bow=float(rng.uniform(0.1,0.3)),
                            up=rng.normal(0,1,size=(3,)))
    right_pre2 = interp_SE3(T_right_pre, T_left_handover @ make_T(rot_z(np.pi), np.zeros(3)), num=T_phase[1])
    right_to_h = interp_SE3(right_pre2[-1], right_pre2[-1], num=T_phase[2])  # stay near handover

    # Align lengths
    T_left = np.concatenate([left_appr, left_pre2, left_to_h], axis=0)
    T_right = np.concatenate([right_appr, right_pre2, right_to_h], axis=0)
    T_total = T_left.shape[0]

    # Gripper signals:
    # left: open -> close at grasp, remain closed until end-overlap, then open
    # right: open until overlap start, then close
    grips_left = np.ones((T_total,), dtype=np.int8)
    grips_right = np.ones((T_total,), dtype=np.int8)
    grasp_close = T_phase[0] + T_phase[1]
    overlap_start = T_total - overlap
    grips_left[grasp_close:] = 0
    grips_right[overlap_start:] = 0
    # left releases near the very end
    grips_left[-(overlap//2):] = 1

    # Object pose:
    # - before left grasps: fixed at T_w_obj0
    # - after grasp_close: attached to left (T_left @ inv(T_obj_left))
    # - during overlap: stay attached to left (simpler); you can blend to right if you want
    obj_ds = downsample(obj_pts, n=n_obj, rng=rng)
    table = sample_table_plane(x_range=(0.0,0.6), y_range=(-0.35,0.35), z=ws.z_table, n=n_table, rng=rng)
    pcds = np.zeros((T_total, n_table+n_obj, 3), dtype=np.float32)

    T_left_obj = np.linalg.inv(T_obj_left)
    for t in range(T_total):
        if t < grasp_close:
            T_w_obj = T_w_obj0
        else:
            T_w_obj = T_left[t] @ T_left_obj
        pts_obj = transform_points(T_w_obj, obj_ds)
        pts = np.concatenate([table, pts_obj], axis=0)
        pts = pts + rng.normal(0.0, 0.001, size=pts.shape)
        pcds[t] = pts.astype(np.float32)

    meta = dict(
        task_type='handover',
        obj=obj_props,
        obj_pose=pose_meta,
        handover_point=p_h.tolist(),
        timing=dict(T_phase=list(T_phase), overlap=int(overlap))
    )

    return dict(
        pcds=pcds,
        T_w_es_left=T_left.astype(np.float32),
        T_w_es_right=T_right.astype(np.float32),
        grips_left=grips_left,
        grips_right=grips_right,
        meta=meta,
        extras=dict()
    )
