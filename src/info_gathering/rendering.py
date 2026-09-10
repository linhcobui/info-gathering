"""Shared rendering: build the model, pose bodies kinematically, render frames."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

from .scene_builder import build_scene_xml


def build_model(cfg, meshdir_abs):
    """Compile the scene and return (model, data, handles)."""
    xml = build_scene_xml(cfg, meshdir_abs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    handles = {
        "cup_mocap": [model.body(f"cup{k}").mocapid[0] for k in range(3)],
        "ball_qadr": model.jnt_qposadr[model.joint("ball_free").id],
    }
    return model, data, handles


def render_episode(model, data, handles, ep, cfg, renderer=None):
    """Kinematically pose cups + ball each frame and render.
    ep: dict from choreograph() with cup_xyz (T,3,3) and ball_xyz (T,3).
    Returns rgb uint8 array (T, res, res, 3)."""
    res = cfg["resolution"]
    cam = cfg["camera"]
    cup_mocap = handles["cup_mocap"]
    bq = handles["ball_qadr"]
    T = ep["cup_xyz"].shape[0]

    own_renderer = renderer is None
    if own_renderer:
        renderer = mujoco.Renderer(model, height=res, width=res)

    frames = np.empty((T, res, res, 3), dtype=np.uint8)
    try:
        for t in range(T):
            for k in range(3):
                data.mocap_pos[cup_mocap[k]] = ep["cup_xyz"][t, k]
            data.qpos[bq:bq + 3] = ep["ball_xyz"][t]
            data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)  # kinematic pose, no physics
            renderer.update_scene(data, camera=cam)
            frames[t] = renderer.render()
    finally:
        if own_renderer:
            renderer.close()
    return frames
