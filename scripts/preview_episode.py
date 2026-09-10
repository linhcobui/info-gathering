"""Generate ONE episode and save it as an MP4 you can watch."""
import os, sys, yaml
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from info_gathering.scene_builder import build_scene_xml
from info_gathering.choreography import generate_plan, choreograph

def main():
    cfg_path = os.path.join(ROOT, "configs", "memtest0.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    meshdir_abs = os.path.join(ROOT, "submodules", "object_sim")

    # build scene
    xml = build_scene_xml(cfg, meshdir_abs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # mocap indices for cups (identity order cup0,cup1,cup2)
    cup_mocap = [model.body(f"cup{k}").mocapid[0] for k in range(3)]
    ball_qadr = model.jnt_qposadr[model.joint("ball_free").id]

    # choreograph
    plan = generate_plan(cfg, rng)
    ep = choreograph(cfg, plan, rng)
    T = ep["cup_xyz"].shape[0]
    print(f"episode length T={T} frames, {len(plan)} transitions")

    renderer = mujoco.Renderer(model, height=cfg["resolution"], width=cfg["resolution"])
    frames = []

    try:
        for t in range(T):
            # set cup mocap positions (identity k -> its xyz)
            for k in range(3):
                data.mocap_pos[cup_mocap[k]] = ep["cup_xyz"][t, k]
            # set ball position (teleport during swaps; physics settles between)
            data.qpos[ball_qadr:ball_qadr+3] = ep["ball_xyz"][t]
            data.qpos[ball_qadr+3:ball_qadr+7] = [1, 0, 0, 0]  # quat
            data.qvel[:] = 0
            # a few physics substeps so ball rests on floor / inside cup
            for _ in range(cfg["sim_substeps"]):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cfg["camera"])
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    out = os.path.join(ROOT, cfg["video_path"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=cfg["fps"], quality=8)
    print(f"wrote {out}  ({T} frames @ {cfg['fps']}fps)")

if __name__ == "__main__":
    main()
