"""Milestone 1: load the shell-game scene, render one head-on frame, save a PNG."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import imageio.v2 as imageio

SCENE = os.path.expanduser("~/info-gathering/assets/scenes/shell_game.xml")
OUT = os.path.expanduser("~/info-gathering/data/test_frame.png")

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

# Cups are up (revealed) so we can see the ball. mocap_pos indexed by mocap body order.
# Raise all three cups to reveal; ball sits at slot 0.
# mocap bodies: cup0, cup1, cup2 -> mocap indices 0,1,2
slots_x = [-0.12, 0.0, 0.12]
up_z = 0.05 + 0.12  # resting z + lift height
for i, x in enumerate(slots_x):
    data.mocap_pos[i] = [x, 0.0, up_z]

# put ball in slot 0
data.qpos[0:3] = [slots_x[0], 0.0, 0.015]
data.qvel[:] = 0

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=224, width=224)
renderer.update_scene(data, camera="head_on")
img = renderer.render()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
imageio.imwrite(OUT, img)
print(f"rendered {img.shape} -> {OUT}")
