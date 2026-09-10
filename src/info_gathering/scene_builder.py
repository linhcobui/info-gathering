"""Build the shell-game MuJoCo XML from a config dict."""
import os

_CONTACTS = "\n".join(
    f'      <geom name="{{cup}}_c{i}" mesh="contact{i}" class="object_col"/>'
    for i in range(12)
)


def _cup_body(name, x, rest_z, rgba):
    r = " ".join(str(v) for v in rgba)
    contacts = _CONTACTS.format(cup=name)
    return f'''    <body name="{name}" mocap="true" pos="{x} 0 {rest_z}" euler="3.14159 0 0" childclass="grab">
      <geom name="{name}_visual" mesh="cup" material="cup_mat" rgba="{r}"/>
{contacts}
    </body>'''


def build_scene_xml(cfg, meshdir_abs):
    spacing = cfg["cup_spacing"]
    rest_z = cfg["cup_rest_z"]
    slots_x = [(s - 1) * spacing for s in range(3)]

    if cfg["cup_color_mode"] == "same":
        colors = [cfg["cup_color_same"]] * 3
    else:
        colors = cfg["cup_colors_distinct"]

    cups = "\n".join(_cup_body(f"cup{k}", slots_x[k], rest_z, colors[k]) for k in range(3))
    res = cfg["resolution"]
    br = cfg["ball_radius"]
    bz = cfg["ball_z"]
    brgba = " ".join(str(v) for v in cfg["ball_rgba"])

    cam_pos = cfg.get("camera_pos", [0, -0.6, 0.32])
    cam_xyaxes = cfg.get("camera_xyaxes", [1, 0, 0, 0, 0.5, 0.87])
    cam_pos_s = " ".join(str(v) for v in cam_pos)
    cam_xyaxes_s = " ".join(str(v) for v in cam_xyaxes)

    floor_texrepeat = cfg.get("floor_texrepeat", 5)
    sky_rgb1 = " ".join(str(v) for v in cfg.get("sky_rgb1", [0.3, 0.5, 0.7]))
    sky_rgb2 = " ".join(str(v) for v in cfg.get("sky_rgb2", [0.0, 0.0, 0.0]))

    return f'''<mujoco model="shell_game">
  <compiler angle="radian" meshdir="{meshdir_abs}"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <default class="grab">
      <joint limited="false" margin="0.01" armature="0.001" damping="0" frictionloss="0.001"/>
      <geom type="mesh" rgba=".93 .99 .97 1.0"/>
      <site size="0.005 0 0" rgba="0.4 0.9 0.4 1"/>
      <default class="object_col">
        <geom type="mesh" density="1250" contype="1" conaffinity="1"
              friction="1 0.5 0.01" margin="0.0005" condim="4"
              rgba=".3 .4 .5 1" group="3"/>
      </default>
    </default>
  </default>

  <visual>
    <global offheight="{res}" offwidth="{res}"/>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.5 0.5 0.5" specular="0.0 0.0 0.0"/>
    <quality shadowsize="4096" offsamples="16"/>
    <map force="0.1" znear="0.01"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <include file="{os.path.join(meshdir_abs, 'cup', 'assets.xml')}"/>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="{sky_rgb1}" rgb2="{sky_rgb2}" width="512" height="3072"/>
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="floor_mat" texture="groundplane" texuniform="true"
              texrepeat="{floor_texrepeat} {floor_texrepeat}" reflectance="0.2"/>
    <material name="cup_mat" specular="0.0" shininess="0.0" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light name="key" directional="false" pos="0.3 -0.4 0.8"
           dir="-0.3 0.4 -0.8" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"
           castshadow="true"/>
    <light name="fill" directional="false" pos="-0.4 -0.3 0.6"
           dir="0.4 0.3 -0.6" diffuse="0.3 0.3 0.35" specular="0.0 0.0 0.0"
           castshadow="false"/>

    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" material="floor_mat"/>
    <camera name="head_on" pos="{cam_pos_s}" xyaxes="{cam_xyaxes_s}"/>

    <body name="ball" pos="{slots_x[0]} 0 {bz}">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{br}" mass="0.0027"
            rgba="{brgba}" friction="0.4 0.02 0.001"
            contype="1" conaffinity="1" condim="4"/>
    </body>

{cups}
  </worldbody>
</mujoco>'''
