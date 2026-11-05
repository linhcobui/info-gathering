"""OGBar single-bar environment.

Adapted from submodules/ogbench/ogbench/manipspace/envs/cube_env.py (ogbench a977253, MIT; see
src/ogbar/LICENSE_OGBENCH). The cube becomes a 4 x 4 x 24 cm massless bar carrying two hidden point masses
(see ogbar.mass); only OGBench's 'data_collection' mode is supported.

Contact physics note: the Robotiq pad geoms have priority=1, so in pad-bar contacts MuJoCo takes condim and
friction from the *pad*. The pivot / torsional-friction regime is therefore set on the pads
(`physics.pad_condim`, `physics.pad_friction`), not on the bar.
"""
import mujoco
import numpy as np
from dm_control import mjcf

from ogbar import lie
from ogbar.envs.manipspace_env import ManipSpaceEnv
from ogbar.mass import BarMass, make_mass, sample_w_right

# Wall planes from descriptions/floor_wall.xml (inner faces).
WALL_X = (0.22, 0.63)
WALL_Y = 0.38
PAD_BODIES = {'left': 'ur5e/robotiq/left_pad', 'right': 'ur5e/robotiq/right_pad'}
# Pad box geometry from 2f85.xml (pad_box1 / pad_box2 defaults): centre y, and the z-range both boxes span.
PAD_CENTRE_Y = -0.0026
PAD_MID_Z = 0.01875
STATE_XYZ_CENTER = np.array([0.425, 0.0, 0.0])   # OGBench's state-observation normalisation
STATE_XYZ_SCALER = 10.0
STATE_GRIPPER_SCALER = 3.0


def _point_rect_distance(p, lo, hi):
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def _point_segment_distance(q, p0, p1):
    v = p1 - p0
    u = np.clip((q - p0) @ v / max(v @ v, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(q - (p0 + u * v)))


def _segment_rect_distance(p0, p1, lo, hi):
    """Distance in the plane between segment p0-p1 and the filled axis-aligned rectangle [lo, hi] (0 if they touch)."""
    # Liang-Barsky clip: does the segment cross the rectangle?
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(2):
        for pk, qk in ((-d[k], p0[k] - lo[k]), (d[k], hi[k] - p0[k])):
            if abs(pk) < 1e-12:
                if qk < 0:
                    t0, t1 = 1.0, 0.0
            else:
                r = qk / pk
                if pk < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
    if t0 <= t1:
        return 0.0
    # Otherwise the closest pair involves a segment endpoint or a rectangle corner.
    corners = [np.array([x, y]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    return min([_point_rect_distance(p0, lo, hi), _point_rect_distance(p1, lo, hi)]
               + [_point_segment_distance(q, p0, p1) for q in corners])


class BarEnv(ManipSpaceEnv):
    """UR5e + Robotiq picking up one bar whose centre of mass is hidden.

    Args:
        bar_cfg: the full data config dict (uses the `bar`, `mass`, `physics`, `debug` blocks).
        **kwargs: forwarded to ManipSpaceEnv (ob_type, width, height, visualize_info, ...).
    """

    def __init__(self, bar_cfg, **kwargs):
        self._cfg = bar_cfg
        self._bar = bar_cfg['bar']
        self._mass_cfg = bar_cfg['mass']
        self._phys = bar_cfg['physics']
        self._ik_tilt_ki = float(self._phys.get('ik_tilt_ki') or 0.0)
        self._min_offscreen = int(bar_cfg['debug']['resolution'])
        self._pixel_camera = (bar_cfg.get('env') or {}).get('pixel_camera', 'front_pixels')   # observation camera
        kwargs.setdefault('mode', 'data_collection')
        assert kwargs['mode'] == 'data_collection', 'BarEnv only implements data_collection mode'
        super().__init__(**kwargs)

        self._plat = bar_cfg.get('platform') or {'enabled': False}
        self._fence = bar_cfg.get('fence') or {'enabled': False}
        self._cradle = bar_cfg.get('cradle') or {'enabled': False}
        self._slot = bar_cfg.get('slot') or {'enabled': False}
        assert sum([self._plat['enabled'], self._fence['enabled'], self._cradle['enabled'], self._slot['enabled']]) <= 1, \
            'enable at most one of platform / fence / cradle / slot'
        # The fence or the cradle splits the floor into two sides; resting poses are sampled per side. The slot sits
        # on the -y side, so floor poses only use the +y side.
        self._div = next((d for d in (self._fence, self._cradle, self._slot) if d['enabled']), None)
        if self._div is not None:
            # Poses are rejection-sampled per side of the fence (_sample_side_pose); these only feed the hidden
            # bounding-box site OGBench draws.
            self._object_sampling_bounds = np.asarray(self._bar['boundary'], dtype=np.float64)
            self._target_sampling_bounds = np.asarray(self._bar['boundary'], dtype=np.float64)
        elif self._plat['enabled']:
            # Bar starts on the floor; targets alternate between the raised platform and the floor.
            self._object_sampling_bounds = np.asarray(self._plat['floor_bounds'], dtype=np.float64)
            self._target_sampling_bounds = np.asarray(self._plat['platform_bounds'], dtype=np.float64)
        else:
            self._object_sampling_bounds = np.asarray(self._bar['object_sampling_bounds'], dtype=np.float64)
            self._target_sampling_bounds = np.asarray(self._bar['target_sampling_bounds'], dtype=np.float64)
        self._check_bounds_clear_walls()

        self._bar_color = np.array(self._bar['rgba'], dtype=np.float64)
        self._target_task = 'bar'
        self._target_block = 0
        self._mass: BarMass | None = None
        self._pending_w_right = None
        self._debug_grasp_s = None
        self._debug_renderer = None

    # ------------------------------------------------------------------ build

    def _check_bounds_clear_walls(self):
        """Every sampled pose must keep the whole bar (any yaw) inside the walls."""
        if self._div is not None:
            return                     # side poses are rejection-sampled to fit; see _sample_side_pose
        half_diag = float(np.hypot(self._bar['half_length'], self._bar['half_width']))
        for b in (self._object_sampling_bounds, self._target_sampling_bounds):
            assert b[0, 0] - half_diag > WALL_X[0] and b[1, 0] + half_diag < WALL_X[1], 'bar can hit an x wall'
            assert max(abs(b[0, 1]), abs(b[1, 1])) + half_diag < WALL_Y, 'bar can hit a y wall'
        if self._plat['enabled']:
            # Whole bar (any yaw) on one side of the platform edge: floor poses at y > edge, platform poses below.
            edge = self._plat['y_range'][1]
            ramp = self._plat.get('ramp') or {'enabled': False}
            floor_edge = edge + (ramp['width'] if ramp['enabled'] else 0.0)
            assert self._object_sampling_bounds[0, 1] - half_diag > floor_edge, 'floor pose can overlap the board/ramp'
            assert self._target_sampling_bounds[1, 1] + half_diag < edge, 'platform pose can overhang its edge'

    def set_tasks(self):
        pass

    def modify_gripper_mjcf(self, gripper_mjcf):
        """Pad contact model.

        'box' (OGBench): two flat boxes per pad. Their contact points spread over the pad face, so the grip resists
        twisting about the closing axis with a couple that scales with grip force; the bar barely tilts.
        'point': the boxes stop colliding and each pad touches the bar through one small sphere at the pad centre.
        Resistance to twisting about the closing axis is then only torsional friction (condim >= 4), set directly by
        `pad_torsional_friction`, while sliding friction still carries the bar's weight.
        """
        p = self._phys
        for side in ('pad_box1', 'pad_box2'):
            geom = gripper_mjcf.find('default', side).geom
            sliding = geom.friction[0] if geom.friction is not None else 1.0
            if p.get('pad_sliding_friction') is not None:
                sliding = p['pad_sliding_friction']
            geom.friction = [sliding, p['pad_torsional_friction'], p['pad_rolling_friction']]
            geom.condim = int(p['pad_condim'])
            if p['pad_contact'] in ('point', 'rod'):
                geom.contype = 0
                geom.conaffinity = 0
        if p['pad_contact'] in ('point', 'rod'):
            # 'point': a sphere per pad. 'rod': a short capsule along the pad's z axis, which is vertical when the
            # gripper points down. Against a horizontal round bar, crossed cylinders touch at one point whose normal
            # is horizontal at any contact height; a sphere touching the bar below its axis pushes it upwards and
            # pops it out of the grasp.
            sliding = p.get('pad_sliding_friction') or 0.7
            shape = dict(type='sphere', size=[p['pad_point_radius']]) if p['pad_contact'] == 'point' else \
                dict(type='capsule', size=[p['pad_point_radius'], p['pad_rod_half_length']])
            for body_name in ('left_pad', 'right_pad'):
                gripper_mjcf.find('body', body_name).add(
                    'geom', name=f'{body_name}_point', **shape,
                    pos=[0, PAD_CENTRE_Y, PAD_MID_Z], mass=0, priority=1, group=3,
                    condim=int(p['pad_condim']),
                    friction=[sliding, p['pad_torsional_friction'], p['pad_rolling_friction']],
                    solimp=[0.95, 0.99, 0.001], solref=[p['pad_point_solref'], 1],
                )
        elif p['pad_contact'] != 'box':
            raise ValueError(f"unknown physics.pad_contact: {p['pad_contact']}")
        if p.get('gripper_forcerange') is not None:
            act = gripper_mjcf.find('actuator', 'fingers_actuator')
            act.forcerange = [-p['gripper_forcerange'], p['gripper_forcerange']]

    def add_objects(self, arena_mjcf):
        bar_mjcf = mjcf.from_path((self._desc_dir / 'bar.xml').as_posix())
        hl, hw = self._bar['half_length'], self._bar['half_width']
        bar_default = bar_mjcf.find('default', 'bar').geom
        if self._bar['shape'] == 'box':
            bar_default.size = [hl, hw, hw]
        elif self._bar['shape'] == 'capsule':
            # Cylinder of radius hw with hemispherical caps, hl overall half-length, long axis along local x
            # (MuJoCo capsules run along z, so rotate +90 deg about y). Rounded ends: it cannot stand on an end
            # unless the centre of mass sits exactly at a cap's centre (an exact 80/0 split).
            bar_default.type = 'capsule'
            bar_default.size = [hw, hl - hw]
            for name in ('object_0', 'target_object_0'):
                bar_mjcf.find('geom', name).quat = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
        elif self._bar['shape'] == 'pyramid_box':
            # Square 2hw x 2hw prism with a square pyramid on each end (tips at +-hl on the axis), as one convex mesh.
            # Flat sides: it does not roll. Pointed ends: it cannot stand on an end.
            tip = self._bar['pyramid_length']
            bx = hl - tip
            verts = [[sx * bx, sy * hw, sz * hw] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            verts += [[-hl, 0.0, 0.0], [hl, 0.0, 0.0]]
            bar_mjcf.asset.add('mesh', name='bar_mesh', vertex=np.ravel(verts).tolist())
            bar_default.type = 'mesh'
            bar_default.mesh = 'bar_mesh'
            # Debug markers must clear the square's corners (half-diagonal hw * sqrt 2).
            bar_mjcf.find('geom', 'debug_com').size = [hw * 1.55, 0.006]
            bar_mjcf.find('geom', 'debug_grasp').size = [hw * 1.5, 0.0015]
        else:
            raise ValueError(f"unknown bar.shape: {self._bar['shape']}")
        bar_default.condim = int(self._phys['bar_condim'])
        bar_default.friction = list(self._phys['bar_friction'])
        bar_mjcf.find('geom', 'debug_mass_L').pos = [-self._mass_cfg['offset_d'], 0, 0]
        bar_mjcf.find('geom', 'debug_mass_R').pos = [self._mass_cfg['offset_d'], 0, 0]
        bar_mjcf.find('body', 'object_0').pos[2] = hw
        arena_mjcf.include_copy(bar_mjcf)

        if not self._phys.get('walls', True):
            # No invisible walls: they only kept objects in bounds and the bar got stuck on them. Leaving the
            # workspace ends the episode instead (see bar_out_of_bounds / ogbar.generate).
            for i in range(4):
                wall = arena_mjcf.find('geom', f'wall_{i}')
                wall.contype = 0
                wall.conaffinity = 0
        elif self._phys.get('walls_at_boundary'):
            # OGBench's walls (x 0.22-0.63) are too tight for the 40 cm bar and the slot: move them out to the
            # `bar.boundary` rectangle. They still only collide with the bar (conaffinity 2), not the robot.
            (bx0, by0), (bx1, by1) = self._bar['boundary']
            cx, cy, hx, hy, t = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2, (by1 - by0) / 2, 0.003
            for i, (pos, size) in enumerate((([bx0, cy, 0.1], [t, hy, 0.2]), ([cx, by0, 0.1], [hx, t, 0.2]),
                                             ([bx1, cy, 0.1], [t, hy, 0.2]), ([cx, by1, 0.1], [hx, t, 0.2]))):
                wall = arena_mjcf.find('geom', f'wall_{i}')
                wall.pos, wall.size = pos, size
        if self._phys.get('walls', True) and self._phys.get('wall_friction') is not None:
            # OGBench's invisible workspace walls only keep objects in bounds. Make them slippery (priority wins the
            # contact's friction) so a bar pressed against one is not held there, e.g. on the ramp in a board corner.
            for i in range(4):
                wall = arena_mjcf.find('geom', f'wall_{i}')
                wall.priority = 2
                wall.friction = [self._phys['wall_friction'], 0.0, 0.0]

        if self._plat['enabled']:
            # Raised platform: a box over the left half of the floor (image-left = world -y).
            (x0, x1), (y0, y1), h = self._plat['x_range'], self._plat['y_range'], self._plat['height']
            arena_mjcf.worldbody.add(
                'geom', name='platform', type='box', size=[(x1 - x0) / 2, (y1 - y0) / 2, h / 2],
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, h / 2], rgba=self._plat['rgba'], group=1,
            )
            ramp = self._plat.get('ramp') or {'enabled': False}
            if ramp['enabled']:
                self._add_ramp(arena_mjcf, 'ramp', (x0, x1), y1, +1, ramp, self._plat['rgba'])

        if self._cradle['enabled']:
            # Elevated V cradle ("hot-dog holder"): two narrow plates sloped up from a central apex line along x, on a
            # thin post. Narrow enough that the gripper's fingers pass outside its edges when the bar sits in it.
            cr = self._cradle
            a, L, hw, tk = np.radians(cr['slope_deg']), cr['length'], cr['plate_half_width'], cr['thickness']
            if cr.get('shape', 'v') == 'flat':
                # User: just a flat rectangle on top of the post (top face at apex_z). Narrower than the gap between
                # the closed fingertips, so they pass either side of it.
                fw = cr['flat_half_width']
                arena_mjcf.worldbody.add(
                    'geom', name='cradle_top', type='box', size=[L / 2, fw, tk / 2],
                    pos=[cr['x'], cr['y'], cr['apex_z'] - tk / 2], rgba=cr['rgba'], group=1)
            # Flat top: the post ends just under the plate, so a bar resting on the top never touches the post.
            post_h = cr['apex_z'] - (tk if cr.get('shape', 'v') == 'flat' else 0.0)
            for name, sd in ((('cradle_left', -1), ('cradle_right', 1)) if cr.get('shape', 'v') == 'v' else ()):
                mid = np.array([cr['x'], cr['y'] + sd * hw / 2, cr['apex_z'] + hw / 2 * np.tan(a)])
                n = np.array([0.0, -sd * np.sin(a), np.cos(a)])                 # top-face normal (points into the V)
                ang = sd * a
                arena_mjcf.worldbody.add(
                    'geom', name=name, type='box', size=[L / 2, hw / (2 * np.cos(a)), tk / 2],
                    pos=(mid - n * tk / 2).tolist(), quat=[np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0],
                    rgba=cr['rgba'], group=1)
            arena_mjcf.worldbody.add(
                'geom', name='cradle_post', type='box', size=[cr['post_half_length'], cr['post_half_width'], post_h / 2],
                pos=[cr['x'], cr['y'], post_h / 2], rgba=cr['rgba'], group=1)
            ramp = cr.get('ramp') or {'enabled': False}
            if ramp['enabled']:
                # Same mini ramps as the fence had: a bar resting near the cradle cannot be grasped (the UR5e wrist
                # hits the cradle top), so the floor next to it slopes away and bars roll out to where they can be.
                pw = cr['post_half_width']
                self._add_ramp(arena_mjcf, 'ramp', tuple(ramp['x_range']), cr['y'] + pw, +1, ramp, cr['rgba'])
                self._add_ramp(arena_mjcf, 'ramp_neg', tuple(ramp['x_range']), cr['y'] - pw, -1, ramp, cr['rgba'])

        if self._slot['enabled']:
            self._add_slot(arena_mjcf)

        if self._fence['enabled']:
            # Fence across the middle of the floor (y = fence y), wall to wall: the bar has to be carried over it.
            f = self._fence
            (x0, x1), y0, h, tk = f['x_range'], f['y'], f['height'], f['thickness']
            if f.get('collision', 'mesh') == 'mesh':
                # Same box, but as a convex mesh: MuJoCo's dedicated capsule-box test misses a capsule whose axis
                # passes right through a box thinner than the capsule (reports +1 cm, no contact), so a falling bar
                # could end up lying through the fence. Mesh collisions use general convex collision (GJK/EPA).
                hx, hy = (x1 - x0) / 2, tk / 2
                verts = [[sx * hx, sy * hy, z] for z in (0.0, h) for sy in (-1, 1) for sx in (-1, 1)]
                arena_mjcf.asset.add('mesh', name='fence_mesh', vertex=np.ravel(verts).tolist())
                arena_mjcf.worldbody.add('geom', name='fence', type='mesh', mesh='fence_mesh',
                                         pos=[(x0 + x1) / 2, y0, 0.0], rgba=f['rgba'], group=1)
            else:
                arena_mjcf.worldbody.add(
                    'geom', name='fence', type='box', size=[(x1 - x0) / 2, tk / 2, h / 2],
                    pos=[(x0 + x1) / 2, y0, h / 2], rgba=f['rgba'], group=1,
                )
            ramp = f.get('ramp') or {'enabled': False}
            if ramp['enabled']:
                self._add_ramp(arena_mjcf, 'ramp', (x0, x1), y0 + tk / 2, +1, ramp, f['rgba'])
                self._add_ramp(arena_mjcf, 'ramp_neg', (x0, x1), y0 - tk / 2, -1, ramp, f['rgba'])

        cameras = {  # copied from cube_env.py
            'front': {
                'pos': (1.287, 0.000, 0.509),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            # ogbar: side view D (user, 2026-10-14), from the +y side looking along -y at the slot and floor:
            # free camera lookat (0.425, 0.02, 0.03), distance 0.72, azimuth -90, elevation -28.
            'side_pixels': {
                'pos': (0.425, 0.6557, 0.3680),
                'xyaxes': (-1.000, 0.000, 0.000, 0.000, -0.4695, 0.8829),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def _add_slot(self, arena_mjcf):
        """Raised slot the bar slides into sideways (user design).

        Only two raised end blocks, one at each pyramid tip; nothing between them (no floor, roof or side walls), so the
        fingers can hold the bar anywhere along it. Each block has a horizontal V-groove facing the bar, running along
        y: seen from the side (along y) it is a triangle matching the pyramid tip, `end_clearance` (normal) larger.
        The bar slides in along -y, level and at the right height, both tips entering both grooves at once; a back
        stop on the -y side of each groove ends the slide. Each block = two convex wedges (above / below the groove).
        """
        sl = self._slot
        hl, hw, tip = self._bar['half_length'], self._bar['half_width'], self._bar['pyramid_length']
        x0, y0, clr = sl['x'], sl['y'], sl['end_clearance']
        zc = sl['height'] + hw                                   # bar centre height when seated
        sh = clr * np.hypot(hw, tip) / hw                        # x shift giving `clr` normal clearance on the faces
        t, w, th = sl['wedge_thickness'], sl['back_thickness'], sl['stop_thickness']
        y_lo, y_hi = y0 - hw - clr - th, y0 + hw + sl['entry_extend']
        self._slot_geom_names = []

        def add_prism(name, xz):
            verts = np.array([[x, y, z] for (x, z) in xz for y in (y_lo, y_hi)])
            ctr = verts.mean(0)
            arena_mjcf.asset.add('mesh', name=f'{name}_mesh', vertex=(verts - ctr).ravel().tolist())
            arena_mjcf.worldbody.add('geom', name=name, type='mesh', mesh=f'{name}_mesh', pos=ctr.tolist(),
                                     rgba=sl['rgba'], group=1)
            self._slot_geom_names.append(name)

        for sx, tag in ((-1, 'n'), (1, 'p')):
            if sl.get('end_shape') == 'round':
                # v3 (user): capsule bar with hemispherical tips. Each end block is a half-ring around the tip: seen from
                # the side, a semicircular pocket of radius hw + end_clearance centred on the tip's sphere centre,
                # wall `wedge_thickness` thick, running along y like the v2 grooves (roof over the whole tip, so the
                # bar can only slide in). Built from convex wedge segments; a back stop closes the -y side.
                cxs = x0 + sx * (hl - hw)                        # centre of the tip's hemisphere
                rin, rout, nseg = hw + clr, hw + clr + t, int(sl.get('round_segments', 12))
                angs = np.linspace(np.pi / 2, -np.pi / 2, nseg + 1)
                for k in range(nseg):
                    pts = [(cxs + sx * rr * np.cos(th_), zc + rr * np.sin(th_)) for th_ in (angs[k], angs[k + 1]) for rr in (rin, rout)]
                    add_prism(f'slot_{tag}_arc{k}', pts)
                arena_mjcf.worldbody.add('geom', name=f'slot_{tag}_stop', type='box',
                                         size=[rin / 2, th / 2, rin], pos=[cxs + sx * rin / 2, y_lo + th / 2, zc],
                                         rgba=sl['rgba'], group=1)
                self._slot_geom_names.append(f'slot_{tag}_stop')
                if zc - rout > 1e-6:
                    # Raised slot (user): solid base under the lower half-ring, the block's full depth (y), from the
                    # floor up to the ring's outer surface, so nothing can get underneath. One convex prism per segment.
                    for k in range(nseg // 2, nseg):
                        (xa, za), (xb, zb) = [(cxs + sx * rout * np.cos(th_), zc + rout * np.sin(th_))
                                              for th_ in (angs[k], angs[k + 1])]
                        add_prism(f'slot_{tag}_base{k}', [(xa, za), (xb, zb), (xb, 0.0), (xa, 0.0)])
                continue
            # Horizontal length of each sloped face (user: shorter = bigger gap between the bar's flat end and the block,
            # less slope for a tip to land on). The face keeps the pyramid's angle; only its outer part is kept.
            ls = sl.get('slope_length') or tip
            a = x0 + sx * (hl + sh - ls)                         # inner edge of the sloped face
            b = x0 + sx * (hl + sh)                              # groove apex (the tip's line)
            e = b + sx * w                                       # outer face of the block
            # open_top (user): cut at the groove's peak; only the lower, sloped half is left, so the bar is lowered in.
            for sz, part in (((-1, 'bot'),) if sl.get('open_top') else ((1, 'top'), (-1, 'bot'))):
                z_face, z_out = zc + sz * hw * ls / tip, zc + sz * (hw + t)   # face parallel to the pyramid's, shifted out by sh
                add_prism(f'slot_{tag}_{part}', [(a, z_face), (b, zc), (e, zc), (e, z_out), (a, z_out)])
            if sl.get('end_wall_height'):
                # User: vertical end wall above the cut, at the groove apex line, so a lifted (light) end cannot ride
                # up its slope and out -- which is what let a lopsided bar slide over and drop its heavy end.
                xw = sorted([b, e]); zt = zc + sl['end_wall_height']
                arena_mjcf.worldbody.add('geom', name=f'slot_{tag}_wall', type='box',
                                         size=[(xw[1] - xw[0]) / 2, (y_hi - y_lo) / 2, (zt - zc) / 2],
                                         pos=[(xw[0] + xw[1]) / 2, (y_lo + y_hi) / 2, (zc + zt) / 2],
                                         rgba=sl['rgba'], group=1)
                self._slot_geom_names.append(f'slot_{tag}_wall')
            # Back stop on the far (-y) side of the groove: the bar slides in along -y until its tips meet it.
            xs = sorted([a, b]); zs = (zc - hw - sh, zc if sl.get('open_top') else zc + hw + sh)
            arena_mjcf.worldbody.add('geom', name=f'slot_{tag}_stop', type='box',
                                     size=[(xs[1] - xs[0]) / 2, th / 2, (zs[1] - zs[0]) / 2],
                                     pos=[(xs[0] + xs[1]) / 2, y_lo + th / 2, (zs[0] + zs[1]) / 2], rgba=sl['rgba'], group=1)
            self._slot_geom_names.append(f'slot_{tag}_stop')
            # Post from the floor up to the block.
            z_bot = zc - (hw + t)
            if z_bot <= 1e-6:
                continue                                         # blocks sit on the floor: no post
            arena_mjcf.worldbody.add('geom', name=f'slot_{tag}_post', type='box',
                                     size=[sl['post_half_size'], sl['post_half_size'], z_bot / 2],
                                     pos=[(a + e) / 2 + sx * 0.005, y0, z_bot / 2], rgba=sl['rgba'], group=1)
            self._slot_geom_names.append(f'slot_{tag}_post')

    @staticmethod
    def _add_ramp(arena_mjcf, name, x_range, y_edge, direction, ramp, rgba):
        """Gentle ramp along a tall side at y = y_edge, sloping down towards +y (direction +1) or -y (-1).

        A bar that comes to rest against a tall side rolls back out to where the gripper can reach it (a bar centre
        closer than ~7 cm to a 20 cm side cannot be reached: the gripper housing hits its top edge). A thin box whose
        top face runs from (y_edge, height) down to (y_edge + direction * width, 0); its lower part sinks into the floor.
        """
        (x0, x1) = x_range
        w, th, tk = ramp['width'], np.radians(ramp['angle_deg']), ramp['thickness']
        rh = w * np.tan(th)
        n = np.array([0.0, direction * np.sin(th), np.cos(th)])                 # top-face normal
        centre = np.array([(x0 + x1) / 2, y_edge + direction * w / 2, rh / 2]) - n * tk / 2
        ang = -direction * th
        arena_mjcf.worldbody.add(
            'geom', name=name, type='box', size=[(x1 - x0) / 2, np.hypot(w, rh) / 2, tk / 2],
            pos=centre.tolist(), quat=[np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0], rgba=rgba, group=1,
        )

    def post_compilation_objects(self):
        m = self._model
        if self._phys.get('sliding_friction') is not None:
            # One sliding-friction coefficient for every geom in the scene (bar, floor, fence, ramps, arm), except
            # the gripper's fingers when `keep_finger_friction` is set: they keep their own values (fingertip pads /
            # rods 0.7), and since the fingertips have contact priority the grip on the bar is unchanged.
            # Torsional and rolling coefficients are left as configured.
            keep = np.zeros(m.ngeom, dtype=bool)
            if self._phys.get('keep_finger_friction', False):
                for g in range(m.ngeom):
                    keep[g] = m.body(m.geom_bodyid[g]).name.startswith('ur5e/robotiq/')
            m.geom_friction[~keep, 0] = self._phys['sliding_friction']
        if self._phys.get('disable_midphase', True):
            # MuJoCo's midphase (bounding-volume tree over a body's geoms) drops some bar-fence contacts among the
            # world body's static geoms, so a falling bar could end up lying through the fence with no contact at
            # all. Only a handful of static geoms exist, so checking every candidate pair costs almost nothing.
            m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        if self._phys.get('noslip_iterations'):
            # MuJoCo's no-slip solver pass: removes the slow slip ('creep', ~1-3 deg/s) that soft contacts allow under
            # any sustained tangential load, so a grip that friction can hold stays put instead of slowly pivoting.
            m.opt.noslip_iterations = int(self._phys['noslip_iterations'])
        self._bar_body_id = m.body('object_0').id
        self._bar_geom_id = m.geom('object_0').id
        if self._phys.get('bar_sliding_friction') is not None:
            # User: the bar's own sliding friction, above the scene's. MuJoCo takes the larger of the two geoms'
            # values, so this sets bar-floor / bar-slot contacts; the fingertips (contact priority) are unaffected.
            m.geom_friction[self._bar_geom_id, 0] = self._phys['bar_sliding_friction']
        self._bar_jnt = m.joint('object_joint_0')
        self._bar_qpos_adr = self._bar_jnt.qposadr[0]
        self._bar_qvel_adr = self._bar_jnt.dofadr[0]
        self._target_mocap_id = m.body('object_target_0').mocapid[0]
        self._target_geom_id = m.geom('target_object_0').id
        self._debug_geom_ids = {k: m.geom(f'debug_{k}').id for k in ('mass_L', 'mass_R', 'com', 'grasp')}
        self._pad_body_ids = {side: m.body(name).id for side, name in PAD_BODIES.items()}
        self._floor_geom_id = m.geom('floor').id
        self._platform_geom_id = m.geom('platform').id if self._plat['enabled'] else -1
        self._fence_geom_id = m.geom('fence').id if self._fence['enabled'] else -1
        self._slot_geom_ids = {m.geom(n).id for n in getattr(self, '_slot_geom_names', [])}
        self._cradle_geom_ids = ({m.geom(n).id for n in ('cradle_left', 'cradle_right', 'cradle_top', 'cradle_post')
                                  if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
                                 if self._cradle['enabled'] else set())
        self._ramp_geom_ids = {m.geom(n).id for n in ('ramp', 'ramp_neg') if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
        self._wall_geom_ids = {m.geom(f'wall_{i}').id for i in range(4)}

    # ------------------------------------------------------------------ mass

    def set_mass(self, mass: BarMass):
        """Write the hidden mass distribution into the compiled model (no recompilation)."""
        m, b = self._model, self._bar_body_id
        m.body_mass[b] = mass.total
        m.body_ipos[b] = (mass.com_offset, 0.0, 0.0)
        m.body_iquat[b] = (1.0, 0.0, 0.0, 0.0)
        m.body_inertia[b] = mass.inertia_diag
        # mj_setConst uses its MjData as scratch space and leaves it in a different configuration (this jammed the
        # gripper's closed-loop linkage). Give it a throwaway MjData; with that, it only changes the bar's derived
        # constants (body/dof invweight0, dof_M0, subtree mass).
        if getattr(self, '_scratch_data', None) is None or self._scratch_data.qpos.shape != self._data.qpos.shape:
            self._scratch_data = mujoco.MjData(m)
        mujoco.mj_setConst(m, self._scratch_data)
        self._mass = mass

    @property
    def mass(self) -> BarMass:
        return self._mass

    # ------------------------------------------------------------------ episode

    def reset(self, seed=None, options=None, **kwargs):
        """options may contain 'w_right' (kg) to fix the split (otherwise it is sampled from np_random) and 'bar_pose'
        (x, y, yaw) to fix the bar's start pose on the floor (eval episodes)."""
        self._pending_w_right = None if options is None else options.get('w_right')
        self._pending_bar_pose = None if options is None else options.get('bar_pose')
        return super().reset(seed=seed, options=options, **kwargs)

    def _sample_bar_pose(self, bounds, support_z=0.0):
        xy = self.np_random.uniform(*bounds)
        yaw = self.np_random.uniform(0, 2 * np.pi)
        return np.array([*xy, support_z + self._bar['half_width']]), yaw

    def bar_footprint(self):
        """Seen from above: the bar's axis segment (p0, p1) and the radius r it is widened by."""
        hw, shape = self._bar['half_width'], self._bar['shape']
        r = hw if shape == 'capsule' else np.hypot(hw, hw)
        half_seg = self._bar['half_length'] - {'capsule': hw, 'pyramid_box': self._bar.get('pyramid_length', 0.0)}.get(shape, 0.0)
        # From the joint state (not xpos), so it is valid right after a pose is written, before kinematics run.
        q = self._data.joint('object_joint_0').qpos
        c = q[:3]
        axis = lie.SO3(wxyz=q[3:]).as_matrix()[:, 0]
        return (c - half_seg * axis)[:2], (c + half_seg * axis)[:2], r

    def side_region(self, side):
        """Rectangle (lo, hi) a resting bar must lie within on one side of the fence (+1: y > fence, -1: y < fence)."""
        f, m = self._div, self._div['spawn_margin']
        (bx0, by0), (bx1, by1) = self._bar['boundary']
        if f is self._fence:
            ramp_w = (f.get('ramp') or {}).get('width', 0.0) if (f.get('ramp') or {}).get('enabled') else 0.0
            inner = f['thickness'] / 2 + ramp_w + m
        else:
            ramp = f.get('ramp') or {'enabled': False}
            inner = max(f['clear_half_width'], f['post_half_width'] + ramp['width'] if ramp['enabled'] else 0.0) + m
        outer = min(by1 - m, f['view_half_y'])      # keep the whole resting bar inside the camera's view
        if side > 0:
            return np.array([bx0 + m, f['y'] + inner]), np.array([bx1 - m, outer])
        return np.array([bx0 + m, max(by0 + m, -f['view_half_y'])]), np.array([bx1 - m, f['y'] - inner])

    def bar_side(self):
        """+1 / -1: the whole bar (from above) is on that side of the fence; 0: it straddles the fence."""
        p0, p1, r = self.bar_footprint()
        ys = np.array([p0[1], p1[1]]) - self._fence['y']
        half = self._fence['thickness'] / 2 + r
        if ys.min() > half:
            return 1
        if ys.max() < -half:
            return -1
        return 0

    def _in_camera_view(self, points):
        """All world points project inside every recorded image (front_pixels and the observation camera),
        `view_margin_px` from its edges."""
        return all(self._in_one_camera_view(points, c) for c in sorted({'front_pixels', self._pixel_camera}))

    def _in_one_camera_view(self, points, camera):
        m = self._model
        cam = m.camera(camera).id
        pos = m.cam_pos[cam]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.cam_quat[cam])
        R = R.reshape(3, 3)
        res = self._render_width
        f = 0.5 * res / np.tan(np.radians(m.cam_fovy[cam]) / 2)
        v = (np.asarray(points) - pos) @ R                       # camera frame: x right, y up, looking along -z
        if np.any(v[:, 2] >= 0):
            return False
        u = res / 2 + f * v[:, 0] / -v[:, 2]
        w = res / 2 - f * v[:, 1] / -v[:, 2]
        mg = self._div.get('view_margin_px', 1)
        return bool(np.all((u >= mg) & (u <= res - mg) & (w >= mg) & (w <= res - mg)))

    def _sample_side_pose(self, side):
        """Uniform centre and yaw, rejected until the whole bar (from above) lies inside that side's region and
        the whole bar is inside the camera image."""
        lo, hi = self.side_region(side)
        r = self._bar['half_width']
        half_seg = self._bar['half_length'] - r
        for _ in range(5000):
            xy = self.np_random.uniform(lo, hi)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            d = half_seg * np.array([np.cos(yaw), np.sin(yaw)])
            ends = np.array([xy - d, xy + d])
            if not (np.all(ends.min(0) - r >= lo) and np.all(ends.max(0) + r <= hi)):
                continue
            tips = np.array([[*(xy - d * (1 + r / half_seg)), z] for z in (0.0, 2 * r)]
                            + [[*(xy + d * (1 + r / half_seg)), z] for z in (0.0, 2 * r)])
            if self._div.get('view_check', True) and not self._in_camera_view(tips):
                continue
            return np.array([*xy, r]), yaw
        raise RuntimeError(f'cannot fit the bar on side {side}: region {lo}..{hi}')

    def bar_out_of_bounds(self):
        """The *entire* bar is outside `bar.boundary` (the old OGBench wall rectangle by default), seen from above.

        From above, the capsule is its axis segment widened by its radius r, so it has fully left once the segment is
        more than r from the rectangle. (A box bar uses its half-diagonal cross-section as r: slightly conservative.)
        """
        lo, hi = np.asarray(self._bar['boundary'], dtype=np.float64)
        p0, p1, r = self.bar_footprint()
        return bool(_segment_rect_distance(p0, p1, lo, hi) > r)

    def cradle_seat_z(self):
        """Height of the bar centre when it rests in the V (bar radius r on walls sloped at `slope_deg`)."""
        cr, hw, a = self._cradle, self._bar['half_width'], np.radians(self._cradle['slope_deg'])
        if cr.get('shape', 'v') == 'flat':
            return cr['apex_z'] + hw                       # resting on the flat top (capsule or square bar)
        if self._bar['shape'] == 'capsule':
            return cr['apex_z'] + hw / np.cos(a)
        return cr['apex_z'] + hw * np.tan(a) + hw          # square bar flat side down, its bottom edges on the plates

    def bar_in_slot(self, ignore_fingers=False):
        """Bar seated in the slot: held by the end grooves only, centre within 1 cm of the slot centre, level, not held.
        ignore_fingers: the same, but a fingertip may still be touching it (the gripper is letting go)."""
        if not self._slot['enabled']:
            return False
        c = self.contacts()
        held = (c['left'] or c['right']) and not ignore_fingers
        if held or c['floor'] or not any(self._bar_touches(g) for g in self._slot_geom_ids):
            return False
        sl, p = self._slot, self._data.joint('object_joint_0').qpos[:3]
        tilt = abs(np.arcsin(np.clip(self.bar_axis_world()[2], -1.0, 1.0)))
        return bool(np.hypot(p[0] - sl['x'], p[1] - sl['y']) < 0.01 and tilt < np.radians(3.0))

    def bar_in_cradle(self):
        """Bar resting on the cradle and nothing else (not held, not touching the floor, a ramp or a wall).

        The cradle is a short V (user: ~2 cm long), so this only lasts if the centre of mass is over it.
        """
        if not self._cradle['enabled']:
            return False
        c = self.contacts()
        if not c['cradle'] or c['left'] or c['right'] or c['floor'] or c['ramp'] or c['wall']:
            return False
        if self._cradle.get('shape', 'v') == 'flat':
            # On the flat top only and (nearly) level: a bar leaning on the top's edge and the pillar is not placed
            # (user: no cheap wins without knowing the centre of mass).
            if self._bar_touches(self._model.geom('cradle_post').id):
                return False
            tilt = abs(np.arcsin(np.clip(self.bar_axis_world()[2], -1.0, 1.0)))
            return bool(tilt < np.radians(self._cradle.get('max_rest_tilt_deg', 10.0)))
        return True

    def _bar_touches(self, geom_id):
        for i in range(self._data.ncon):
            k = self._data.contact[i]
            if {k.geom1, k.geom2} == {self._bar_geom_id, geom_id}:
                return True
        return False

    def bar_on_platform(self):
        """Bar centre over the platform and resting at (or above) its height."""
        if not self._plat['enabled']:
            return False
        p = self._data.joint('object_joint_0').qpos[:3]
        (x0, x1), (y0, y1) = self._plat['x_range'], self._plat['y_range']
        return bool(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] > self._plat['height'])

    def initialize_episode(self):
        w_right = self._pending_w_right
        if w_right is None:
            w_right = sample_w_right(self._mass_cfg, self.np_random)
        self.set_mass(make_mass(self._mass_cfg, w_right))

        self._model.geom(self._bar_geom_id).rgba = self._bar_color
        self._model.geom(self._target_geom_id).rgba[:3] = self._bar_color[:3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)
        self._ik_tilt_corr = np.zeros(3)
        if self._slot['enabled']:
            # Start (and the oracle's retreat poses, same bounds) well clear of the slot's top.
            top = self._slot['height'] + 2 * self._bar['half_width'] + self._slot['wedge_thickness']
            self._arm_sampling_bounds[0, 2] = max(self._arm_sampling_bounds[0, 2], top + 0.06)
        self.initialize_arm()

        if getattr(self, '_pending_bar_pose', None) is not None:
            x, y, yaw = self._pending_bar_pose
            pos, yaw = np.array([x, y, self._bar['half_width']]), float(yaw)
        elif self._slot['enabled']:
            pos, yaw = self._sample_side_pose(1)                                 # on the floor, right of the slot
        elif self._div is not None:
            pos, yaw = self._sample_side_pose(self.np_random.choice([-1, 1]))   # start on a random side
        else:
            pos, yaw = self._sample_bar_pose(self._object_sampling_bounds)
        self._data.joint('object_joint_0').qpos[:3] = pos
        self._data.joint('object_joint_0').qpos[3:] = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        mujoco.mj_forward(self._model, self._data)      # fresh contacts: set_new_target asks whether the bar is in the cradle
        self.set_new_target(return_info=False)

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()
        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.0):
        if self._slot['enabled']:
            if self.bar_in_slot():
                pos, yaw = self._sample_side_pose(1)                                    # back to the floor
            else:
                sl = self._slot
                pos = np.array([sl['x'], sl['y'], sl['height'] + self._bar['half_width']])   # into the slot
                yaw = 0.0 if self.np_random.random() < 0.5 else np.pi
        elif self._cradle['enabled']:
            if self.bar_in_cradle():
                pos, yaw = self._sample_side_pose(self.np_random.choice([-1, 1]))   # back to the floor
            else:
                cr = self._cradle
                pos = np.array([cr['x'], cr['y'], self.cradle_seat_z()])            # into the cradle
                yaw = 0.0 if self.np_random.random() < 0.5 else np.pi
        elif self._fence['enabled']:
            # Target on the other side of the fence from the bar (random side if the bar straddles it).
            side = self.bar_side()
            pos, yaw = self._sample_side_pose(-side if side != 0 else self.np_random.choice([-1, 1]))
        elif self._plat['enabled'] and self._plat['alternate'] and self.bar_on_platform():
            pos, yaw = self._sample_bar_pose(self._object_sampling_bounds)                   # back to the floor
        elif self._plat['enabled']:
            pos, yaw = self._sample_bar_pose(self._target_sampling_bounds, self._plat['height'])  # onto the platform
        else:
            pos, yaw = self._sample_bar_pose(self._target_sampling_bounds)
        self._data.mocap_pos[self._target_mocap_id] = pos
        self._data.mocap_quat[self._target_mocap_id] = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        self._model.geom(self._target_geom_id).rgba[3] = 0.2 if self._visualize_info else 0.0
        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _compute_success(self):
        if self._slot['enabled']:
            return self.bar_in_slot()
        if self._cradle['enabled']:
            return self.bar_in_cradle()      # target = the cradle; where along the bar it sits is up to the placer
        obj_pos = self._data.joint('object_joint_0').qpos[:3]
        tar_pos = self._data.mocap_pos[self._target_mocap_id]
        ok = np.linalg.norm(obj_pos - tar_pos) <= self._bar['success_pos_tol']
        if ok and self._bar.get('success_yaw_tol') is not None:
            d = self.bar_yaw() - lie.SO3(wxyz=self._data.mocap_quat[self._target_mocap_id]).compute_yaw_radians()
            d = (d + np.pi / 2) % np.pi - np.pi / 2   # bar is symmetric under 180 deg
            ok = abs(d) <= self._bar['success_yaw_tol']
        return bool(ok)

    def post_step(self):
        self._success = self._compute_success()

    # ------------------------------------------------------------------ privileged state

    def bar_quat(self):
        return self._data.joint('object_joint_0').qpos[3:].copy()

    def bar_yaw(self):
        return lie.SO3(wxyz=self._data.joint('object_joint_0').qpos[3:]).compute_yaw_radians()

    def bar_axis_world(self):
        """Unit vector of the bar's local +x (long) axis in the world frame."""
        return self._data.xmat[self._bar_body_id].reshape(3, 3)[:, 0].copy()

    def bar_point_world(self, s):
        """World position of the point at local x = s on the bar's centreline."""
        return self._data.xpos[self._bar_body_id] + s * self.bar_axis_world()

    def contacts(self):
        """Booleans: bar touching left pad, right pad, floor, any wall."""
        out = {'left': False, 'right': False, 'floor': False, 'wall': False, 'platform': False, 'ramp': False,
               'fence': False, 'cradle': False, 'slot': False}
        g = self._bar_geom_id
        for i in range(self._data.ncon):
            con = self._data.contact[i]
            if con.geom1 == g:
                other = con.geom2
            elif con.geom2 == g:
                other = con.geom1
            else:
                continue
            body = self._model.geom_bodyid[other]
            if body == self._pad_body_ids['left']:
                out['left'] = True
            elif body == self._pad_body_ids['right']:
                out['right'] = True
            elif other == self._floor_geom_id:
                out['floor'] = True
            elif other == self._platform_geom_id:
                out['platform'] = True
            elif other in self._ramp_geom_ids:
                out['ramp'] = True
            elif other == self._fence_geom_id:
                out['fence'] = True
            elif other in self._cradle_geom_ids:
                out['cradle'] = True
            elif other in getattr(self, '_slot_geom_ids', ()):
                out['slot'] = True
            elif other in self._wall_geom_ids:
                out['wall'] = True
        return out

    def add_object_info(self, ob_info):
        d = self._data
        bar_quat = self.bar_quat()
        yaw = self.bar_yaw()
        ob_info['privileged/block_0_pos'] = d.joint('object_joint_0').qpos[:3].copy()
        ob_info['privileged/block_0_quat'] = bar_quat
        ob_info['privileged/block_0_yaw'] = np.array([yaw])

        axis = self.bar_axis_world()
        ob_info['privileged/bar_axis'] = axis
        ob_info['privileged/bar_com_world'] = d.xipos[self._bar_body_id].copy()
        ob_info['privileged/bar_linvel'] = d.qvel[self._bar_qvel_adr:self._bar_qvel_adr + 3].copy()
        ob_info['privileged/bar_angvel'] = d.qvel[self._bar_qvel_adr + 3:self._bar_qvel_adr + 6].copy()
        # Velocity of the centre of mass (freejoint: linear part is the body origin in world frame, angular part is
        # in the body frame). With all the mass at one end the massless far end can whip around quickly, so health
        # checks use this rather than the body-origin velocity.
        omega_world = d.xmat[self._bar_body_id].reshape(3, 3) @ ob_info['privileged/bar_angvel']
        ob_info['privileged/bar_com_linvel'] = ob_info['privileged/bar_linvel'] + np.cross(
            omega_world, d.xipos[self._bar_body_id] - d.xpos[self._bar_body_id])
        ob_info['privileged/bar_yaw_mod_pi'] = np.array([yaw % np.pi])
        # Signed tilt: positive when the bar's local +x end (the W_R end) is higher than the -x end.
        ob_info['privileged/bar_tilt'] = np.array([np.arcsin(np.clip(axis[2], -1.0, 1.0))])

        pinch = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(d.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=d.site_xpos[self._pinch_site_id],
        )
        bar = lie.SE3.from_rotation_and_translation(rotation=lie.SO3(wxyz=bar_quat), translation=d.xpos[self._bar_body_id])
        in_hand = pinch.inverse() @ bar
        ob_info['privileged/bar_in_hand_pose'] = np.concatenate([in_hand.translation(), in_hand.rotation().wxyz])

        c = self.contacts()
        ob_info['privileged/pad_contact_left'] = np.array([c['left']])
        ob_info['privileged/pad_contact_right'] = np.array([c['right']])
        ob_info['privileged/bar_floor_contact'] = np.array([c['floor']])
        ob_info['privileged/bar_wall_contact'] = np.array([c['wall']])
        ob_info['privileged/bar_platform_contact'] = np.array([c['platform']])
        ob_info['privileged/bar_ramp_contact'] = np.array([c['ramp']])
        ob_info['privileged/bar_fence_contact'] = np.array([c['fence']])
        ob_info['privileged/bar_cradle_contact'] = np.array([c['cradle']])
        ob_info['privileged/bar_in_cradle'] = np.array([self.bar_in_cradle()])
        ob_info['privileged/bar_slot_contact'] = np.array([c['slot']])
        ob_info['privileged/bar_in_slot'] = np.array([self.bar_in_slot()])
        if self._fence['enabled']:
            ob_info['privileged/bar_side'] = np.array([self.bar_side()])
            ob_info['privileged/target_side'] = np.array([
                1 if d.mocap_pos[self._target_mocap_id][1] > self._fence['y'] else -1])
        else:
            ob_info['privileged/bar_side'] = np.array([0])
            ob_info['privileged/target_side'] = np.array([0])
        ob_info['privileged/bar_on_platform'] = np.array([self.bar_on_platform()])
        ob_info['privileged/target_on_platform'] = np.array([
            self._plat['enabled'] and self._data.mocap_pos[self._target_mocap_id][2] > self._plat['height']])

        ob_info['privileged/target_task'] = self._target_task
        ob_info['privileged/target_block'] = self._target_block
        ob_info['privileged/target_block_pos'] = d.mocap_pos[self._target_mocap_id].copy()
        ob_info['privileged/target_block_yaw'] = np.array(
            [lie.SO3(wxyz=d.mocap_quat[self._target_mocap_id]).compute_yaw_radians()]
        )

    # ------------------------------------------------------------------ observations

    def compute_state_observation(self):
        """OGBench cube-single's 28-D state vector, computed for the bar (body origin = geometric centre)."""
        info = self.compute_ob_info()
        return np.concatenate([
            info['proprio/joint_pos'],
            info['proprio/joint_vel'],
            (info['proprio/effector_pos'] - STATE_XYZ_CENTER) * STATE_XYZ_SCALER,
            np.cos(info['proprio/effector_yaw']),
            np.sin(info['proprio/effector_yaw']),
            info['proprio/gripper_opening'] * STATE_GRIPPER_SCALER,
            info['proprio/gripper_contact'],
            (info['privileged/block_0_pos'] - STATE_XYZ_CENTER) * STATE_XYZ_SCALER,
            info['privileged/block_0_quat'],
            np.cos(info['privileged/block_0_yaw']),
            np.sin(info['privileged/block_0_yaw']),
        ])

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        return self.compute_state_observation()

    def compute_oracle_observation(self):
        return (self._data.joint('object_joint_0').qpos[:3] - STATE_XYZ_CENTER) * STATE_XYZ_SCALER

    # ------------------------------------------------------------------ debug rendering

    def set_debug_grasp(self, s):
        """Grasp offset to mark with the red tick in debug renders (None hides it)."""
        self._debug_grasp_s = s

    def render_debug(self, camera=None):
        """Render with the hidden mass markers visible. Never used for observations.

        Shows the two point masses (balls sized by mass, W_L blue, W_R orange), the centre of mass (green), the
        current grasp point (red tick) and the target ghost, with the bar shell semi-transparent. All model edits are
        restored before returning, so subsequent observation renders are unaffected.
        """
        m = self._model
        camera = camera or self._cfg['debug']['camera']
        res = int(self._cfg['debug']['resolution'])
        if self._debug_renderer is None:
            self._debug_renderer = mujoco.Renderer(model=m, height=res, width=res)
        mass, ids = self._mass, self._debug_geom_ids
        dbg = self._cfg['debug']
        saved = {
            'bar_rgba': m.geom_rgba[self._bar_geom_id].copy(),
            'target_rgba': m.geom_rgba[self._target_geom_id].copy(),
        }
        max_r = dbg['mass_ball_max_radius']
        for key, w in (('mass_L', mass.w_left), ('mass_R', mass.w_right)):
            frac = (w / mass.total) ** (1.0 / 3.0)
            m.geom_size[ids[key], 0] = max(dbg['mass_ball_min_radius'], max_r * frac)
        m.geom_pos[ids['com']] = (mass.com_offset, 0.0, 0.0)
        grasp_visible = self._debug_grasp_s is not None
        if grasp_visible:
            m.geom_pos[ids['grasp']] = (self._debug_grasp_s, 0.0, 0.0)
        m.geom_rgba[ids['grasp'], 3] = 0.8 if grasp_visible else 0.0
        m.geom_rgba[self._bar_geom_id, 3] = dbg['bar_alpha']
        m.geom_rgba[self._target_geom_id] = dbg['target_rgba']

        # The marker poses above live in the model, so the live data's geom poses are stale until the next step.
        # Render from a scratch copy of the state instead (never touch self._data: even a kinematics refresh on it
        # perturbs the next physics step).
        if getattr(self, '_debug_data', None) is None:
            self._debug_data = mujoco.MjData(m)
        dd = self._debug_data
        dd.qpos[:], dd.qvel[:] = self._data.qpos, self._data.qvel
        dd.mocap_pos[:], dd.mocap_quat[:] = self._data.mocap_pos, self._data.mocap_quat
        mujoco.mj_kinematics(m, dd)
        mujoco.mj_camlight(m, dd)
        opt = mujoco.MjvOption()
        opt.geomgroup[5] = 1
        try:
            self._debug_renderer.update_scene(dd, camera=camera, scene_option=opt)
            frame = self._debug_renderer.render()
        finally:
            m.geom_rgba[self._bar_geom_id] = saved['bar_rgba']
            m.geom_rgba[self._target_geom_id] = saved['target_rgba']
        return frame

    def close(self):
        if self._debug_renderer is not None:
            self._debug_renderer.close()
            self._debug_renderer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
