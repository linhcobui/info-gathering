"""Atomic motion primitives, plan generation, and per-frame choreography."""
from dataclasses import dataclass
import numpy as np


def slot_x(slot, spacing):
    return (slot - 1) * spacing


def slot_positions(spacing):
    return np.array([slot_x(s, spacing) for s in range(3)])


@dataclass
class Transition:
    kind: str
    a: int = -1
    b: int = -1
    slot: int = -1


def generate_plan(cfg, rng):
    plan = []
    for _ in range(cfg["blocks_per_episode"]):
        for _ in range(cfg["swaps_per_block"]):
            a, b = rng.choice(3, size=2, replace=False)
            plan.append(Transition("swap", a=int(a), b=int(b)))
        plan.append(Transition("lift", slot=int(cfg["lift_slot"])))
    return plan


def hold(n, cup_pos, ball_pos):
    cups = np.repeat(cup_pos[None], n, axis=0)
    balls = np.repeat(ball_pos[None], n, axis=0)
    return cups, balls


def vertical_move(n, cup_pos, ball_pos, which, z_from, z_to, ball_rides=False):
    cups = np.repeat(cup_pos[None], n, axis=0).copy()
    balls = np.repeat(ball_pos[None], n, axis=0).copy()
    ts = np.linspace(0, 1, n)
    ss = ts * ts * (3 - 2 * ts)
    for f in range(n):
        z = z_from + (z_to - z_from) * ss[f]
        for k in which:
            cups[f, k, 2] = z
    return cups, balls


def swap_arc(n, cup_pos, ball_pos, id_a, id_b, spacing,
             arc_front, arc_back, ball_rides_id):
    cups = np.repeat(cup_pos[None], n, axis=0).copy()
    balls = np.repeat(ball_pos[None], n, axis=0).copy()

    xa0, ya0 = cup_pos[id_a, 0], cup_pos[id_a, 1]
    xb0, yb0 = cup_pos[id_b, 0], cup_pos[id_b, 1]
    xa1, xb1 = xb0, xa0

    ts = np.linspace(0, 1, n)
    ss = ts * ts * (3 - 2 * ts)
    a_moves_right = xa1 > xa0

    for f in range(n):
        s = ss[f]
        xa = xa0 + (xa1 - xa0) * s
        ya = ya0 + (arc_front if a_moves_right else -arc_back) * np.sin(np.pi * s)
        cups[f, id_a, 0] = xa
        cups[f, id_a, 1] = ya
        xb = xb0 + (xb1 - xb0) * s
        yb = yb0 + (arc_front if not a_moves_right else -arc_back) * np.sin(np.pi * s)
        cups[f, id_b, 0] = xb
        cups[f, id_b, 1] = yb
        if ball_rides_id == id_a:
            balls[f, 0] = xa
            balls[f, 1] = ya
        elif ball_rides_id == id_b:
            balls[f, 0] = xb
            balls[f, 1] = yb
    return cups, balls


def choreograph(cfg, plan, rng):
    spacing = cfg["cup_spacing"]
    rest_z = cfg["cup_rest_z"]
    up_z = cfg["cup_up_z"]
    ball_z = cfg["ball_z"]
    xs = slot_positions(spacing)

    if cfg["ball_start_slot"] is None:
        ball_slot0 = int(rng.integers(3))
    else:
        ball_slot0 = int(cfg["ball_start_slot"])

    slot_of_identity = np.array([0, 1, 2])
    ball_identity = ball_slot0

    cup_pos = np.stack([[xs[slot_of_identity[k]], 0.0, up_z] for k in range(3)])
    ball_pos = np.array([xs[ball_slot0], 0.0, ball_z])

    C, B, A, PH, BALLSLOT, CUPORDER = [], [], [], [], [], []

    def emit(cups, balls, action_vec, phase_id):
        n = cups.shape[0]
        C.append(cups)
        B.append(balls)
        A.append(np.repeat(np.array(action_vec)[None], n, axis=0))
        PH.append(np.full(n, phase_id, dtype=np.uint8))
        bs = slot_of_identity[ball_identity]
        BALLSLOT.append(np.full(n, bs, dtype=np.uint8))
        order = np.zeros(3, dtype=np.uint8)
        for k in range(3):
            order[slot_of_identity[k]] = k
        CUPORDER.append(np.repeat(order[None], n, axis=0))

    NO_ACT = [0, 0, 0]
    phase = {"reveal": 0, "cover": 1, "pause": 2, "swap": 3, "lift": 4}

    cups, balls = hold(cfg["reveal_frames"], cup_pos, ball_pos)
    emit(cups, balls, NO_ACT, phase["reveal"])

    cups, balls = vertical_move(cfg["cover_frames"], cup_pos, ball_pos,
                                which=[0, 1, 2], z_from=up_z, z_to=rest_z)
    cup_pos = cups[-1].copy()
    ball_pos = balls[-1].copy()
    emit(cups, balls, NO_ACT, phase["cover"])

    for tr in plan:
        cups, balls = hold(cfg["pause_frames"], cup_pos, ball_pos)
        emit(cups, balls, NO_ACT, phase["pause"])

        if tr.kind == "swap":
            id_a = int(np.where(slot_of_identity == tr.a)[0][0])
            id_b = int(np.where(slot_of_identity == tr.b)[0][0])
            cups, balls = swap_arc(cfg["swap_frames"], cup_pos, ball_pos,
                                   id_a, id_b, spacing,
                                   cfg["arc_height_front"], cfg["arc_height_back"],
                                   ball_rides_id=ball_identity)
            cup_pos = cups[-1].copy()
            ball_pos = balls[-1].copy()
            slot_of_identity[id_a], slot_of_identity[id_b] = \
                slot_of_identity[id_b], slot_of_identity[id_a]
            emit(cups, balls, NO_ACT, phase["swap"])
        else:
            id_lift = int(np.where(slot_of_identity == tr.slot)[0][0])
            action = [0, 0, 0]
            action[tr.slot] = 1
            up = cfg["lift_up_frames"]
            pz = cfg["lift_pause_frames"]
            dn = cfg["lift_down_frames"]
            c1, b1 = vertical_move(up, cup_pos, ball_pos, [id_lift], rest_z, up_z)
            c2, b2 = hold(pz, c1[-1], b1[-1])
            c3, b3 = vertical_move(dn, c2[-1], b2[-1], [id_lift], up_z, rest_z)
            cups = np.concatenate([c1, c2, c3], axis=0)
            balls = np.concatenate([b1, b2, b3], axis=0)
            cup_pos = cups[-1].copy()
            ball_pos = balls[-1].copy()
            emit(cups, balls, action, phase["lift"])

    return {
        "cup_xyz": np.concatenate(C, axis=0).astype(np.float32),
        "ball_xyz": np.concatenate(B, axis=0).astype(np.float32),
        "action": np.concatenate(A, axis=0).astype(np.uint8),
        "phase": np.concatenate(PH, axis=0),
        "ball_slot": np.concatenate(BALLSLOT, axis=0),
        "cup_order": np.concatenate(CUPORDER, axis=0),
    }
