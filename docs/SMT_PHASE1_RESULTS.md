# SMT phase 1 — results

Eval set: `data/memtest4_eval.zarr` (seed 1), 315 episodes, 63 per cup-swap count, positive rate 0.359.  
Seen / unseen: whether the episode's exact (start slot, swap sequence) occurs in that model's training episodes (phase 1: training split, 25 holdout episodes excluded; DreamerV3: all training episodes).  
F1 cells are `F1 (n)`.

## Ablations

| id | model | training data | train steps | decoder input | reward target | supervised frames |
|---|---|---|---|---|---|---|
| P1-250 | phase 1, binary head, obs decoder | memtest4 (250) | 20k | memory + actions + frames | ball under lifted cup | lift, offsets 1–9 |
| P1-500 | phase 1, binary head, obs decoder | memtest4_big (500) | 20k | memory + actions + frames | ball under lifted cup | lift, offsets 1–9 |
| P1-500-noobs | phase 1, binary head, no-obs decoder | memtest4_big (500) | 20k | memory + actions | ball under lifted cup | lift, offsets 1–9 |
| P1-500-slot | phase 1, slot head (lift frames), no-obs decoder | memtest4_big (500) | 20k | memory + actions | ball slot 0/1/2 | lift, offsets 1–9 |
| P1-500-dense | phase 1, slot head (every frame), no-obs decoder | memtest4_big (500) | 20k | memory + actions | ball slot 0/1/2 (switch at swap halfway) | every real frame |
| DV3-250 | DreamerV3 (R2-Dreamer) | memtest4 (250) | 15k | — | — | — |
| DV3-500 | DreamerV3 (R2-Dreamer) | memtest4_big (500) | 30k | — | — | — |

## Training-set coverage of the swap-sequence space

| swaps | distinct sequences (3 × 3ⁿ) | memtest4 episodes | memtest4 unique | memtest4_big episodes | memtest4_big unique |
|---|---|---|---|---|---|
| 0 | 3 | 51 | 3 | 63 | 3 |
| 1 | 9 | 51 | 9 | 63 | 9 |
| 2 | 27 | 51 | 22 | 126 | 27 |
| 3 | 81 | 49 | 36 | 126 | 65 |
| 4 | 243 | 48 | 41 | 122 | 95 |

## Pixel eval — imagined F1 by cup swaps (all episodes)

| id | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw | all | accuracy (all) |
|---|---|---|---|---|---|---|---|
| P1-250 | 1.000 | 0.963 | 0.667 | 0.410 | 0.471 | 0.712 | 0.800 |
| P1-500 | 0.977 | 1.000 | 0.897 | 0.868 | 0.651 | 0.884 | 0.917 |
| P1-500-noobs | 1.000 | 1.000 | 1.000 | 0.880 | 0.625 | 0.892 | 0.924 |
| P1-500-slot | 0.955 | 1.000 | 0.929 | 0.962 | 0.837 | 0.942 | 0.959 |
| P1-500-dense | 1.000 | 1.000 | 1.000 | 0.962 | 0.863 | 0.960 | 0.971 |
| DV3-250 | 0.976 | 0.926 | 0.667 | 0.500 | 0.510 | 0.714 | 0.794 |
| DV3-500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Pixel eval — imagined F1, SEEN swap sequences

| id | 1 sw | 2 sw | 3 sw | 4 sw | 2–4 sw | 2–4 sw accuracy | 4 sw always-ball F1 |
|---|---|---|---|---|---|---|---|
| P1-250 | 0.963 (63) | 0.818 (52) | 1.000 (19) | 1.000 (8) | 0.905 (79) | 0.949 | 0.857 |
| P1-500 | 1.000 (63) | 0.897 (63) | 0.917 (53) | 0.783 (29) | 0.880 (145) | 0.917 | 0.585 |
| P1-500-noobs | 1.000 (63) | 1.000 (63) | 0.870 (53) | 0.720 (29) | 0.866 (145) | 0.910 | 0.585 |
| P1-500-slot | 1.000 (63) | 0.929 (63) | 1.000 (53) | 1.000 (29) | 0.980 (145) | 0.986 | 0.585 |
| P1-500-dense | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) | 1.000 | 0.585 |
| DV3-250 | 0.926 (63) | 0.783 (52) | 0.600 (20) | 0.909 (8) | 0.773 (80) | 0.875 | 0.857 |
| DV3-500 | 1.000 (63) | 1.000 (63) | 1.000 (54) | 1.000 (30) | 1.000 (147) | 1.000 | 0.605 |

## Pixel eval — imagined F1, UNSEEN swap sequences

| id | 1 sw | 2 sw | 3 sw | 4 sw | 2–4 sw | 2–4 sw accuracy | 4 sw always-ball F1 |
|---|---|---|---|---|---|---|---|
| P1-250 | — (0) | 0.364 (11) | 0.258 (44) | 0.308 (55) | 0.296 (110) | 0.482 | 0.514 |
| P1-500 | — (0) | — (0) | 0.400 (10) | 0.500 (34) | 0.480 (44) | 0.705 | 0.553 |
| P1-500-noobs | — (0) | — (0) | 1.000 (10) | 0.522 (34) | 0.593 (44) | 0.750 | 0.553 |
| P1-500-slot | — (0) | — (0) | 0.500 (10) | 0.632 (34) | 0.609 (44) | 0.795 | 0.553 |
| P1-500-dense | — (0) | — (0) | 0.500 (10) | 0.741 (34) | 0.710 (44) | 0.795 | 0.553 |
| DV3-250 | — (0) | 0.400 (11) | 0.474 (43) | 0.400 (55) | 0.432 (109) | 0.541 | 0.514 |
| DV3-500 | — (0) | — (0) | 1.000 (9) | 1.000 (33) | 1.000 (42) | 1.000 | 0.533 |

## Pixel eval — 4 cup swaps, confusion counts

| id | subset | n | TP | FP | TN | FN | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| P1-250 | seen | 8 | 6 | 0 | 2 | 0 | 1.000 | 1.000 |
| P1-250 | unseen | 55 | 6 | 14 | 22 | 13 | 0.308 | 0.509 |
| P1-500 | seen | 29 | 9 | 2 | 15 | 3 | 0.783 | 0.828 |
| P1-500 | unseen | 34 | 5 | 2 | 19 | 8 | 0.500 | 0.706 |
| P1-500-noobs | seen | 29 | 9 | 4 | 13 | 3 | 0.720 | 0.759 |
| P1-500-noobs | unseen | 34 | 6 | 4 | 17 | 7 | 0.522 | 0.676 |
| P1-500-slot | seen | 29 | 12 | 0 | 17 | 0 | 1.000 | 1.000 |
| P1-500-slot | unseen | 34 | 6 | 0 | 21 | 7 | 0.632 | 0.794 |
| P1-500-dense | seen | 29 | 12 | 0 | 17 | 0 | 1.000 | 1.000 |
| P1-500-dense | unseen | 34 | 10 | 4 | 17 | 3 | 0.741 | 0.794 |
| DV3-250 | seen | 8 | 5 | 0 | 2 | 1 | 0.909 | 0.875 |
| DV3-250 | unseen | 55 | 8 | 13 | 23 | 11 | 0.400 | 0.564 |
| DV3-500 | seen | 30 | 13 | 0 | 17 | 0 | 1.000 | 1.000 |
| DV3-500 | unseen | 33 | 12 | 0 | 21 | 0 | 1.000 | 1.000 |

## Pixel eval — accuracy by slack, swaps 2–4

Slack = static frames between the last swap and the lift. Training slack range 0–12; eval 0–20.

| id | seen seq, same slack | seen seq, new slack ≤ train max | seen seq, slack > train max | unseen seq |
|---|---|---|---|---|
| P1-250 | 1.000 (5) | 1.000 (39) | 0.886 (35) | 0.482 (110) |
| P1-500 | 0.938 (16) | 0.932 (74) | 0.891 (55) | 0.705 (44) |
| P1-500-noobs | 1.000 (16) | 0.892 (74) | 0.909 (55) | 0.750 (44) |
| P1-500-slot | 1.000 (16) | 0.986 (74) | 0.982 (55) | 0.795 (44) |
| P1-500-dense | 1.000 (16) | 1.000 (74) | 1.000 (55) | 0.795 (44) |
| DV3-250 | 1.000 (5) | 0.950 (40) | 0.771 (35) | 0.541 (109) |
| DV3-500 | 1.000 (19) | 1.000 (72) | 1.000 (56) | 1.000 (42) |

## Pixel eval — ceilings

| id | recon | readout(teacher) | readout(posterior) | imagined |
|---|---|---|---|---|
| P1-250 | 1.000 | 1.000 | 0.996 | 0.712 |
| P1-500 | 1.000 | 1.000 | 1.000 | 0.884 |
| P1-500-noobs | 1.000 | 0.946 | 1.000 | 0.892 |
| P1-500-slot | 1.000 | 0.991 | 1.000 | 0.942 |
| P1-500-dense | 1.000 | 0.964 | 0.987 | 0.960 |

## Onset probe — ball-under-lifted-cup F1 from the reward head at the lift onset

Read-out r_t+1 (memory + lift action, no frame). teacher = 60-frame window encode at the onset; posterior = one encode at t=0, then posterior updater per frame; prior = posterior, then 9 prior steps.


### all

| id | state | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw | 2–4 sw |
|---|---|---|---|---|---|---|---|
| P1-250 | teacher | 1.000 (63) | 1.000 (63) | 0.774 (63) | 0.450 (63) | 0.471 (63) | 0.541 (189) |
| P1-250 | posterior | 1.000 (63) | 0.963 (63) | 0.692 (63) | 0.368 (63) | 0.511 (63) | 0.505 (189) |
| P1-250 | prior | 1.000 (63) | 0.963 (63) | 0.621 (63) | 0.410 (63) | 0.480 (63) | 0.492 (189) |
| P1-500 | teacher | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.945 (63) | 0.723 (63) | 0.868 (189) |
| P1-500 | posterior | 0.875 (63) | 1.000 (63) | 0.839 (63) | 0.857 (63) | 0.682 (63) | 0.794 (189) |
| P1-500 | prior | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.857 (63) | 0.651 (63) | 0.807 (189) |
| P1-500-noobs | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.980 (63) | 0.732 (63) | 0.898 (189) |
| P1-500-noobs | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.857 (63) | 0.625 (63) | 0.797 (189) |
| P1-500-noobs | prior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.880 (63) | 0.625 (63) | 0.806 (189) |
| P1-500-slot | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.627 (63) | 0.840 (189) |
| P1-500-slot | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.962 (63) | 0.837 (63) | 0.926 (189) |
| P1-500-slot | prior | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.962 (63) | 0.837 (63) | 0.918 (189) |
| P1-500-dense | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.980 (63) | 0.681 (63) | 0.871 (189) |
| P1-500-dense | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.807 (63) | 0.905 (189) |
| P1-500-dense | prior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.962 (63) | 0.840 (63) | 0.922 (189) |

### SEEN

| id | state | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw | 2–4 sw |
|---|---|---|---|---|---|---|---|
| P1-250 | teacher | 1.000 (63) | 1.000 (63) | 0.952 (52) | 1.000 (19) | 1.000 (8) | 0.976 (79) |
| P1-250 | posterior | 1.000 (63) | 0.963 (63) | 0.824 (52) | 1.000 (19) | 1.000 (8) | 0.919 (79) |
| P1-250 | prior | 1.000 (63) | 0.963 (63) | 0.737 (52) | 1.000 (19) | 1.000 (8) | 0.872 (79) |
| P1-500 | teacher | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.960 (53) | 0.960 (29) | 0.961 (145) |
| P1-500 | posterior | 0.875 (63) | 1.000 (63) | 0.839 (63) | 0.902 (53) | 0.833 (29) | 0.868 (145) |
| P1-500 | prior | 1.000 (63) | 1.000 (63) | 0.963 (63) | 0.889 (53) | 0.783 (29) | 0.884 (145) |
| P1-500-noobs | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) |
| P1-500-noobs | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.844 (53) | 0.720 (29) | 0.854 (145) |
| P1-500-noobs | prior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 0.870 (53) | 0.720 (29) | 0.866 (145) |
| P1-500-slot | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) |
| P1-500-slot | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) |
| P1-500-slot | prior | 1.000 (63) | 1.000 (63) | 0.963 (63) | 1.000 (53) | 1.000 (29) | 0.990 (145) |
| P1-500-dense | teacher | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) |
| P1-500-dense | posterior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 0.923 (29) | 0.980 (145) |
| P1-500-dense | prior | 1.000 (63) | 1.000 (63) | 1.000 (63) | 1.000 (53) | 1.000 (29) | 1.000 (145) |

### UNSEEN

| id | state | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw | 2–4 sw |
|---|---|---|---|---|---|---|---|
| P1-250 | teacher | — (0) | — (0) | 0.400 (11) | 0.312 (44) | 0.308 (55) | 0.321 (110) |
| P1-250 | posterior | — (0) | — (0) | 0.444 (11) | 0.200 (44) | 0.343 (55) | 0.297 (110) |
| P1-250 | prior | — (0) | — (0) | 0.400 (11) | 0.258 (44) | 0.316 (55) | 0.304 (110) |
| P1-500 | teacher | — (0) | — (0) | — (0) | 0.800 (10) | 0.455 (34) | 0.519 (44) |
| P1-500 | posterior | — (0) | — (0) | — (0) | 0.400 (10) | 0.500 (34) | 0.480 (44) |
| P1-500 | prior | — (0) | — (0) | — (0) | 0.500 (10) | 0.500 (34) | 0.500 (44) |
| P1-500-noobs | teacher | — (0) | — (0) | — (0) | 0.667 (10) | 0.353 (34) | 0.400 (44) |
| P1-500-noobs | posterior | — (0) | — (0) | — (0) | 1.000 (10) | 0.522 (34) | 0.593 (44) |
| P1-500-noobs | prior | — (0) | — (0) | — (0) | 1.000 (10) | 0.522 (34) | 0.593 (44) |
| P1-500-slot | teacher | — (0) | — (0) | — (0) | 0.667 (10) | 0.296 (34) | 0.364 (44) |
| P1-500-slot | posterior | — (0) | — (0) | — (0) | 0.500 (10) | 0.632 (34) | 0.609 (44) |
| P1-500-slot | prior | — (0) | — (0) | — (0) | 0.500 (10) | 0.632 (34) | 0.609 (44) |
| P1-500-dense | teacher | — (0) | — (0) | — (0) | 0.667 (10) | 0.348 (34) | 0.385 (44) |
| P1-500-dense | posterior | — (0) | — (0) | — (0) | 0.667 (10) | 0.710 (34) | 0.703 (44) |
| P1-500-dense | prior | — (0) | — (0) | — (0) | 0.500 (10) | 0.692 (34) | 0.667 (44) |

## Onset probe — 3-way slot accuracy at the lift onset (slot-head arms; chance 0.333)

| id | state | all | 4 sw seen | 4 sw unseen | 2–4 sw seen | 2–4 sw unseen |
|---|---|---|---|---|---|---|
| P1-500-slot | teacher | 0.898 (315) | 1.000 (29) | 0.265 (34) | 1.000 (145) | 0.273 (44) |
| P1-500-slot | posterior | 0.943 (315) | 1.000 (29) | 0.735 (34) | 1.000 (145) | 0.591 (44) |
| P1-500-slot | prior | 0.940 (315) | 1.000 (29) | 0.706 (34) | 0.993 (145) | 0.591 (44) |
| P1-500-dense | teacher | 0.895 (315) | 1.000 (29) | 0.206 (34) | 1.000 (145) | 0.250 (44) |
| P1-500-dense | posterior | 0.917 (315) | 0.897 (29) | 0.529 (34) | 0.959 (145) | 0.545 (44) |
| P1-500-dense | prior | 0.914 (315) | 0.931 (29) | 0.500 (34) | 0.972 (145) | 0.477 (44) |

## Onset probe — per-frame slot accuracy after swap k lands, by prefix novelty (P1-500-dense)

Prefix = start slot + first k swaps. Chance 0.333.

| after swap | prefix | n | teacher | posterior |
|---|---|---|---|---|
| 1 | seen | 252 | 1.000 | 1.000 |
| 2 | seen | 189 | 1.000 | 1.000 |
| 3 | seen | 122 | 1.000 | 0.984 |
| 3 | unseen | 4 | 0.250 | 0.500 |
| 4 | seen | 29 | 1.000 | 1.000 |
| 4 | unseen | 34 | 0.235 | 0.706 |

## Closure — R² of chained state against the teacher encoding (24 eval episodes)

| id | updater | h=1 | h=5 | h=10 | h=20 | h=40 | h=60 |
|---|---|---|---|---|---|---|---|
| P1-250 | posterior | 0.9998 | 0.9976 | 0.9944 | 0.9883 | 0.9799 | 0.9842 |
| P1-250 | prior | 0.9965 | 0.9665 | 0.9151 | 0.8092 | 0.5767 | 0.6032 |
| P1-500 | posterior | 0.9994 | 0.9944 | 0.9886 | 0.9784 | 0.9633 | 0.9614 |
| P1-500 | prior | 0.9950 | 0.9385 | 0.8110 | 0.6586 | 0.5570 | -0.1534 |
| P1-500-noobs | posterior | 0.9987 | 0.9946 | 0.9920 | 0.9874 | 0.9863 | 0.9884 |
| P1-500-noobs | prior | 0.9955 | 0.9746 | 0.9401 | 0.8838 | 0.8678 | 0.9859 |
| P1-500-slot | posterior | 0.9979 | 0.9898 | 0.9808 | 0.9524 | 0.8897 | 0.8589 |
| P1-500-slot | prior | 0.9937 | 0.9678 | 0.9357 | 0.8545 | 0.7385 | 0.6654 |
| P1-500-dense | posterior | 0.9988 | 0.9921 | 0.9856 | 0.9752 | 0.9556 | 0.9477 |
| P1-500-dense | prior | 0.9944 | 0.9673 | 0.9322 | 0.8412 | 0.6651 | 0.4334 |

---

# Additional ablations

| id | model | training data |
|---|---|---|
| P1-500-nounif | dense slot, no uniformity (coef_unif 0) | memtest4_big (500) |
| P1-DEPTH1 | dense slot, trained on 0-1 swaps ONLY | memtest4_depth1 (500) |
| P1-DEPTH1-DMT | P1-DEPTH1 + DMT (updaters fine-tuned on own rollouts) | memtest4_depth1 (500) |

P1-DEPTH1 and P1-DEPTH1-DMT train on episodes with at most one swap, so on the 0-4 swap eval set the seen/unseen split does not apply: every 2+ swap episode is out of distribution. Swap count is the axis.

## Pixel eval — imagined, by cup swaps

| id | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw | all F1 | all acc | precision | recall |
|---|---|---|---|---|---|---|---|---|---|
| P1-500-nounif | 1.000 | 0.982 | 1.000 | 0.791 | 0.649 | 0.888 | 0.927 | 0.989 | 0.805 |
| P1-DEPTH1 | 0.955 | 0.778 | 0.400 | 0.543 | 0.400 | 0.601 | 0.654 | 0.512 | 0.726 |
| P1-DEPTH1-DMT | 1.000 | 0.789 | 0.407 | 0.494 | 0.560 | 0.623 | 0.613 | 0.479 | 0.894 |

## Onset probe — 3-way slot accuracy by cup swaps (chance 0.333)

| id | state | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw |
|---|---|---|---|---|---|---|
| P1-500-nounif | teacher | 1.000 | 1.000 | 1.000 | 0.921 | 0.571 |
| P1-500-nounif | posterior | 1.000 | 0.857 | 0.937 | 0.873 | 0.825 |
| P1-500-nounif | prior | 1.000 | 0.825 | 0.857 | 0.810 | 0.810 |
| P1-DEPTH1 | teacher | 1.000 | 1.000 | 0.460 | 0.381 | 0.397 |
| P1-DEPTH1 | posterior | 1.000 | 1.000 | 0.444 | 0.381 | 0.365 |
| P1-DEPTH1 | prior | 1.000 | 1.000 | 0.413 | 0.349 | 0.349 |
| P1-DEPTH1-DMT | teacher | 1.000 | 1.000 | 0.460 | 0.381 | 0.397 |
| P1-DEPTH1-DMT | posterior | 1.000 | 1.000 | 0.460 | 0.333 | 0.302 |
| P1-DEPTH1-DMT | prior | 1.000 | 1.000 | 0.444 | 0.349 | 0.302 |

## Slot accuracy at the first at-rest frame after swap k lands (chance 0.333)

Depth-1-trained models on the 0-4 swap eval set. Swaps 2-4 are depths absent from training.

| id | state | after swap 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| P1-DEPTH1 | teacher | 1.000 | 1.000 | 0.484 | 0.429 | 0.344 |
| P1-DEPTH1 | posterior | 1.000 | 1.000 | 0.453 | 0.365 | 0.375 |
| P1-DEPTH1-DMT | teacher (frozen by DMT) | 1.000 | 1.000 | 0.484 | 0.429 | 0.344 |
| P1-DEPTH1-DMT | posterior | 1.000 | 1.000 | 0.379 | 0.381 | 0.344 |

n per column: 158, 126, 95, 63, 32 episodes.

## DMT objective (P1-DEPTH1-DMT)

| | step 250 | step 6000 | holdout, step 6000 |
|---|---|---|---|
| rollout vs teacher, MSE | 3.0e-3 | 6.2e-6 | 1.45e-3 |
| slot CE on rolled states | 5.6e-5 | 3.1e-6 | 3.8e-6 |

## State sensitivity to a content-free time shift

Mean ||z_a - z_b|| at the lift onset over 99 eval episodes. The shift deletes 4 motionless hold
frames: the ball, the cups and the swap order are unchanged.

| model / state | drop 4 static frames | same slot & swap count, different history | different ball slot | control (same input twice) | state norm |
|---|---|---|---|---|---|
| P1-500-dense teacher | 10.17 | 16.67 | 33.18 | — | 50.6 |
| P1-500-dense posterior | 9.86 | 15.89 | 33.46 | — | 50.6 |
| DreamerV3 (500 eps) deter | 1.28 | 0.98 | 2.86 | 0.230 | 3.1 |

## Memory-state structure (P1-500-dense, training episodes, ball starting in slot 0)

10-NN purity over 534 states sampled after each swap; 158 episodes, 70 distinct swap histories.
The last column is what a state depending ONLY on the cup permutation would score.

| label shared with the 10 nearest neighbours | teacher | posterior | permutation-only state |
|---|---|---|---|
| cup permutation | 0.909 | 0.909 | 1.000 |
| ball slot | 0.967 | 0.964 | 1.000 |
| exact swap history | 0.817 | 0.817 | 0.423 |
| swaps done | 0.936 | 0.938 | 0.567 |

## Uniformity ablation (P1-500-dense vs P1-500-nounif), holdout at step 20000

| | with uniformity | coef_unif = 0 |
|---|---|---|
| uniformity value (logged) | -2.85 | -0.77 |
| z_std | 0.524 | 0.261 |
| distance gap, different-slot vs same-slot | 9.32 | 3.54 |
| slot accuracy, all frames | 0.702 | 0.701 |
| obs_sensitivity | 0.077 | 0.046 |

---

# Subspace ablation — where the ball slot lives in the memory

States are the 10 x 256 = 2560 scalars of `z`, taken at MOTIONLESS HOLD frames, where the
rendered scene is identical across episodes and only the hidden ball slot differs. A
"dimension" is one of those 2560 scalars, addressed by (token, channel). `swaps_done` is how
many swaps have completed at that frame. Unless stated, states are from the posterior chain.

## Variance explained per dimension (P1-500-dense, 961 states, 250 training episodes)

| | teacher | posterior |
|---|---|---|
| dims with eta^2 > 0.5 for ball slot | 137 | 140 |
| dims with eta^2 > 0.5 for swaps_done | 371 | 398 |
| dims > 0.5 for BOTH | 0 | 0 |
| dims with slot > 0.5 and swaps_done < 0.1 | 109 | 108 |
| variance inside the 2D plane of the three slot means | 0.261 | 0.261 |

## Top slot dimensions (posterior, ranked on 0-1 swap states)

| rank | token | channel | eta^2 slot (0-1 sw) | eta^2 slot (all) | eta^2 swaps_done |
|---|---|---|---|---|---|
| 1 | 5 | 117 | 0.948 | 0.695 | 0.060 |
| 2 | 6 | 117 | 0.915 | 0.684 | 0.118 |
| 3 | 0 | 7 | 0.898 | 0.656 | 0.074 |
| 4 | 4 | 117 | 0.894 | 0.686 | 0.107 |
| 5 | 5 | 246 | 0.864 | 0.373 | 0.348 |
| 6 | 6 | 79 | 0.861 | 0.718 | 0.003 |

The 64 top dimensions are 34 distinct channels; 30 are the same channel read from another token.
Channel 138 appears in tokens 0,2,4,5,6; channel 117 in 2,4,5,6; channels 12 and 85 in 4 tokens each.

## Role of each memory token (mean eta^2 over its 256 channels)

| token | ball slot | swaps done | elapsed time | dims in the top 64 |
|---|---|---|---|---|
| 0 | 0.447 | 0.284 | 0.174 | 11 |
| 1 | 0.412 | 0.258 | 0.159 | 5 |
| 2 | 0.407 | 0.182 | 0.108 | 12 |
| 3 | 0.233 | 0.373 | 0.228 | 1 |
| 4 | 0.384 | 0.182 | 0.113 | 7 |
| 5 | 0.463 | 0.200 | 0.121 | 13 |
| 6 | 0.382 | 0.188 | 0.116 | 15 |
| 7 | 0.195 | 0.302 | 0.217 | 0 |
| 8 | 0.175 | 0.262 | 0.242 | 0 |
| 9 | 0.315 | 0.337 | 0.198 | 0 |

Overlap of the 64 selected dimensions between teacher and posterior states: 53 / 64.

## Cross-depth transfer of a linear slot read-out

Fit on shallow frames of TRAINING episodes, tested on EVAL episodes, split by whether that
episode's swap sequence occurs in training. Chance 0.333; cell is accuracy (n).

| model | dims | subset | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw |
|---|---|---|---|---|---|---|---|
| P1-500-dense | 4 | seen | 0.917 (300) | 1.000 (475) | 0.779 (299) | 0.672 (296) | 0.414 (133) |
| P1-500-dense | 64 | seen | 1.000 | 1.000 | 0.983 | 0.956 | 0.774 |
| P1-500-dense | 2560 | seen | 1.000 | 1.000 | 0.977 | 0.899 | 0.714 |
| P1-500-dense | 64 | UNSEEN | — | 1.000 (44) | — | 0.740 (96) | 0.461 (167) |
| P1-500-dense | 2560 | UNSEEN | — | 1.000 | — | 0.656 | 0.443 |

Episode-disjoint split within a single episode set (no seen/unseen split available there):

| model | dims | 0 sw | 1 sw | 2 sw | 3 sw | 4 sw |
|---|---|---|---|---|---|---|
| P1-500-dense, fit 0-1 sw | 4 | 1.000 | 1.000 | 0.635 | 0.500 | 0.545 |
| P1-500-dense, fit 0-1 sw | 64 | 1.000 | 1.000 | 1.000 | 0.983 | 1.000 |
| P1-DEPTH1, fit 0-1 sw | 4 | 1.000 | 1.000 | 0.402 | 0.503 | 0.392 |
| P1-DEPTH1, fit 0-1 sw | 64 | 1.000 | 1.000 | 0.433 | 0.477 | 0.402 |

## Linear vs non-linear probes, fit and tested WITHIN each subset

Episode-grouped cross-validation, PCA to 64 components first. The probe never has to generalise
across sequences, so this asks only whether the information is present at all.

| model | subset | n | majority | linear | MLP 64 | MLP 256-64 | kNN-5 |
|---|---|---|---|---|---|---|---|
| P1-500-dense | 4 swaps, seen | 133 | 0.489 | 1.000 | 0.691 | 0.933 | 0.886 |
| P1-500-dense | 4 swaps, UNSEEN | 167 | 0.515 | 0.713 | 0.534 | 0.569 | 0.694 |
| P1-500-dense | 3 swaps, UNSEEN | 96 | 0.500 | 0.750 | 0.604 | 0.479 | 0.479 |
| P1-DEPTH1 | 1 swap | 519 | 0.397 | 1.000 | 1.000 | 1.000 | 1.000 |
| P1-DEPTH1 | 2 swaps | 299 | 0.465 | 0.552 | 0.609 | 0.646 | 0.599 |
| P1-DEPTH1 | 3 swaps | 392 | 0.413 | 0.449 | 0.518 | 0.546 | 0.421 |
| P1-DEPTH1 | 4 swaps | 300 | 0.503 | 0.370 | 0.433 | 0.400 | 0.463 |

---

# Alignment arms (uniformity replaced by Wang & Isola alignment)

Alignment pulls together states at MOTIONLESS HOLD frames that share the ball slot, with a hinge
holding the three slot means >= 20 apart; `coef_unif: 0`. Positive pairs are defined by the
privileged slot label. Both arms: 500-episode training set, no-obs decoder.

| id | decoder head | head supervision | coef_align | coef_sep | coef_unif |
|---|---|---|---|---|---|
| P1-500-ALIGNDENSE | 3-way slot | every real frame | 0.1 | 1e-3 | 0 |
| P1-500-ALIGNLIFT | binary (ball under lifted cup) | 9 lift frames | 0.1 | 1e-3 | 0 |

## Onset probe on memtest4_eval — ball-under-lifted-cup F1 (n)


### posterior

| id | all | 4 sw seen | 4 sw UNSEEN | 2-4 sw UNSEEN |
|---|---|---|---|---|
| P1-500-dense | 0.945 (315) | 0.923 (29) | 0.710 (34) | 0.703 (44) |
| P1-500-nounif | 0.924 (315) | 0.857 (29) | 0.632 (34) | 0.571 (44) |
| P1-500-ALIGNDENSE | 0.987 (315) | 1.000 (29) | 0.917 (34) | 0.897 (44) |
| P1-500-ALIGNLIFT | 0.934 (315) | 0.857 (29) | 0.960 (34) | 0.889 (44) |

### prior

| id | all | 4 sw seen | 4 sw UNSEEN | 2-4 sw UNSEEN |
|---|---|---|---|---|
| P1-500-dense | 0.956 (315) | 1.000 (29) | 0.692 (34) | 0.667 (44) |
| P1-500-nounif | 0.903 (315) | 0.857 (29) | 0.556 (34) | 0.500 (44) |
| P1-500-ALIGNDENSE | 0.983 (315) | 1.000 (29) | 0.963 (34) | 0.882 (44) |
| P1-500-ALIGNLIFT | 0.939 (315) | 0.909 (29) | 0.960 (34) | 0.889 (44) |

### teacher

| id | all | 4 sw seen | 4 sw UNSEEN | 2-4 sw UNSEEN |
|---|---|---|---|---|
| P1-500-dense | 0.928 (315) | 1.000 (29) | 0.348 (34) | 0.385 (44) |
| P1-500-nounif | 0.899 (315) | 0.957 (29) | 0.095 (34) | 0.087 (44) |
| P1-500-ALIGNDENSE | 0.926 (315) | 0.960 (29) | 0.462 (34) | 0.516 (44) |
| P1-500-ALIGNLIFT | 0.926 (315) | 1.000 (29) | 0.545 (34) | 0.500 (44) |

## Onset probe — 3-way slot accuracy (slot-head arms; chance 0.333)

| id | state | 4 sw seen | 4 sw UNSEEN | 2-4 sw UNSEEN |
|---|---|---|---|---|
| P1-500-dense | teacher | 1.000 (29) | 0.206 (34) | 0.250 (44) |
| P1-500-dense | posterior | 0.897 (29) | 0.529 (34) | 0.545 (44) |
| P1-500-dense | prior | 0.931 (29) | 0.500 (34) | 0.477 (44) |
| P1-500-nounif | teacher | 0.966 (29) | 0.235 (34) | 0.295 (44) |
| P1-500-nounif | posterior | 0.897 (29) | 0.765 (34) | 0.705 (44) |
| P1-500-nounif | prior | 0.897 (29) | 0.735 (34) | 0.682 (44) |
| P1-500-ALIGNDENSE | teacher | 0.966 (29) | 0.441 (34) | 0.455 (44) |
| P1-500-ALIGNDENSE | posterior | 1.000 (29) | 0.941 (34) | 0.886 (44) |
| P1-500-ALIGNDENSE | prior | 1.000 (29) | 0.971 (34) | 0.909 (44) |

## Extrapolation to FIVE swaps (data/memtest4_eval5.zarr, 63 episodes)

One swap deeper than any training episode; every sequence is unseen at an unseen depth.
Positive rate 0.349, so always-ball scores F1 0.518 / accuracy 0.349; slot chance 0.333.

| id | state | F1 | acc | TP | FP | TN | FN | slot acc |
|---|---|---|---|---|---|---|---|---|
| P1-500-dense | teacher | 0.490 | 0.603 | 12 | 15 | 26 | 10 | 0.444 |
| P1-500-dense | posterior | 0.600 | 0.619 | 18 | 20 | 21 | 4 | 0.508 |
| P1-500-dense | prior | 0.667 | 0.714 | 18 | 14 | 27 | 4 | 0.508 |
| P1-500-nounif | teacher | 0.356 | 0.540 | 8 | 15 | 26 | 14 | 0.349 |
| P1-500-nounif | posterior | 0.606 | 0.794 | 10 | 1 | 40 | 12 | 0.746 |
| P1-500-nounif | prior | 0.533 | 0.778 | 8 | 0 | 41 | 14 | 0.730 |
| P1-500-ALIGNDENSE | teacher | 0.481 | 0.556 | 13 | 19 | 22 | 9 | 0.429 |
| P1-500-ALIGNDENSE | posterior | 1.000 | 1.000 | 22 | 0 | 41 | 0 | 0.952 |
| P1-500-ALIGNDENSE | prior | 0.880 | 0.905 | 22 | 6 | 35 | 0 | 0.857 |
| P1-500-ALIGNLIFT | teacher | 0.538 | 0.619 | 14 | 16 | 25 | 8 | — |
| P1-500-ALIGNLIFT | posterior | 0.667 | 0.825 | 11 | 0 | 41 | 11 | — |
| P1-500-ALIGNLIFT | prior | 0.706 | 0.841 | 12 | 0 | 41 | 10 | — |

## Pixel eval for the alignment arms — imagined F1 (n)

`memtest4_eval`, 315 episodes. Seen/unseen = whether the episode's exact (start slot, swap
sequence) occurs in the 500-episode training set; 271 seen, 44 unseen (unseen exists only at 3-4 swaps).

| id | all | 4 sw seen | 4 sw UNSEEN | 2-4 sw UNSEEN | all seen | all UNSEEN | precision | recall |
|---|---|---|---|---|---|---|---|---|
| P1-500-dense | 0.960 (315) | 1.000 (29) | 0.741 (34) | 0.710 (44) | 1.000 (271) | 0.710 (44) | 0.956 | 0.965 |
| P1-500-nounif | 0.888 (315) | 0.800 (29) | 0.471 (34) | 0.421 (44) | 0.935 (271) | 0.421 (44) | 0.989 | 0.805 |
| ALIGN-DENSE | 0.987 (315) | 1.000 (29) | 1.000 (34) | 0.938 (44) | 0.995 (271) | 0.938 (44) | 0.974 | 1.000 |
| ALIGN-LIFT | 0.939 (315) | 0.909 (29) | 0.960 (34) | 0.889 (44) | 0.946 (271) | 0.889 (44) | 1.000 | 0.885 |
| ALIGN-EMBED | 0.277 (315) | 0.400 (29) | 0.125 (34) | 0.111 (44) | 0.303 (271) | 0.111 (44) | 0.792 | 0.168 |

## Pixel eval at FIVE swaps (memtest4_eval5, 63 episodes; always-ball F1 0.518)

| id | F1 | acc | TP | FP | TN | FN |
|---|---|---|---|---|---|---|
| ALIGN-DENSE | 0.880 | 0.905 | 22 | 6 | 35 | 0 |
| ALIGN-LIFT | 0.706 | 0.841 | 12 | 0 | 41 | 10 |
| ALIGN-EMBED | 0.222 | 0.667 | 3 | 2 | 39 | 19 |

## Ceilings for the alignment arms

| id | recon | readout(teacher) | readout(posterior) | imagined |
|---|---|---|---|---|
| P1-500-dense | 1.000 | 0.964 | 0.987 | 0.960 |
| P1-500-nounif | 1.000 | 0.945 | 1.000 | 0.888 |
| ALIGN-DENSE | 1.000 | 0.904 | 0.996 | 0.987 |
| ALIGN-LIFT | 1.000 | 0.753 | 1.000 | 0.939 |
| ALIGN-EMBED | 1.000 | 0.991 | 1.000 | 0.277 |

## ALIGN-EMBED: alignment with a RECONSTRUCTION head (no ball label in the head)

The decoder predicts the frame embedding from memory + future actions (SMT's predictive-state
objective). Alignment is unchanged. Stage-1 holdout at step 20000:

| | ALIGN-EMBED | P1-500-dense | P1-500-nounif |
|---|---|---|---|
| z_std | 0.078 | 0.524 | 0.261 |
| obs_sensitivity | 0.022 | 0.077 | 0.046 |
| uniformity value (logged) | -0.57 | -2.85 | -0.77 |
| sep (hinge, holdout) | 11.79 | — | — |
| recon_r2 (holdout) | 0.633 (flat from step 1000) | — | — |

Slot decodable from the state at motionless hold frames, episode-grouped CV, eval episodes:

| swaps | ALIGN-DENSE | ALIGN-EMBED | majority (ALIGN-EMBED) |
|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.333 |
| 1 | 1.000 | 0.977 | 0.459 |
| 2 | 0.967 | 0.750 | 0.472 |
| 3 | 0.964 | 0.594 | 0.352 |
| 4 | 0.877 | 0.352 | 0.380 |

# ALIGN-TARGET — alignment on the 1-bit lift answer instead of the 3-way slot

Identical to ALIGN-LIFT in every respect (`coef_align: 0.1`, `coef_sep: 1e-3`, `align_margin: 20`,
`coef_unif: 0`, binary head, `decoder_obs: false`, 500-episode 0-4 swap training set). The only
change is `align_label: target`: positive pairs are rest states that agree on "is the ball under
the cup that will be lifted", at any swap depth, instead of rest states sharing the 3-way slot.

`episode_target` is `[500, 0, 0]` on the training set and `[315, 0, 0]` on eval — the lifted cup is
always slot 0 — so the 1-bit label partitions states as {slot 0} vs {slot 1, slot 2}.

## State geometry at rest frames, by 3-way slot

| arm | within | between (0-1, 0-2, 1-2) | between/within |
|---|---|---|---|
| ALIGN-LIFT | 2.94 | 19.51  20.43  20.99 | 6.90 |
| ALIGN-TARGET | 6.76 | 16.79  16.47  **1.90** | 1.73 |

## 10-NN purity, cross-episode neighbours only, 50-D PCA space (884 rest states, 100 episodes)

| arm | 3-way slot | 1-bit "in slot 0" |
|---|---|---|
| ALIGN-LIFT | 0.999 | 0.999 |
| ALIGN-TARGET | 0.806 | 0.920 |
| chance | 0.335 | 0.536 |

## Onset probe, posterior — ball under the lifted cup

"bit-only ceiling" is the accuracy of an optimally-played representation that retains ONLY the
1-bit label. {0} vs {1,2} is not closed under the swaps, so that representation is forced to guess
whenever the state is "not 0" and a swap involves slot 0.

`data/memtest4_eval`, 315 episodes:

| arm | swaps | n | F1 | acc | TP | FP | TN | FN | bit-only ceiling (acc) |
|---|---|---|---|---|---|---|---|---|---|
| ALIGN-LIFT | 0 | 63 | 1.000 | 1.000 | 21 | 0 | 42 | 0 | 1.000 |
| ALIGN-LIFT | 1 | 63 | 1.000 | 1.000 | 28 | 0 | 35 | 0 | 0.667 |
| ALIGN-LIFT | 2 | 63 | 1.000 | 1.000 | 13 | 0 | 50 | 0 | 0.825 |
| ALIGN-LIFT | 3 | 63 | 0.762 | 0.841 | 16 | 0 | 37 | 10 | 0.603 |
| ALIGN-LIFT | 4 | 63 | 0.913 | 0.937 | 21 | 0 | 38 | 4 | 0.603 |
| ALIGN-LIFT | all | 315 | 0.934 | 0.956 | 99 | 0 | 202 | 14 | |
| ALIGN-TARGET | 0 | 63 | 1.000 | 1.000 | 21 | 0 | 42 | 0 | 1.000 |
| ALIGN-TARGET | 1 | 63 | 1.000 | 1.000 | 28 | 0 | 35 | 0 | 0.667 |
| ALIGN-TARGET | 2 | 63 | 1.000 | 1.000 | 13 | 0 | 50 | 0 | 0.825 |
| ALIGN-TARGET | 3 | 63 | 0.881 | 0.889 | 26 | 7 | 30 | 0 | 0.603 |
| ALIGN-TARGET | 4 | 63 | 0.522 | 0.651 | 12 | 9 | 29 | 13 | 0.603 |
| ALIGN-TARGET | all | 315 | 0.873 | 0.908 | 100 | 16 | 186 | 13 | |

`data/memtest4_eval5`, 63 episodes (always-ball F1 0.518, acc 0.349):

| arm | F1 | acc | TP | FP | TN | FN | bit-only ceiling (acc) |
|---|---|---|---|---|---|---|---|
| ALIGN-LIFT | 0.667 | 0.825 | 11 | 0 | 41 | 11 | 0.651 |
| ALIGN-TARGET | 0.383 | 0.540 | 9 | 16 | 25 | 13 | 0.651 |

## Bit-only ceiling by depth (optimal play on a perfectly collapsed 1-bit state)

| swaps | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| ceiling acc | 1.000 | 0.667 | 0.825 | 0.603 | 0.603 | 0.651 |
| episodes ending in a coin-flip state | 0.0% | 47.6% | 47.6% | 30.2% | 19.0% | 12.7% |

No pixel decoder or pixel eval was run for this arm.

## Onset probe — TEACHER vs POSTERIOR, by prefix novelty

`teacher` re-reads the whole 60-frame window ending at the lift onset (one shot, no updater).
`posterior` encodes the window at t=0 only, then chains the posterior updater to the onset.
Seen/unseen = whether the episode's exact (start slot, swap-pair sequence) occurs among the 475
episodes the model trains on (the anchors config holds out 25 of the 500).

`data/memtest4_eval`, 315 episodes — F1 (n):

| arm | state | all (315) | seen (271) | UNSEEN (44) |
|---|---|---|---|---|
| ALIGN-LIFT | teacher | 0.926 | 0.979 | 0.500 |
| ALIGN-LIFT | posterior | 0.934 | 0.941 | 0.889 |
| ALIGN-TARGET | teacher | 0.904 | 0.990 | 0.375 |
| ALIGN-TARGET | posterior | 0.873 | 0.929 | 0.545 |

4 swaps only — F1 (n):

| arm | state | seen (29) | UNSEEN (34) |
|---|---|---|---|
| ALIGN-LIFT | teacher | 1.000 | 0.545 |
| ALIGN-LIFT | posterior | 0.857 | 0.960 |
| ALIGN-TARGET | teacher | 1.000 | 0.345 |
| ALIGN-TARGET | posterior | 0.526 | 0.519 |

`data/memtest4_eval5`, 63 episodes, all unseen — F1 / acc:

| arm | teacher | posterior |
|---|---|---|
| ALIGN-LIFT | 0.538 / 0.619 | 0.667 / 0.825 |
| ALIGN-TARGET | 0.431 / 0.540 | 0.383 / 0.540 |

Confusion counts, `memtest4_eval` posterior: ALIGN-LIFT FP=0 in every cell (all errors are misses);
ALIGN-TARGET FP=7 seen, 9 unseen, and 16 at depth 5.

# ALIGN-DTW — alignment with NO privileged label

Positives come from the DTW distance between FUTURE EMBEDDING TRAJECTORIES, computed from the
cached embeddings alone (`scripts/build_dtw_matrix.py`, 4672x4672 over every rest anchor).
Restricted to anchors whose future still contains motion — a flag also read off the embeddings,
agreeing with the schedule-derived version on 98.5% of anchors. No `slot_dense_seq`, no
`ball_under_target`. See ALIGN.md §11 for the derivation.

Two variants of how the distance is used:

| id | use of dtw_ij | coef_align | coef_unif | batch |
|---|---|---|---|---|
| ALIGN-DTW-regress | metric regression, `(‖z_i−z_j‖ − 2·dtw_ij)²` | 5e-4 | 0 | 256 |
| ALIGN-DTW-attract | soft pull, `w=exp(−dtw_ij/2)` | 0.2 | 1e-2 | 256 |

Coefficients set by measured encoder-gradient norms, matched to ALIGN-LIFT's total alignment
pressure (align 0.163 + sep 0.918 = 1.08). batch 256 rather than 128 because only 8.2% of
anchors are eligible; at 128 a step scores ~45 pairs, at 256 it scores 136–299.

## Static decodability of the stage-1 states (rest frames, 1404 states, 150 episodes)

kNN purity counts CROSS-EPISODE neighbours only, in 50-D PCA space. Probe = logistic regression
on the 3-way slot, fit on half the episodes and tested on the other half.

| arm | label used | kNN purity | probe (all dims) | probe (top 16) | eta² top-1 | dims eta²>0.1 |
|---|---|---|---|---|---|---|
| ALIGN-LIFT | 3-way slot | 0.999 | 0.996 | 0.999 | 0.993 | 2530 |
| ALIGN-TARGET | 1 bit | 0.819 | 0.863 | 0.765 | 0.781 | 1465 |
| ALIGN-DTW-regress | none | 0.875 | 0.919 | 0.843 | 0.754 | 588 |
| ALIGN-DTW-attract | none | 0.755 | 0.807 | 0.665 | 0.847 | 447 |
| chance | | 0.335 | 0.333 | | | |

## Centroid geometry at rest frames (misleading on its own — see the subspace column above)

| arm | within | between (0-1, 0-2, 1-2) | ratio |
|---|---|---|---|
| ALIGN-LIFT | 2.94 | 19.51  20.43  20.99 | 6.90 |
| ALIGN-TARGET | 6.76 | 16.79  16.47  1.90 | 1.73 |
| ALIGN-DTW-regress | 11.76 | 7.62  7.38  4.92 | 0.56 |

## Onset probe — `memtest4_eval`, F1 (n)

| arm | state | all (315) | seen (271) | UNSEEN (44) |
|---|---|---|---|---|
| ALIGN-LIFT | teacher | 0.926 | 0.979 | 0.500 |
| ALIGN-LIFT | posterior | 0.934 | 0.941 | 0.889 |
| ALIGN-TARGET | teacher | 0.904 | 0.990 | 0.375 |
| ALIGN-TARGET | posterior | 0.873 | 0.929 | 0.545 |
| ALIGN-DTW-regress | teacher | 0.955 | 1.000 | 0.615 |
| ALIGN-DTW-regress | posterior | 0.870 | 0.942 | 0.424 |
| ALIGN-DTW-attract | teacher | 0.938 | 1.000 | 0.500 |
| ALIGN-DTW-attract | posterior | 0.858 | 0.930 | 0.437 |

Both label-free arms are the only ones reaching teacher F1 1.000 on seen sequences, and both have
the WORST posterior on unseen (0.424, 0.437) against ALIGN-LIFT's 0.889. The label-free objective
builds the richer encoder; what the 3-way label buys is a state the recurrent updater can carry.

## Onset probe — posterior F1 by swap depth

| arm | 0 | 1 | 2 | 3 | 4 | 5 (eval5) |
|---|---|---|---|---|---|---|
| ALIGN-LIFT | 1.000 | 1.000 | 1.000 | 0.762 | 0.913 | 0.667 |
| ALIGN-TARGET | 1.000 | 1.000 | 1.000 | 0.881 | 0.522 | 0.383 |
| ALIGN-DTW-regress | 1.000 | 1.000 | 0.867 | 0.943 | 0.586 | 0.561 |
| ALIGN-DTW-attract | 1.000 | 0.809 | 0.960 | 0.943 | 0.654 | 0.415 |

`memtest4_eval5` teacher: ALIGN-LIFT 0.538, ALIGN-TARGET 0.431, ALIGN-DTW-regress 0.429,
ALIGN-DTW-attract 0.341.

attract vs regress is inconclusive on the task: attract wins at 4 swaps (0.654 vs 0.586) and loses
at 5 (0.415 vs 0.561), despite winning decisively on the training objective (align 22x lower,
readout loss 30x lower) and compressing into fewer dimensions (447 vs 588).

## Pixel eval — imagined F1 (n)

| arm | all (315) | seen (271) | UNSEEN (44) | by depth 0–4 | eval5 (63) |
|---|---|---|---|---|---|
| ALIGN-LIFT | 0.939 | 0.946 | 0.889 | 1.00 1.00 1.00 0.76 0.94 | 0.706 |
| ALIGN-TARGET | 0.831 | 0.877 | 0.556 | 0.98 0.93 0.84 0.83 0.59 | 0.408 |
| ALIGN-DTW-regress | 0.884 | 0.942 | 0.444 | 0.93 0.95 0.93 0.98 0.63 | 0.588 |

ALIGN-DTW-regress uses NO privileged label and still beats ALIGN-TARGET, which is given the
task's own output bit at every rest frame: 0.884 vs 0.831 overall and 0.588 vs 0.408 at five
swaps. It loses to ALIGN-TARGET only on unseen sequences (0.444 vs 0.556), the same weakness the
onset probe shows. Both trail ALIGN-LIFT's 3-way privileged label (0.939 / 0.706).

Only `segment == "lift"` rows are a valid measurement; the `reveal` rows score F1 0.500 even on
`gt_ball`.
