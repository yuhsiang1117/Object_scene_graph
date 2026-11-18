# Reachability ceiling: where the 1 m rule meets the floor plan

run `outputs/osg_released_maps/flat_anchor_v2_island_close`; DualMap `outputs/dualmap_official_bench/seed12`; navmesh recomputed for radius 0.18 m, height 0.88 m (what habitat-lab builds for our agent); `shipped` is the HM3D file.

`navmesh` is the nearest point the agent can occupy on its start island, measured to the object; `best attempt` is the nearest any of the trial's scored stops came.

## in_anchor

- trials with **no navmesh point within 1.0 m**: 10/54 on our navmesh, 8/54 on the shipped one
- of the 25 failures, 10 are those trials; their best attempt minus the navmesh optimum (m): [0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.06, 0.12, 0.41, 0.49]
- DualMap scored 1 of those 10 (its follower is not held to the navmesh)
- **reachable subset** (44 trials): ours 29/44 = 65.9%, DualMap 34/44 = 77.3%; full split: ours 29/54, DualMap 35/54

failures by bucket x reachability:

| bucket | reachable | unreachable |
|---|---:|---:|
| committed_elsewhere | 4 | 9 |
| near_miss | 0 | 1 |
| never_in_view | 3 | 0 |
| perception_close | 3 | 0 |
| perception_wall | 5 | 0 |

## cross_anchor

- trials with **no navmesh point within 1.0 m**: 5/53 on our navmesh, 1/53 on the shipped one
- of the 26 failures, 5 are those trials; their best attempt minus the navmesh optimum (m): [0.04, 0.99, 1.22, 2.53, 3.78]
- DualMap scored 1 of those 5 (its follower is not held to the navmesh)
- **reachable subset** (48 trials): ours 27/48 = 56.2%, DualMap 15/48 = 31.2%; full split: ours 27/53, DualMap 16/53

failures by bucket x reachability:

| bucket | reachable | unreachable |
|---|---:|---:|
| committed_elsewhere | 4 | 2 |
| never_in_view | 7 | 0 |
| perception_close | 4 | 0 |
| perception_wall | 6 | 3 |

## Every failure

| trial | bucket | navmesh | shipped | best attempt | final | DualMap final |
|---|---|---:|---:|---:|---:|---:|
| 00829-QaLdnwvtxbs__cross_anchor__0128-1__pitcher | perception_close | 0.43 | 0.33 | 1.12 | 5.34 | 11.73 |
| 00829-QaLdnwvtxbs__cross_anchor__0128-1__scissors | perception_wall | 0.56 | 0.46 | 4.08 | 10.75 | 11.60 |
| 00829-QaLdnwvtxbs__cross_anchor__0128-2__cracker_box | never_in_view | 0.45 | 0.35 | 4.30 | 1.87 | 0.42 |
| 00829-QaLdnwvtxbs__cross_anchor__0128-2__scissors | never_in_view | 0.60 | 0.51 | 6.98 | 5.60 | 7.34 |
| 00829-QaLdnwvtxbs__cross_anchor__0129-1__scissors | perception_wall | 1.02 ** | 0.95 | 4.81 | 4.72 | 11.58 |
| 00829-QaLdnwvtxbs__cross_anchor__0129-1__soup_can | perception_wall | 0.52 | 0.43 | 4.94 | 3.85 | 6.53 |
| 00829-QaLdnwvtxbs__in_anchor__0116__cracker_box | committed_elsewhere | 0.37 | 0.27 | 4.57 | 10.10 | 0.31 |
| 00829-QaLdnwvtxbs__in_anchor__0116__soup_can | perception_close | 0.62 | 0.52 | 1.24 | 5.37 | 0.50 |
| 00829-QaLdnwvtxbs__in_anchor__0117__cracker_box | perception_close | 0.53 | 0.43 | 4.31 | 4.31 | 0.34 |
| 00829-QaLdnwvtxbs__in_anchor__0118__cracker_box | perception_close | 0.52 | 0.42 | 4.27 | 4.48 | 0.45 |
| 00829-QaLdnwvtxbs__in_anchor__0118__pitcher | committed_elsewhere | 1.15 ** | 1.04 | 1.56 | 11.72 | 11.26 |
| 00848-ziup5kvtCCR__cross_anchor__0128-1__cracker_box | perception_wall | 1.07 ** | 0.98 | 2.29 | 6.32 | 7.09 |
| 00848-ziup5kvtCCR__cross_anchor__0128-1__mug | perception_wall | 0.61 | 0.51 | - | 9.04 | 0.69 |
| 00848-ziup5kvtCCR__cross_anchor__0128-1__pitcher | never_in_view | 0.64 | 0.56 | 7.17 | 9.92 | 8.74 |
| 00848-ziup5kvtCCR__cross_anchor__0128-1__scissors | perception_wall | 0.85 | 0.71 | 5.91 | 10.74 | 14.26 |
| 00848-ziup5kvtCCR__cross_anchor__0128-2__cracker_box | never_in_view | 0.89 | 0.79 | 7.67 | 15.10 | 12.23 |
| 00848-ziup5kvtCCR__cross_anchor__0128-2__pitcher | perception_close | 0.59 | 0.49 | 1.28 | 9.20 | 5.88 |
| 00848-ziup5kvtCCR__cross_anchor__0128-2__plate | never_in_view | 0.91 | 0.81 | 8.02 | 3.04 | 0.97 |
| 00848-ziup5kvtCCR__cross_anchor__0128-2__scissors | perception_wall | 0.80 | 0.72 | - | 12.69 | 1.22 |
| 00848-ziup5kvtCCR__cross_anchor__0129-1__cracker_box | never_in_view | 0.64 | 0.54 | 7.58 | 12.58 | 12.24 |
| 00848-ziup5kvtCCR__cross_anchor__0129-1__pitcher | committed_elsewhere | 1.00 ** | 0.86 | 1.99 | 1.99 | 0.79 |
| 00848-ziup5kvtCCR__cross_anchor__0129-1__scissors | never_in_view | 0.53 | 0.43 | - | 14.87 | 0.84 |
| 00848-ziup5kvtCCR__in_anchor__0116__mug | perception_wall | 0.32 | 0.20 | - | 2.36 | 0.87 |
| 00848-ziup5kvtCCR__in_anchor__0116__pitcher | committed_elsewhere | 1.26 ** | 1.12 | 1.39 | 8.04 | 1.64 |
| 00848-ziup5kvtCCR__in_anchor__0117__banana | committed_elsewhere | 1.07 ** | 0.93 | 1.08 | 8.04 | 0.91 |
| 00848-ziup5kvtCCR__in_anchor__0117__mug | perception_wall | 0.36 | 0.26 | - | 4.02 | 0.73 |
| 00848-ziup5kvtCCR__in_anchor__0117__pitcher | committed_elsewhere | 1.00 | 0.89 | 1.02 | 3.07 | 2.43 |
| 00848-ziup5kvtCCR__in_anchor__0117__scissors | never_in_view | 0.45 | 0.40 | - | 2.38 | 0.79 |
| 00848-ziup5kvtCCR__in_anchor__0118__cracker_box | committed_elsewhere | 1.30 ** | 1.16 | 1.80 | 4.02 | 10.14 |
| 00848-ziup5kvtCCR__in_anchor__0118__mug | perception_wall | 0.41 | 0.32 | - | 1.85 | 0.60 |
| 00848-ziup5kvtCCR__in_anchor__0118__pitcher | committed_elsewhere | 1.16 ** | 1.06 | 1.16 | 8.47 | 2.17 |
| 00848-ziup5kvtCCR__in_anchor__0118__plate | committed_elsewhere | 1.42 ** | 1.32 | 1.48 | 10.13 | 5.76 |
| 00848-ziup5kvtCCR__in_anchor__0118__scissors | never_in_view | 0.82 | 0.72 | - | 4.35 | 4.18 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-1__bowl | committed_elsewhere | 0.72 | 0.62 | 4.66 | 4.66 | 4.71 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-1__plate | committed_elsewhere | 0.66 | 0.56 | - | 2.30 | 4.94 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-1__scissors | perception_close | 0.51 | 0.40 | - | 3.77 | 7.12 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-2__plate | committed_elsewhere | 0.53 | 0.44 | 1.59 | 5.98 | 0.96 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-2__scissors | perception_wall | 0.91 | 0.80 | - | 2.02 | 0.93 |
| 00880-Nfvxx8J5NCo__cross_anchor__0128-2__soup_can | perception_wall | 1.06 ** | 0.92 | 3.59 | 10.55 | 1.00 |
| 00880-Nfvxx8J5NCo__cross_anchor__0129-1__pitcher | committed_elsewhere | 1.83 ** | 1.71 | 1.88 | 12.38 | 7.07 |
| 00880-Nfvxx8J5NCo__cross_anchor__0129-1__scissors | perception_close | 0.60 | 0.50 | - | 6.20 | 6.08 |
| 00880-Nfvxx8J5NCo__cross_anchor__0129-1__soup_can | committed_elsewhere | 0.59 | 0.50 | 2.55 | 12.10 | 4.12 |
| 00880-Nfvxx8J5NCo__in_anchor__0116__pitcher | committed_elsewhere | 0.97 | 0.85 | 1.03 | 6.46 | 0.98 |
| 00880-Nfvxx8J5NCo__in_anchor__0116__plate | committed_elsewhere | 1.26 ** | 1.16 | 1.28 | 10.76 | 14.04 |
| 00880-Nfvxx8J5NCo__in_anchor__0117__cracker_box | near_miss | 1.09 ** | 0.97 | 1.09 | 1.09 | 2.76 |
| 00880-Nfvxx8J5NCo__in_anchor__0117__plate | committed_elsewhere | 1.82 ** | 1.75 | 1.85 | 16.05 | 14.22 |
| 00880-Nfvxx8J5NCo__in_anchor__0117__scissors | perception_wall | 0.50 | 0.40 | - | 4.41 | 0.57 |
| 00880-Nfvxx8J5NCo__in_anchor__0117__soup_can | committed_elsewhere | 0.48 | 0.29 | 1.26 | 5.17 | 0.98 |
| 00880-Nfvxx8J5NCo__in_anchor__0118__cracker_box | committed_elsewhere | 1.18 ** | 1.08 | 1.18 | 3.08 | 2.89 |
| 00880-Nfvxx8J5NCo__in_anchor__0118__plate | never_in_view | 0.71 | 0.61 | - | 2.07 | 13.40 |
| 00880-Nfvxx8J5NCo__in_anchor__0118__scissors | perception_wall | 0.28 | 0.18 | - | 14.88 | 0.53 |

`**` marks a trial with no navmesh point within 1 m of the object.
