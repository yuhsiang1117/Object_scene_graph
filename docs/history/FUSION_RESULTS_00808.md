# Fused pipeline A/B -- `outputs/fusion_iter`

Episodes per arm: base 16, containers 16, fused 16, v3 16, v4 16, v5 16

Paired across every arm: 16

## 1. Outcome, paired

| split | episodes | base SR | containers SR | fused SR | v3 SR | v4 SR | v5 SR | base floor | containers floor | fused floor | v3 floor | v4 floor | v5 floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cross_floor cross_anchor | 2 | 0/2 = 0.0% | 0/2 = 0.0% | 0/2 = 0.0% | 0/2 = 0.0% | 0/2 = 0.0% | 0/2 = 0.0% | 1 | 1 | 1 | 1 | 1 | 1 |
| same_floor cross_anchor | 6 | 2/6 = 33.3% | 2/6 = 33.3% | 2/6 = 33.3% | 2/6 = 33.3% | 2/6 = 33.3% | 2/6 = 33.3% | 6 | 6 | 6 | 6 | 6 | 6 |
| same_floor in_anchor | 8 | 4/8 = 50.0% | 4/8 = 50.0% | 4/8 = 50.0% | 4/8 = 50.0% | 4/8 = 50.0% | 4/8 = 50.0% | 8 | 8 | 8 | 8 | 8 | 8 |
| all | 16 | 6/16 = 37.5% | 6/16 = 37.5% | 6/16 = 37.5% | 6/16 = 37.5% | 6/16 = 37.5% | 6/16 = 37.5% | 15 | 15 | 15 | 15 | 15 | 15 |

## 2. Mechanism counters (sum over paired episodes)

| counter | base | containers | fused | v3 | v4 | v5 |
|---|---:|---:|---:|---:|---:|---:|
| cross_floor_request_held | 0 | 0 | 4 | 4 | 4 | 0 |
| floor_mass_no_opinion | 0 | 0 | 0 | 0 | 0 | 0 |
| prior_stair_switch_attempts | 0 | 0 | 0 | 0 | 0 | 0 |
| floor_unreachable_fallback | 0 | 0 | 0 | 24 | 26 | 29 |
| floor_llm_asks | 0 | 0 | 0 | 0 | 10 | 9 |
| floor_llm_override | 0 | 0 | 0 | 0 | 0 | 0 |
| floor_llm_stay | 0 | 0 | 0 | 0 | 10 | 9 |
| cross_floor_search_requests | 67 | 59 | 33 | 33 | 28 | 30 |
| directed_floor_switch_attempts | 43 | 35 | 9 | 9 | 2 | 1 |
| floor_switch_attempts | 47 | 39 | 13 | 13 | 6 | 2 |
| absence_checks | 2 | 1 | 2 | 2 | 2 | 1 |
| absence_abandon | 2 | 1 | 2 | 2 | 2 | 1 |
| stale_anchor_stop | 0 | 0 | 0 | 0 | 0 | 0 |
| search_surface | 0 | 0 | 0 | 0 | 0 | 23 |
| portals_seen | 51 | 44 | 51 | 51 | 51 | 44 |
| storey request before any absence test | 5 | 5 | 5 | 5 | 5 | 5 |
| median steps | 500 | 500 | 500 | 500 | 500 | 500 |
| episodes at budget | 10 | 10 | 10 | 10 | 10 | 10 |

## 3. Episodes whose outcome changed against base

- **containers**: +0 / -0
- **fused**: +0 / -0
- **v3**: +0 / -0
- **v4**: +0 / -0
- **v5**: +0 / -0
