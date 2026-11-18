# What the fused pipeline is worth, measured

Six arms, paired episode for episode against the baseline. Every flag defaults
off, so each arm is an explicit assertion and the baseline is reproducible.

Read the mechanism counters first. This benchmark's noise floor is about three
trials, so a swing of that size in success rate says nothing on its own.

## The arms

| arm | preset | what it adds |
|---|---|---|
| base | `ycb_authored_15` | the shipped configuration |
| containers | `..._containers` | a support surface is measured from its own floor |
| fused | `..._fused` | test the stale anchor before choosing a storey |
| v3 | `..._fused_v3` | + a standing request stops vetoing this floor; remembered staircase |
| v4 | `..._fused_v4` | + the LLM answers the storey question |
| v5 | `..._fused_v5` | + containers fix, and a storey score that is not a furniture count |

## 1. Scene 00808, sixteen episodes, all six arms

Every arm scored 6 of 16, and **no episode's outcome differed from base in any
arm**. The mechanisms, meanwhile, moved a long way.

| counter | base | containers | fused | v3 | v4 | v5 |
|---|---:|---:|---:|---:|---:|---:|
| surfaces inspected | 0 | 0 | 0 | 0 | 0 | 23 |
| cross-floor requests | 67 | 59 | 33 | 33 | 28 | 30 |
| directed switch attempts | 43 | 35 | 9 | 9 | 2 | 1 |
| storey requests held | 0 | 0 | 4 | 4 | 4 | 0 |
| LLM asked / said stay | 0 | 0 | 0 | 0 | 10 / 10 | 9 / 9 |

The one genuinely new behaviour is v5's 23 surface inspections against the
baseline's zero: the container posterior running on an upper storey for the
first time. It converted nothing here.

00808 is an awkward scene for this question. Split by which floor the episode
starts on, the baseline scores 6 of 8 downstairs and 0 of 8 upstairs, and six of
those eight upstairs failures never get the target into view at all.

## 2. Scene 00814, fourteen episodes

Base 2 of 14, containers 2 of 14, no episode changed. The container fix also
introduced five cross-floor requests and five directed switches that the
baseline did not make, because an upper floor with surfaces now has mass to
compete with.

00814 has **no cross-floor relocations at all**, which only became visible after
the manifest instrument was fixed.

## 3. The cross-floor half, which is the actual question

`ycb.cross_floor_relocations_only=true` over the four scenes that have both a
prior map and a cross-floor relocation. Eight paired episodes, seven of them
genuinely cross-floor.

| | base | v5 |
|---|---:|---:|
| success, cross-floor only | 0/7 | 0/7 |
| reached the object's floor | 2/8 | 2/8 |
| surfaces inspected | 7 | 16 |
| LLM asked / overrode / stayed | 0 | 12 / 0 / 3 |

Where the seven are lost:

- **Three never tried.** 00800 twice and 00810 once: no portal ever seen, no
  switch attempted, vertical range 0.00 to 0.19 m. `floor.use_prior_stairs`
  cannot help, because `try_switch` returns before the portal block when the
  gate refuses and no directed request exists, so the remembered staircase is
  never consulted.
- **One tried fifty-five times.** 00821's cracker box saw 14 portals, made 55
  switch attempts, and ascended 0.17 m.
- **One ascended part way.** 00808's toy airplane climbed 0.4 m under base and
  1.29 m under v5, and arrived nowhere.
- **One arrived and could not stop.** 00808's yellow bottle climbed 3.2 m to the
  right floor, saw the target for 20 frames at 0.43 m with a detector score of
  0.94, and never issued STOP before the budget ran out.

## 3b. A seventh arm, and a clear regression

`v6` adds `floor.prefer_prior_stairs`: use the staircase the prior map recorded
*ahead of* any portal the agent can currently see, rather than only when it sees
none. The mechanism fired 106 times over the same eight episodes and made things
worse.

| | base | v5 | v6 |
|---|---:|---:|---:|
| success, cross-floor only | 0/7 | 0/7 | 0/7 |
| reached the object's floor | 2/8 | 2/8 | **1/8** |
| remembered staircase preferred over a portal | 0 | 0 | 106 |

The episode it changed, it broke. 00808's yellow bottle climbed 3.2 m and reached
the right floor under base and v5, and rose 1.377 m and did not under v6.
00821's cracker box went from 0.17 m of ascent to none.

A remembered mouth is one point from one pass-1 traversal. Driving to it from
wherever the agent happens to be can be far worse than driving to a staircase it
can see right now. The fallback form costs nothing because it only runs where
there was no alternative; the preference form discards live evidence. Keep
`use_prior_stairs`, leave `prefer_prior_stairs` off.

## 3c. All three scenes together

00808, 00814 and 00800, baseline against the full fused stack, paired.

| | base | v5 |
|---|---:|---:|
| success | 11/42 | 11/42 |
| reached the object's floor | 39/42 | 39/42 |
| episodes at budget | 31 | 31 |
| **episodes whose outcome changed** | | **0** |

Per scene: 00808 6 and 6, 00814 2 and 2, 00800 3 and 3.

The mechanisms were live throughout: 11 storey requests held for an untested
anchor, 13 rounds where the posterior declined to choose a storey, 31 questions
put to the model. Not one episode ended differently.

## 4. What this says

**The decision layer is sound where it is exercised; the execution layer is
not.** On 00821 the model was asked nine times and its stated reason correctly
identified the storey holding the cracker box. `FloorStack.up()` resolved that to
the right floor key, the posterior agreed, and the agent still never climbed. A
storey choice cannot register while the climb fails.

**The LLM neither helped nor hurt, and its trigger is now right.** It was asked
21 times across the runs and overrode the posterior zero times. Its stated
reasoning is on-topic and, where checkable, correct. This is the first
measurement of it in this agent at all: `decide()` previously had no caller.

**The benchmark cannot answer the question at this size.** The whole 15-scene
set holds 19 cross-floor relocations, all in `cross_anchor` at layout index 1
and spread over 9 scenes; an in-anchor move is never cross-floor. Seven of them
have both a prior map and a run here.

## 5. What to do next

1. **Fix the climb.** Why 55 switch attempts produce 0.17 m of ascent is the
   whole cross-floor question now. Nothing in the decision layer can show up
   until this moves.
2. **Let the prior staircase be reachable.** Consult `stair_edges` when the gate
   refuses, not only when `find_portals` comes back empty, so the three episodes
   that never tried at least try.
3. **Close the last metre upstairs.** One of seven was reached and lost to a
   missing STOP.
4. **Build the five missing prior maps** (00878, 00869, 00873, 00824, 00871) to
   take the cross-floor sample from 7 to 19.
