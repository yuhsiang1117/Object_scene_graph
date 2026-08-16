# 多楼层导航：方法与结构说明

本文说明 `osg` 目前的多楼层支持是**怎么做的**、**每个模块负责什么**、以及**为什么这样设计**。
英文版的文献综述、实验数据与失败结论见 **[MULTI_FLOOR.md](MULTI_FLOOR.md)**；
本文侧重实现结构，供后续修改代码时参考。

---

## 1. 问题：为什么原本跨楼层必然失败

HM3D ObjectNav 的 v1 验证集里，**20 个场景有 14 个是多层建筑**，
2000 个 episode 中 **411 个（20.6%）** 的目标物体不在起始楼层上——
也就是说 agent 不上楼就不可能成功（用 `scripts/scene_floors.py` 实测）。

改造之前，这类 episode 的成功率是 **0/24，而且一次楼层切换都没有发生过**。
原因不是「爬楼梯失败」，而是**根本没尝试过**。根源有两条：

1. **上层楼永远不会被建图。**
   `Costmap2D.update` 用「相对地面高度」来筛点：只保留
   `floor_y + 0.15 ≤ h < floor_y + 1.5` 的点当障碍物。
   而 `floor_y` 在 episode 的**第一帧就被锁死**，之后再不更新。
   agent 一旦上楼（高出 2.8 m），所有观测都落在波段之外被直接丢弃 →
   上层楼没有任何栅格 → 没有 frontier → 没有可探索的东西。

2. **楼梯几何会污染下层楼的地图。**
   agent 走到楼梯中段时，它周围的几何又回落到波段内，
   于是楼梯踏面和从楼梯上看到的东西，被盖在**下层楼**地图的同一个 (x, z) 上。
   而 `_raycast_batch` 写入 `OCCUPIED` 之后**永不清除**，这个污染是永久的。

此外还有一个隐蔽的 bug：`sim/habitat_env.py` 的 `action_to_goal` /
`is_reachable` 会把 agent **自己当前的高度** `pos[1]` 塞进任何 2D 目标点再去
`snap_point`。所以「楼上的椅子」会被吸附到楼下地板，`find_path` 失败，
`_check_candidates` 判定它不可达并拉黑——**每一个跨楼层目标都被主动丢弃了**。

---

## 2. 总体结构

```
                    ┌──────────────────────────────────────┐
   每帧相机高度  ──▶ │ FloorEstimator   mapping/floors.py   │
                    │  当前在第几层 / 是否在楼梯上          │
                    └───────────────┬──────────────────────┘
                                    │ floor_id
                    ┌───────────────▼──────────────────────┐
   深度点云     ──▶ │ FloorStack       mapping/floor_stack │
                    │  每层一张 Costmap2D + 房间分割        │
                    │  .costmap 属性 = 当前层的地图         │
                    └───────────────┬──────────────────────┘
                                    │ 一张普通 2D 地图
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  frontier 提取              find_portals                 SceneGraph
  planner / 控制器         mapping/portals.py         floor→room→object
  viewpoint / 可视化        「哪里能看到别层」          graph/scene_graph.py
                                    │                           │
                                    ▼                           ▼
                        FloorSwitchPolicy  ◀──── floor_target_evidence
                        「现在该不该换层」          graph/priors.py
                                    │
                                    ▼
                        3D 导航目标 sim/habitat_env.py
                        （navmesh 自己走楼梯）
```

**设计上最关键的一点**：`FloorStack` 的每一层仍然是一张**普通的、同一 (x, z) 平面上的
`Costmap2D`**。`NavAgent.costmap` 变成一个 property，返回当前层的地图：

```python
@property
def costmap(self) -> Costmap2D:
    return self._floor_stack.costmap
```

这就是整个改造的**唯一接缝**。planner、frontier 提取、房间分割、viewpoint 规划、
控制器、调试可视化——全部**一行都不用改**，它们拿到的仍然是一张 2D 地图。
「楼层」只是一个字典的 key，从来不是坐标约定的改变。

---

## 3. 各模块详解

### 3.1 FloorEstimator —— 现在在第几层（`mapping/floors.py`）

**主信号是 agent 自己的站立高度**（`camera_position[1] - camera_height`），
不是深度点云。理由很直接：**agent 只会站在地板上**，所以每一个采样天然就是
一次「地板高度」观测；而深度直方图在天花板、桌面上同样会出峰。
点云直方图（HOV-SG 的做法）只作为**可选的次要来源**，
用来预登记「看到过但还没去过」的楼层。

楼层 id 是**稳定的**：后来才发现的地下室会拿到新 id，而不是把已有的重新编号。
所有以楼层 id 为 key 的东西（地图、房间 id、frontier、黑名单）否则会被悄悄指到别层。

**两条独立的「确认为新楼层」路径**，满足其一即可：

| 条件 | 参数 | 含义 |
|---|---|---|
| 与已知层高差足够大 | `new_level_m = 1.8` | 一整层楼 |
| 或：在该高度上**横向走得够远** | `min_horizontal_run_m = 2.5` | 楼梯休息平台走不了这么远 |

为什么需要第二条？因为单一阈值**无解**：

- 阈值太低（0.6）→ 一次下楼被登记成 **4 层**，凭空造出 1.879 m 和 1.253 m
  两个「楼层」，实际上那是**楼梯休息平台**。地图会在楼梯中间被劈开，
  劈出来的那一层没有任何可达 frontier，episode 直接卡死——**比不改还糟**。
- 阈值 1.8 → 平台被正确排除，但 agent 爬到 1.6 m 就折返的真实爬楼**也确认不了**。

横向位移把两者干净地分开：**1.2 m 的休息平台把你困在 1.2 m 之内，
真正的楼层则可以任意走开**。注意必须是**位移**（离第一次到达该高度的点多远），
不是路径长度——在平台上来回踱步，路径长度轻易就上几十米。

其他参数：`level_tol_m = 0.35`（在这个范围内算「站在该层上」，
超出即 `on_stairs`，此时楼层 id **冻结**）、`min_dwell_steps = 6`（去抖）、
`merge_m = 0.6`（下沉客厅、门槛这类高差绝不算新楼层）。

层高还会**在线修正**：新层第一次登记时往往是在楼梯半途（实测：真实楼面 0.196 m
却登记成 0.832 m）。0.64 m 的误差超过 `level_tol_m`，会导致 agent 站在真实
楼面上却一直被判成「在楼梯上」。因此每层维护一个采样窗口，取**中位数**
作为层高——中位数会收敛到 agent 待得最久的平地，而不是少数几级台阶。

### 3.2 FloorStack —— 每层一张地图（`mapping/floor_stack.py`）

每层持有：`Costmap2D`、**独立的房间分割器**、`room_labels`、首次到达点 `entry_xy`。

**房间分割器为什么必须每层一个**：`VoronoiRoomSegmenter._stabilize_ids` 靠
2D 重叠把房间和上一次的结果对上。共用一个分割器时，
**正上方的房间会仅仅因为「在正上方」就继承下方房间的 id**——
连同缓存的 LLM 房间标签一起。分成多个实例这个 bug 就自动消失，
`_stabilize_ids` 本身一行都不用动。

但房间 id 必须**全局唯一**（`SceneGraph.rooms` 是一个扁平字典，
LLM 房间标签缓存也按 id 索引），所以各层分割器**共享一个 `RoomIdCounter`**。

其他随之楼层化的东西：`Frontier.floor` 字段、frontier 黑名单、放弃点
（`_last_giveup_pt`）。这些都**无条件**带上楼层，单层场景下该字段恒为 0，
比较退化为恒真——不需要维护第二条代码路径。

### 3.3 Portals —— 哪里能看到别的楼层（`mapping/portals.py`）

**这里刻意绕开了楼梯检测。** 基于高度梯度的几何楼梯检测**是失败的**
（见 §6），所以跨楼层探索不依赖它。

`Costmap2D` 可选地维护一层 **height layer**：每个格子记录**观测到的最低表面高度**
（取最小值，这样桌面、天花板都盖不住底下的地板）。
它的记录范围是 `|相对高度| < height_span = 3.5 m`，**上下都记**，
和障碍物波段无关——因为：

- 向上要放得下**整段楼梯**（HM3D 层高 2.5–3.4 m），
  用障碍物波段的 1.5 m 会把楼梯截断；
- 向下是**唯一**能看见下行楼梯口的办法，障碍物波段在脚下 0.3 m 就截止了。

**portal = height layer 中「离当前层一整层远」的连通块**：

| 参数 | 值 | 作用 |
|---|---|---|
| `min_delta` | `new_level_m = 1.8` | 低于此是错层/家具，不是楼层 |
| `portal_max_delta_m` | 4.0 | 排除隔两层的读数（中庭） |
| `portal_min_cells` | 20 | 去噪 |
| `merge_m` | 1.5 | 同一个楼梯井从多个位姿看到会产生多块 |

portal **不是楼梯本身**，而是「朝那儿走」的一个目标点；真正爬楼梯由 navmesh 完成。

### 3.4 FloorSwitchPolicy —— 现在该不该换层（`mapping/portals.py`）

采用 **ASCENT 的门控**，而不是 MFNP 的加权评分。理由是针对本仓库的：
MFNP 的评分需要一个 LLM 项，而 `INVESTIGATION.md` 记录了
**LLM frontier 评分在这里产出的轨迹与几何启发式逐字节相同**；
它另外两项还需要新的累加器，其权重要在 n=65 上调——那里的标准误约 6 个点，根本调不动。

**两条硬约束**（任何情况下都不能违反）：

- `no_switch_after_frac = 0.7` → 第 350 步之后不再换层（换了也没预算搜）
- `switch_min_interval = 50` → 两次切换至少间隔 50 步（防止来回横跳）

**几何路径**（`use_target_evidence` 关闭时的唯一规则）：
当前层已经没有近的 frontier 了（`best_path_cost > near_frontier_m = 4.0`，
或者根本选不出来），且 `step ≥ no_switch_before = 50`。

**目标类别证据**（默认开启，见 §3.5）会把这个决定**双向**修正：

| 情况 | 行为 | 参数 |
|---|---|---|
| 本层已建图够多，却几乎没有目标的「同伴物体」 | **提前走** | `min_objects_to_judge = 8`, `early_switch_step = 30` |
| 本层看起来就是目标该在的地方 | **留下**，即使没有近 frontier 了 | `strong_evidence = 2` |
| 「留下」但已经找了很久还是没有 | 证据**过期**，恢复几何规则 | `evidence_patience_steps = 120` |
| 本层已经建图到**目标类别本身** | **绝不离开** | 证据 ≥ 10 |

为什么需要「过期」？因为**本层有个卫生间，不代表本层这个卫生间里有马桶**。
早期版本没有过期，「留下」规则把 agent 永久钉在原地，
跨楼层切换尝试从 14/24 掉到 **3/24**。
同样地，早期要求证据**恰好为 0** 才能提前走也太严——
几乎任何楼层都有一两个偶然的同伴物体。

### 3.5 类别上下文先验（`graph/priors.py`）

一张固定的**共现表**，覆盖 6 个 ObjectNav 目标类别：

```
toilet     ← sink, bathtub, shower, towel, mirror
bed        ← pillow, cushion, nightstand, wardrobe, dresser, lamp
sofa       ← cushion, tv_monitor, table, lamp, fireplace
tv_monitor ← sofa, cushion, cabinet, table, fireplace
chair      ← table, desk, cabinet, shelf
plant      ← （空）
```

`floor_target_evidence(scene_graph, floor_id, target)` 数的是**该层上出现了
几种不同的同伴类别**，目标类别本身额外 +10；同时返回该层的物体总数，
调用方需要它来区分「确实不在这层」和「还没看过这层」。

`plant` 故意留空：盆栽在任何房间都会出现，它的缺席不提供信息，
硬造一个先验只会给最弱的类别（16.7%）增加噪声。此时返回 `None`，
门控退回纯几何规则。

**刻意不用 LLM。** 这份表是免费的、可检视的、可单测的，而 LLM 版本在本仓库
已被证明与几何启发式产出逐字节相同的轨迹。这也是 Stage 6 里
`floor_id` 字段唯一的真实消费者。

### 3.6 3D 导航目标（`sim/habitat_env.py`）

`action_to_goal` / `is_reachable` 现在接受 3D 目标点，或额外的 `floor_y`：

```python
def _goal3d(self, goal, floor_y=None):
    g = np.asarray(goal, float).ravel()
    if g.size == 3:
        return g.astype(np.float32)
    y = agent_y if floor_y is None else float(floor_y)
    return np.array([g[0], y, g[1]], dtype=np.float32)
```

`agent.navmesh_3d_goals` 打开后，候选目标按**它自己所在楼层的高度**吸附，
而不是 agent 的高度。注意吸附高度用的是**楼面**而非物体椭球中心——
物体中心浮在地面之上 0.3–1.0 m，在夹层边缘足以吸附到错误的楼层。

**一个曾经踩过的坑**：高度是否传递，最初是按 `state == GOTO_FRONTIER` 判断的。
这在 Stage 3 写的时候是对的（frontier 目标都在本层），
但 Stage 5 让 portal 追逐**复用了同一个 state**，于是 portal 的目标高度被丢掉，
被吸附到夹层**正下方**的地面。agent 走过去、「到达」、一厘米也没爬升，
追逐被当作「没有垂直进展」放弃——100 个 episode 里 35 次放弃中有 26 次是这个原因。
现在按**目标是否跨层**（`_portal_active`）判断，并有回归测试双向锁定。

### 3.7 场景图的楼层层级（`graph/scene_graph.py`）

新增 `FloorNode`，`RoomNode` 和 `ObjectNodeView` 各加一个 `floor_id`。
物体的楼层由它**自己的 3D 中心高度**决定——这正是 `rebuild()` 以前**丢掉**的
`center[1]`。在此之前，楼上的床和它正下方的床对所有消费者来说是同一个东西。

`to_prompt_text(group_by_floor=True)` 会按**高度升序**（不是 id 顺序，
因为 id 是创建顺序）把房间嵌套在 `Floor N [y=...]` 之下。

目前**房间**的楼层是按「其中多数物体在哪层」投票决定的：
共用一张 costmap 时，房间是一个 2D 区域，本身无法按高度切分。
等每层都有自己的 costmap 之后，应改为逐层做房间分割。

---

## 4. 一次跨楼层的完整流程

```
 1. 每一帧：FloorEstimator.update(相机高度, xy) → floor_id
 2. FloorStack 切到该层 → costmap.update() 只喂这一层的几何
 3. 选 frontier 时：本层没有近的 frontier 了？或类别证据说「不在这层」？
 4. find_portals(当前层 costmap, floor_y) → 高度层里离本层一整层远的连通块
 5. 选最近的、且优先没去过的楼层的 portal；用 is_reachable(xy, target_y) 确认
 6. 设为目标，_portal_active = True，deadline = +120 步
 7. 用 navmesh 驱动（目标按 target_y 吸附）→ navmesh 自己走楼梯
 8. 追逐期间锁住目标：只要在爬升或 on_stairs，就不许重新选本层 frontier
    （否则 agent 会挑一个楼下的 frontier 走回去——实测有 3 个 episode
      爬到 1.6 m 就这样折返了）
 9. 到达新层：FloorEstimator 确认新层 → FloorStack 换图 → 追逐结束
10. 在新层上正常探索（一张全新的空地图）
```

---

## 5. 配置一览

全部默认**关闭**，由 `+experiment=full_v1_navmesh` 打开。

```yaml
floor:
  enabled: true            # 楼层估计
  estimate_only: false     # false = 真的拿去用（true 只记录，便于先验证估计器）
  per_floor_costmap: true  # 每层一张地图
  cross_floor: true        # portal + 换层决策
  stairs: false            # 几何楼梯检测——不工作，见 §6

agent:
  navmesh_3d_goals: true   # 目标按其所在楼层吸附

exploration:
  frontier_cost_free_cell: true   # frontier 排序代价算到自由格（见下）
```

---

## 6. 实测结果与两个失败结论

### 结果（full v1，100 episodes）

| 配置 | SR | SPL | 单层 | 多层 | 跨层 |
|---|---|---|---|---|---|
| 全部关闭 | 40.0 | 0.200 | 68.6 | 24.6 | **0.0** |
| + 楼层/每层地图/3D 目标 | 42.0 | 0.211 | 68.6 | 27.7 | 0.0 |
| + 类别定时换层 | 44.0 | 0.217 | 71.4 | 29.2 | 0.0 |
| + frontier 代价修正 | 46.0 | 0.215 | 71.4 | 32.3 | 4.2 |
| + 放宽换层门控 | 46.0 | 0.228 | 68.6 | 33.8 | **16.7** |
| + 终端 creep | **48.0** | **0.234** | 68.6 | **36.9** | 16.7 |

**回归门槛**：35 个单层 episode 在开启所有开关后必须**逐字节一致**
（实测 35/35 通过）。注意这个对比**必须用 `verification=off`**——
托管 VLM 校验器是不可复现的，相同代码相同配置两次运行会在 4 个 episode 里
差 2 个（详见 `INVESTIGATION.md` 的 Method 一节）。

### 失败结论 1：几何楼梯检测不工作

判据是「格子与 8 邻域的最大高差在 `min_dh` 和 `climb_limit` 之间」。
问题是**真实楼梯的踏面是平的**：踏面内部 `Δh ≈ 0`，低于 `min_dh` 被排除，
只剩下竖板边缘这些**互不相连的细条**。实测一段 0.28 m 踏面 / 0.17 m 竖板的
合成楼梯：**碎成 9 块，最大 200 格，检测到 0 个区域**；
而同样坡度的**斜坡**只有 1 个连通块、1500 格，检测得好好的。

阈值取 0.3 会在 35 个单层 episode 里的 28 个上误触发（重标 43k 个格子，
每块只升高 0.33–0.75 m，其实是门槛、斜地面和矮家具）；
取 1.0 则**哪里都不触发**，包括 agent 确实爬了楼的 episode。中间没有可用值。

根本原因是把 ZONDA 的判据用反了：ZONDA 用 `Δh < H_agent` 作**可通行性过滤**
（平地也通过），再靠**语义标签**挑出楼梯。这里却把 `Δh` 本身当成了检测器。
`test_discrete_treads_are_detected` 是一个 strict xfail，专门钉住这个缺陷。

顺带一提：在最佳配置下 costmap **并不参与运动**（`_follow_path` 交给 navmesh），
所以就算楼梯掩码能工作，它影响的也只是 frontier 提取，不是「能不能走上去」。

### 失败结论 2：上下文先验不能用来把关「提交目标」

最大的单项损失是走向**完全错误的物体**：34 次 approach 失败里
**22 次终点离目标 >3 m**。VLM 校验器抓不到它们（78 次拒绝仍然放行），
因为它们是**类别正确的错误实例**。设想用上下文把关：
被沙发和电视围着的马桶不会是卫生间那个马桶。

**被证伪，而且原因是结构性的**：放宽阈值几乎没有提高触发率（22 → 24，
预期是翻倍）。真正的约束是——**agent 在房间被建图之前就提交了**
（track 一满足 `min_obs` 就提交），决策发生的那一刻根本没有上下文可查。
两组共 46 次拒绝，把目标指标从 22 → 21 → 22，等于没动；
放宽的那一组还把真目标误杀成「从未提交」（explore-fail 20 → 23）。

未验证的变体：把这个门控放到**终点 STOP** 而不是提交时——那时房间已经建好图了。

### 失败结论 3：ASCENT 的 coarse-to-fine（粗到细）LLM 推理反而更差

按 `ascent/llm_planner.py` 完整移植为 `exploration/coarse_to_fine.py`：
先由 LLM 选**楼层**（每层的房间类型、已建图物体、是否已探索完，
加上 HM3D train 集统计的楼层先验，允许回答「留在本层」），
再由 LLM 选**区域**（utility 排名前 3 的 frontier，用房间类型 + 周围物体描述）。
两个决策都有门控：3 m 内有 frontier 就直接走，不问 LLM——
这正是 ASCENT 把 LLM 调用压到每 episode 2~3 次的原因。

100 个配对 episode 的结果：

| | baseline | + coarse-to-fine |
|---|---|---|
| SR | **51.0%** | **44.0%** |
| SPL | 0.240 | 0.227 |
| 跨楼层 | 20.8% | 8.3% |
| explore-fail | 18 | **22** |
| 走错物体（>3 m） | 25 | 24 |

2 个变好 / 9 个变坏，McNemar p = 0.065。

**这次不是接口故障**（第一次实验就是被这个毁掉的，见下）：
157 次调用、**0 次错误**、**每 episode 2.49 次**（ASCENT 报告 2.0~2.7），
且 LLM 有 55.9% 的情况维持几何最优解，而随机三选一只会是 33%——
它确实在判断，不是在乱选。损失也精确地落在它插手的地方：
它改变过决策的 58 个 episode，SR 从 41.4% 掉到 32.8%；
它没插手的 42 个，27 → 25（这就是 VLM 校验器的噪声底线）。

**原因是它打断了连续扫掠**。我们的目标函数里有动量项
（`continuity_weight=2.0`），这是本项目历史上最大的单项探索收益（+8.5 SR）；
而 ASCENT 的 fine 步骤直接用一个**既不看距离也不看朝向**的选择
覆盖几何 argmax，于是每次覆盖都打断一次正在进行的追逐。
ASCENT 本身没有动量项可打断。9 个变坏的 episode 里有 5 个
最后是 explore-fail——从头到尾没有提交过任何目标。

**最容易想到的辩解已经被量化排除**：ASCENT 的区域描述带 Places365 房间类型，
我们没有。但在 15 episode 的调试运行里记录的 91 条区域描述中，
带房间标签的是 **0%**，带物体的是 **100%（平均 9.8 个物体）**，
32 次决策没有一次是「三个选项完全一样」，
选项之间物体集合的平均 Jaccard 相似度是 0.48（17% 的配对几乎不相交）。
模型拿到了真实且可区分的上下文，然后做出了更差的选择。

**一条方法论教训**：第一次跑这个 A/B 时，托管端点开始卡住，
每次调用耗掉 `timeout_s`×3（约 6 分钟）后回退到几何选择。
那次实验会正常跑完并报出一个「与 baseline 无差异」的结论——
**因为它本来就是 baseline**。凡是依赖网络组件的 A/B，
都必须逐 episode 记录调用次数和错误数，并在读 SR 之前先检查
（`episodes.jsonl` 里的 `ctf_calls` / `ctf_errors` 就是为此存在的）。

---

## 7. 目前的瓶颈

跨楼层 SR 是 **16.7%（4/24）**，机制本身已经跑通：
**10 次尝试 → 10 次成功换层 → 4 次成功**（转化率 40%）。

真正的瓶颈是 **24 个跨楼层 episode 里有 14 个从头到尾 `portals_seen == 0`**——
agent 压根没走到任何能看见别层的位置。portal 需要观测到另一层的**表面**，
而从楼梯底下往上看，看到的多半是天花板。

两个候选方向：Stage 4e 里写了但从未实现的**低头扫描**；
或者从 navmesh 采样候选高度（`build_navmesh_vertices` 知道每一个可通行高度，
仅限仿真，但运动本来就已经依赖 navmesh 了）。
若把参与率从 10/24 提到 20/24 而转化率不变，跨楼层可到 8/24，
整体 SR 约 **+4 个点**——这是目前已识别的最大空间。
