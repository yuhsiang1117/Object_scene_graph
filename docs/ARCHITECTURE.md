# 系統架構與實作細節

> 對應版本：2026-07-16。本文件說明每個模組的職責、
> 演算法與實作決策的緣由。專案狀態與路線圖見
> [DESIGN_AND_ROADMAP.md](DESIGN_AND_ROADMAP.md)；測試見 [TESTING.md](TESTING.md)。

## 目錄

1. [設計目標與原則](#1-設計目標與原則)
2. [座標與資料約定](#2-座標與資料約定)
3. [資料流總覽](#3-資料流總覽)
4. [core — 基礎型別與幾何](#4-core)
5. [perception — 偵測與 keyframe](#5-perception)
6. [objects — 橢球物件層](#6-objects)
7. [mapping — 佔據地圖、frontier、房間](#7-mapping)
8. [graph — 階層式 scene graph](#8-graph)
9. [llm — LLM/VLM client 與 prompts](#9-llm)
10. [exploration — 評分、非同步、選擇](#10-exploration)
11. [planning — A* 與運動控制](#11-planning)
12. [verification — 視點與目標驗證](#12-verification)
13. [agent — 導航狀態機](#13-agent)
14. [sim — Habitat 封裝](#14-sim)
15. [eval — 評估與視覺化](#15-eval)
16. [Docker 部署與 VRAM 預算](#16-docker-部署與-vram-預算)
17. [Config 參考](#17-config-參考)

---

## 1. 設計目標與原則

以 RA-L 論文（`docs/RA_L__2025_.pdf`）為規格，重建 real-time
object-based hierarchical 3D scene graph ObjectNav 系統，加上三項現代化改進：

| 改進 | 內容 | 實作位置 |
|---|---|---|
| A | YOLOE 取代 YOLO-World（偵測+分割一體、無 SAM） | `perception/detector.py` |
| B | VLM 多模態 frontier 評分（keyframe 影像進 prompt） | `exploration/vlm_scorer.py` |
| C | Target verification / last-mile（視點 + VLM 驗證 + blacklist） | `verification/` |

**設計原則**
- **FrameData 是唯一貨幣**：模擬器（或未來的 ScanNet/真機）與 pipeline
  之間只透過 `FrameData` 交換，換資料來源只需寫新 loader。
- **LLM 永不阻塞控制迴圈**：所有 frontier 評分走 `AsyncScorer`；
  唯一的阻塞呼叫是罕見的目標驗證，單獨計時回報。
- **每模組計時**：real-time 是論文主張，`Profiler` 包住每個階段，
  輸出 `timing.csv`，控制迴圈 FPS 與 LLM 決策延遲分開呈現。
- **Ablation 即 config**：`random/nearest/llm_text/vlm` 評分階梯與
  verification on/off 都是 Hydra config group，一鍵切換。

## 2. 座標與資料約定

這是全專案最容易踩雷的部分，統一如下：

- **相機**：OpenCV pinhole 慣例（z 向前、x 向右、y 向下）。
  Habitat 原生是 OpenGL 慣例（z 向後、y 向上），
  `sim/habitat_env.py` 在建 `FrameData` 時做一次轉換：
  `R_cv = R_gl @ diag(1, -1, -1)`（繞 x 軸轉 180°）。
  之後所有幾何運算（反投影、橢球投影）都在 OpenCV 慣例下進行。
- **世界**：沿用 Habitat 的 y-up 世界系。地面平面取 `(x, z)` 軸
  （`mapping/costmap.py` 的 `PLANE = (0, 2)`），高度軸 `HEIGHT_AXIS = 1`。
- **Heading**：`atan2(f_z, f_x)`，f 為相機 forward（`T_wc[:3, 2]`）。
  注意 habitat 的 `turn_left`（繞 +y 正轉）會**減少**這個 heading，
  所以 controller 中正的 heading 誤差要輸出 `turn_right`
  （`planning/controller.py` 有註解，曾是實際 bug）。
- **深度**：habitat depth sensor 設 `normalize_depth=False`，單位公尺，
  值即 OpenCV 的 z。內參從 HFOV 導出：`fx = W / (2 tan(hfov/2))`。

核心型別（`core/types.py`）：

```python
FrameData(frame_id, rgb(H,W,3)u8, depth(H,W)f32, T_wc(4,4), intrinsics, timestamp)
Detection(label, score, bbox_xyxy(4,), mask(H,W)bool, crop)
CameraIntrinsics(fx, fy, cx, cy, width, height)  # .K() -> 3x3
```

## 3. 資料流總覽

```
HabitatObjectNavEnv.step() ──> FrameData
   │
   ├── 每幀:      Costmap2D.update()          （向量化 raycast）
   │              WaypointController.observe_progress()（卡住偵測）
   │
   └── keyframe（位移>0.25m 或旋轉>30°）:
         YoloeDetector.detect() ──> [Detection]
           └─> ObjectLayer.update()：關聯 → 初始化/觀測 → W2 優化 → 連結
         KeyframeStore.add()（存 jpg，供 VLM 評分）
         每 10 個 keyframe: VoronoiRoomSegmenter.segment()
         SceneGraph.rebuild()（building→room→object）

NavAgent FSM 消費以上狀態:
  frontier 抽取 → AsyncScorer（LLM/VLM）→ select_frontier(argmax P/d)
  → AStarPlanner → WaypointController → habitat 離散動作
  候選目標 → ViewpointPlanner → TargetVerifier(7B VLM) → 終端接近 → stop
```

## 4. core

### `core/types.py`
見 §2。`Detection.crop_from(rgb, pad=8)` 從原圖裁 bbox（供驗證）。

### `core/geometry.py`
- `backproject(depth, intr, T_wc, mask, stride, max_depth)`：
  向量化反投影整張深度圖到世界點雲。
  `x=(u-cx)/fx*d, y=(v-cy)/fy*d, z=d`，再 `pts @ R.T + t`。
- `ellipse_from_mask(mask)`：影像矩擬合橢圓。對均勻填滿的橢圓，
  二階中心矩 = Σ/4，故 `cov = 4 * np.cov(pts.T)`（再加 `+I` 防退化）。
- `bresenham(r0,c0,r1,c1)`：**標準**整數 Bresenham 產生器，保證抵達終點。
  ⚠️ 教訓：手寫誤差項更新曾造成無窮迴圈（pytest 空轉 4 小時），
  任何自製格線走訪都必須用這個共用實作。
- `sqrtm_2x2_spd(M)`：2×2 SPD 閉式平方根
  `(M + √det·I) / √(tr + 2√det)`，供 Wasserstein 殘差。
- `quat_to_matrix` / `rotvec_to_matrix` / `matrix_to_rotvec`：
  避免引入額外四元數套件。

### `core/profiler.py`
`with profiler.timeit("detector"):` 收集樣本；`report()` 給
mean/median/max ms；`write_csv()` 出 `timing.csv`；`fps(name)`。

### `core/config.py`
Hydra structured dataclasses 註冊進 `ConfigStore`（typo 直接 fail）。
`DEFAULT_VOCABULARY`：40 類室內物件，涵蓋全部 6 個 HM3D 目標類別 —
這保證 per-episode 的詞彙表不變（見 §5 的 no-op 設計）。

## 5. perception

### `detector.py` — YoloeDetector（改進 A）

- `ultralytics.YOLOE("yoloe-11s-seg.pt")`，text-prompt 模式，
  一次前向同時輸出 bbox + mask，**不需要 SAM**。
- **dtype 之舞**（除錯結晶，順序不能亂）：
  1. checkpoint 權重是 fp16 → 載入後立刻 `model.model.float()`
     （否則 `get_text_pe` 的 fp32 mobileclip 特徵 × fp16 投影頭會
     dtype mismatch）。
  2. `set_vocabulary()`：類別**正規化**（小寫、底線轉空白）並
     **排序**後才比較 —— 排序讓 per-episode 呼叫
     （`[target] + DEFAULT_VOCABULARY`，target 必在表內）比較恆等
     → no-op。⚠️ 不排序時 dedup 保留首見順序，每個 episode 順序不同
     → 誤判詞彙表變更 → 在被 predict 半精度化的模型上重編碼 → 崩潰。
  3. 真正的詞彙變更：先 `float()` 回 fp32、編碼、
     `model.predictor = None` 強制重建（AutoBackend 快取了 fp16 視圖）。
  4. 編碼完 `del model.model.clip_model` 釋放 572MB 的 mobileclip，
     `torch.cuda.empty_cache()` —— 6GB 卡的關鍵。
- `predict(..., half=True, project="/tmp/yolo_runs")`：
  fp16 推理；project 指到 /tmp 避免在 repo 內產生 root-owned `runs/`。
- `StubDetector`：測試/離線開發用，回傳佇列中的偵測。

### `keyframe.py`
- `KeyframeSelector`：位移 >0.25m（= 一步 forward）或旋轉 >30°
  （= 一次 turn）即為 keyframe → 實務上每個改變 pose 的動作後偵測一次。
- `KeyframeStore`：keyframe 降採樣存 jpg（或記憶體環形緩衝），
  記錄相機位置與朝向；`nearest_facing(target_xy, k)` 回傳
  「最靠近且面向目標」的 k 張（餘弦 >0.3 過濾），供 VLM 評分。

## 6. objects

### `ellipsoid.py` — dual quadric 表示

橢球（中心 t、旋轉 R、半軸 a,b,c）的對偶二次曲面：

```
Q* = Z diag(a², b², c², -1) Zᵀ,   Z = [[R, t], [0, 1]]
```

投影到影像（P = K [R_cw | t_cw]）得對偶圓錐 `C* = P Q* Pᵀ`；
歸一化 `C*[2,2] = -1` 後可讀出：

```
μ = -C*[:2, 2]          （橢圓中心）
Σ = C*[:2,:2] + μμᵀ     （橢圓形狀矩陣，(x-μ)ᵀΣ⁻¹(x-μ)=1）
```

`project()` 做 cheirality 檢查（中心 z < 0.1 回 None）與特徵值檢查
（非 SPD / 過大回 None）。

**初始化**（`init_from_detection`，論文配方）：
mask 影像矩 → 2D 橢圓；mask 內隨機取 ≤50 個有效深度取平均 → z；
橢圓中心反投影 → 3D 中心；半軸 = 橢圓半軸 × z/f（第三軸取兩者平均）；
R 初始化為相機旋轉。半軸 clip 到 [0.01, 3.0] m。

### `association.py` — 資料關聯

每個 keyframe：把既有 track 的橢球投影回影像，與新偵測配對。

- **分數**（論文公式，對大型物件魯棒）：
  `score = max(I/A₁, I/A₂)`，I 用「橢圓外接 bbox 交集」近似。
- **深度一致性閘**：`|median_depth(mask) − depth(投影中心)| < 0.5m`，
  另以深度誤差微調分數（同類多實例時選深度最接近者 —— 取代 VOOM 的
  「離相機最近」啟發式）。
- 同 label 才可配對；分數 ≥0.4 進入貪婪配對；未配對偵測 → 新 track。
- `ObjectTrack` 記錄 `best_score / best_bbox_px / best_crop /
  best_cam_xy`（最佳偵測的相機位置 —— 被證明「可達且看得見物件」，
  是終端停止的候選位姿，見 §13）。

### `optimization.py` — Wasserstein 多視角優化

**刻意不用 g2o**（g2opy 在 py3.9 難維護）：問題規模極小
（每物件 9 參數：center 3 + log 半軸 3 + rotvec 3；殘差 ≤10 個觀測 × 5），
`scipy.optimize.least_squares`（TRF、數值 Jacobian）<10ms 收斂。

殘差（每觀測 5 維）：投影橢圓 vs 觀測橢圓的 2nd-order Wasserstein 代理：

```
[Δμx, Δμy, vech(sqrtm(Σ_proj) − sqrtm(Σ_obs))]
```

觸發：track 累積 ≥3 觀測，之後每 +3 觀測重跑；只用**最近 10 個**觀測
（無上限時曾出現 4.6s 尖峰）。log 半軸參數化天然保正，再 clip。

### `linking.py` — 大型物件連結
同類別、中心距 <1m 的 tracks 以 union-find 連通（L 型沙發等單一橢球
無法表示的物件）；`object_center()` 回傳連通元件中心平均，
供導航與 scene graph 使用。

### `object_layer.py` — 協調器
`update(frame, dets)`：關聯 → 新 track 初始化 → 到期優化 → 重連結。
`candidates(target, min_obs, min_score, min_bbox_px)`：**品質閘門**
（分數 ≥0.45、bbox ≥3000px²、觀測 ≥3）——碎片偵測（如桌下椅子的
一條邊）不得觸發昂貴的接近+驗證流程。`blacklist(id)` 供驗證拒絕使用。

## 7. mapping

### `costmap.py` — 佔據地圖

`grid` int8：`-1 unknown / 0 free / 100 occupied`；0.05m 解析度；
座標超界時自動倍增擴張（內容置中）。

**`update(frame, floor_y, ...)`**：
1. 反投影深度（stride 4、5m 上限）。
2. 依高度分類：floor band（`floor_y ± [-0.3, 0.1)`）與 obstacle band
   （`[0.1, 1.5)`；下限 0.1 是實測值 —— 0.2 時床架/桌腳不可見，
   agent 對隱形家具空推）。
3. **向量化批次 raycast**（`_raycast_batch`）—— 取代逐點 Python
   Bresenham（1620ms → 38ms/幀）：
   - 端點按格去重（數千點 → 數百條射線）；
   - 所有射線以 `t ∈ linspace(0,1,M)`、M = 2×最長射線格數 取樣成
     `(M, N, 2)` 整數格陣（2 倍超取樣補對角縫隙）；
   - 讀取格值 → 每射線 `argmax(occupied)` 找首個牆格 → `idx < 首牆`
     的取樣格標 FREE（不穿牆）；
   - 未被擋的 obstacle 端點最後蓋 OCCUPIED（順序保證牆不被 free 覆寫）。

### `frontier.py`
free 且 4-鄰接 unknown 的格 → 8-連通元件（`scipy.ndimage.label`）→
濾除 <8 格 → 質心去重（<1m 取大者）。
⚠️ **frontier id 每次抽取重新編號**——任何跨時刻的 frontier 記憶
（如 blacklist）都必須以空間位置為鍵（見 §13）。

### `room_seg.py` — 房間分割
1. free mask 距離變換 → `peak_local_max`（間距與門檻 = 0.9m）取種子；
2. `watershed(-dist, seeds, mask=free)`；
3. **開放邊界合併**：兩區共享邊界的最大 clearance > 門寬/2（0.6m）
   表示中間沒有牆縮口 → union-find 合併（開放空間不過度分割）；
4. 濾除 <60 格的碎片；
5. **id 穩定化**：新標籤與上一次結果按格重疊 >30% 對應舊 id，
   否則發新 id —— 房間節點跨重建保持身分（LLM 房型標註快取依賴此）。

## 8. graph

### `scene_graph.py`
純 dataclass（不用 networkx；圖小、查詢客製）。
- `RoomNode(id, label, centroid_xy, n_cells)`：label 由 LLM 標註後快取。
- `ObjectNodeView`：**object 狀態的唯讀視圖**（真身在 ObjectLayer）。
- `rebuild(room_labels, costmap, object_layer)`：物件按中心格的房間 id
  歸屬；不在任何房間內時取 3m 內最近房間質心；保留舊房名。
- 查詢：`objects_near(xy, r)`、`objects_in_room(id)`、
  `room_of_point(xy)`、`unlabeled_rooms()`。

### `serialize.py`
- `to_prompt_text(sg)` → 給 LLM 的緊湊層級文字：

  ```
  Room 1 (bedroom): bed x1 (near lamp), lamp x2 (near bed)
  Room 2: sink x1
  Hallway/other: towel x1
  ```

  object-object "near" 邊（<1.5m）在序列化時即時推導，不持久化。
- `to_json(sg)` → 記錄/未來 ScanNet R@d 評估（M7 介面）。

## 9. llm

### `client.py` — ChatClient
`openai` 套件指向任何 OpenAI-compatible 端點（預設
`$OLLAMA_HOST/v1`；換雲端只改 `configs/llm/openai.yaml`）。
- 影像以 base64 jpeg content-part 傳遞，`max_image_px` 縮圖上限。
- `response_format json_object` + `extract_json()`（正則撈第一個
  `{...}`）+ 解析失敗自動追問一次。

### `prompts.py`
- `ROOM_LABEL_*`：物件列表 → 固定房型詞彙之一。
- `FRONTIER_SCORE_*`：目標 + scene graph 文字 + 每 frontier 局部描述
  （+ 影像註記）→ `{"scores": {id: p}}`。
- `VERIFY_*`：**describe-then-decide** —— 要求模型先描述再判斷
  （直接 yes/no 在邊界影像上會翻面；描述先行可錨定決策，
  `prompt_lab.py` 實測結論）。

## 10. exploration

### 評分階梯（`scorer.py` → `llm_scorer.py` → `vlm_scorer.py`）
統一介面 `score(frontiers, sg, target, keyframes) -> {fid: P}`：

| scorer | 行為 | 用途 |
|---|---|---|
| `RandomScorer` | 均勻隨機 | ablation 下限 |
| `NearestScorer` | P≡1（選擇退化為 1/d） | 論文 "Without LLM" |
| `LLMTextScorer` | 房型標註（快取）+ 全 scene graph 文字 + 每 frontier 3m 子圖描述，單次呼叫評 ≤8 個 | 論文 baseline |
| `VLMScorer` | 繼承上者，另附每 frontier 1 張「最近且面向」keyframe（≤4 frontier/次控 token） | 改進 B |

### `async_scorer.py` — 非同步包裝
單 worker `ThreadPoolExecutor`；in-flight 時丟棄新請求；
`latest()` 回傳最近完成的分數（容忍 stale）；例外被吞並記錄
`last_error`（進 episodes.jsonl）+ **30 秒錯誤退避**
（曾有 454 次連續失敗呼叫刷爆一個 episode）。

### `selector.py`
- `frontier_goal_xy(f, costmap)`：規劃目標取「離質心最近的 frontier
  格」而非質心本身（凹形元件的質心可能落在牆裡）。
- `select_frontier(...)`：候選按 P 排序取 top-5 → 逐一 A* 算真實路徑
  成本 d → `argmax P/d`。未評分者用先驗 P=0.3。規劃失敗的候選寫入
  `failed_out` 供呼叫端**只封鎖失敗者**
  （⚠️ 曾因一輪失敗封鎖全部 frontier 而癱瘓探索）。

## 11. planning

### `planner.py` — AStarPlanner
8-連通 A*，成本 = 步長 × 乘數：

| 格況 | 乘數 |
|---|---|
| free | 1 |
| unknown | 3（可通行 —— frontier 目標本來就在未知邊界） |
| 膨脹帶（0.25m 內非 occupied） | 8（**軟性成本**） |
| occupied | 不可通行 |

**軟性膨脹是 P0 的核心修正**：硬阻擋會把部分觀測地圖的細通道封死、
把 agent 所在的小口袋與世界斷開（A* `no_path searched 300`）。
軟成本保證連通性，路徑仍偏好遠離障礙。
其他：起點在 occupied（幻影障礙）時 `_nudge_free` 在 1m 窗內找最近
可通行格；60k 展開上限（防 12s 尖峰）；`last_failure` 字串供診斷；
回報的 `cost` 是幾何長度（懲罰只引導搜尋，不進 P/d 的 d）。

### `controller.py` — WaypointController
- 追蹤路徑上 0.5m 前瞻 waypoint；heading 誤差 >15° 轉向
  （**err>0 → turn_right**，見 §2），否則 forward。
- **卡住偵測**：forward 後位移 <½ 步長，連續 3 次 → 在面前標記
  **5×5 圓盤** OCCUPIED（單格會被 8-連通斜穿繞過）+ `stuck` 旗標
  → agent 清路徑重規劃。

## 12. verification

### `viewpoint.py` — ViewpointPlanner
物件周圍環取樣（半徑 0.8/1.2/1.5/2.0m × 16 方位）：候選須為 FREE 格、
通過 Bresenham LOS（忽略物件自身 0.3m 內的格）；分數 = clearance −
0.3·|r−1.2|；由內圈往外找，可用 `exclude` 排除已試過的位姿。
⚠️ 已知限界：2D LOS 看不出桌面高度的遮擋（見 §13 的偵測器複查）。

### `verifier.py` — TargetVerifier（改進 C）
驗證語意：「**這個偵測是真的嗎**」→ 證據是偵測器自己的 best crop
（視點 live view 常因 3D 遮擋看不到物件，會誤導小模型拒絕真目標）。

實測演化出的完整配方（每項都對應一輪除錯）：
1. **模型 = qwen2.5vl:7b**（`verification.vlm_model`）：3B 對清晰的
   正例也會拒絕（prompt_lab 三張參考圖無一問法全對），7B 全對。
2. **小圖放大**：crop 最長邊 <320px 時 cubic 放大（156px 椅子
   拒→接受的分水嶺）。
3. **describe-then-decide prompt**（見 §9）。
4. **拒絕時重問一次**（temp 0 → 0.3，任一接受即接受）：邊界影像
   跨執行翻面；誤拒會 blacklist 真目標、通常斷送整個 episode，
   而品質閘門已擋掉垃圾候選，誤收代價低。
5. 失敗開放（VLM 掛掉時回 True = 論文無驗證行為）；`confidence`
   缺漏時預設 0.6；`debug_dir` 存證據影像 + 回應
   （`outputs/<run>/verify_debug/`）。

## 13. agent — NavAgent 狀態機

```
INIT(360°掃描, 12×turn) ─> EXPLORE ⇄ GOTO_FRONTIER
                              │ 候選通過品質閘門
                              ▼
                    GOTO_VERIFY_VIEW ─> VERIFYING ─accept→ APPROACH → DONE(stop)
                              ↑            │reject: blacklist track → EXPLORE
                              └────(偵測器看不到目標→回 best-cam 位姿)
```

每步（`act(frame)`）固定執行：costmap 更新、卡住觀察、
keyframe 時偵測+物件層+scene graph 更新、候選檢查（僅
INIT/EXPLORE/GOTO_FRONTIER 狀態）。

**各狀態邏輯與防呆**（每一條都是實際踩過的坑）：

- **EXPLORE**：每 5 步至多選一次 frontier（抽取+top-5 A* 昂貴；
  等待時原地轉、地圖照樣長）。選不到就轉身。
- **GOTO_FRONTIER**：
  - *放棄網*：15 步位移 <0.2m → 空間 blacklist 該 frontier 100 步
    →重選（隱形障礙/模擬碰撞的保險）。
  - frontier blacklist 以 **位置 (0.6m 半徑)** 為鍵（id 不穩定）。
- **GOTO_VERIFY_VIEW**：終端接近語意 —— 0.35m 內、路徑耗盡、或 80 步
  deadline 任一到達即進 VERIFYING（離散動作幾乎不會精確落點，
  曾繞視點 486 步）。
- **VERIFYING**：先轉身面向物件（誤差 ≤20°）；若偵測器**看不到**
  目標（`_target_visible`：當前畫面跑 YOLOE 找同類別 >0.25）且未
  試過 → 走回 `best_cam_xy`（該 track 最佳偵測的相機位姿 ——
  實證可達且可見）再驗證。之後交給 TargetVerifier。
  拒絕 → blacklist track → EXPLORE。
- **APPROACH**（P1a 終端策略）：驗證通過（或 verifier off 時直接）
  進入，`_do_approach` 逐步邏輯：
  1. 用偵測器複查目前這一步是否看得見目標（`_best_target_detection`）；
  2. **可見**：記錄此位姿為 `_approach_last_good_xy`；bbox ≥
     `agent.approach_stop_bbox_px`（預設 40000px²）→ **停**（夠近夠
     清楚 = 已在 viewpoint 集合內部）；否則往物件方向再走一步；
  3. **不可見**：若曾有 last_good_xy 且離現在 >0.1m → **退回**該處停止
     （這一步跨過了 2D LOS 看不到的 3D 遮擋邊界，如桌緣）；若從未
     可見 → 沒有更好的退路，繼續往物件前進直到 deadline；
  4. 步數上限 `agent.approach_max_steps`（預設 12）或 100 步全域
     deadline 到 → 停在當下。
  - 為何不是固定距離規則：HM3D 的 dtg 量到 **view_points**
    （可見性定義的位姿集合，每物件 100–500 個)的測地距離，貼著物件
    反而衝出集合——三個舊策略（追目標格/抵達判定/接觸式 nudge）都
    卡在 dtg 0.107–0.147m。bbox-可見性驅動的停止條件直接命中這個
    定義：實測 bed 從 dtg 0.107（成功, SPL 0.295）改善到
    **dtg 0.015**（成功, SPL 0.345），首次通過嚴格 0.1m 門檻。
  - `_follow_to(frame, goal_xy)`：與 `_follow_path` 平行的路徑跟隨
    輔助，差別是**顯式接受 goal 參數**並在 goal 改變時重規劃——
    APPROACH 在「前進」與「退回」兩個目標間切換，不像其他終端狀態
    整段只有一個固定目標。
- 儀表：`stats`（plan_ok/fail、select_ok/none、frontier_give_up）與
  `state_log`（狀態轉換序列）隨 episode 落盤。

## 14. sim

### `habitat_env.py`
- `make_objectnav_config()`：載入 habitat-lab 的
  `benchmark/nav/objectnav/objectnav_hm3d.yaml` 後覆蓋感測器解析度/
  HFOV（**必須 int**）、步長/轉角、success_distance、資料路徑。
  ⚠️ 進入前先 `GlobalHydra.instance().clear()`——我們自己的
  `@hydra.main` 佔著全域 Hydra，habitat 需要重新初始化。
- `_to_frame()`：從 `sensor_states["rgb"]` 取位姿（四元數→R），
  GL→CV 轉換（§2），depth squeeze 成 (H,W)。
- 動作映射 `{stop:0, move_forward:1, turn_left:2, turn_right:3, ...}`。

## 15. eval

### `runner.py`
組裝（scorer/detector/verifier 由 config 決定）→ 逐 episode 迴圈 →
每 episode 落一行 `episodes.jsonl`（success/spl/dtg/steps/fps/
llm 與 verify 統計/`agent_stats`/`state_log`/`final_xy`）→
彙總 `summary.json`（含 config 指紋與 dataset version）、
`timing.csv`、`viz/ep*.png` 軌跡圖。
啟動時對 ollama 送 `keep_alive=0` 卸載模型 —— 讓 YOLOE 的一次性
文字編碼獨占 GPU（否則駐留的 VLM 造成 OOM）。

### `metrics.py` / `visualize.py`
SR/SPL/dtg/steps 聚合（另有 per-category）；top-down 圖：
costmap 灰階 + 軌跡（橘）+ 物件標記（綠三角+標籤）+ 起點/目標。

## 16. Docker 部署與 VRAM 預算

### image（`docker/Dockerfile`）
`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` + EGL userland
（`libegl1 libglvnd0` + **`libopengl0 libglx0`**，缺後兩者 habitat-sim
import 即倒）→ miniforge py3.9 → `habitat-sim=0.3.1 headless withbullet`
（aihabitat channel）→ habitat-lab v0.3.1 → torch 2.4.1 cu121 →
ultralytics 8.3 + `numpy<2` + CLIP tokenizer（預裝，否則 AutoUpdate
每個容器重裝一次）。
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
是 headless EGL 的關鍵。原始碼**不 COPY**：bind-mount +
`PYTHONPATH=/workspace/src`（不 pip install → 容器能以非 root 跑）。

### compose（`compose.yaml`）
- `ollama`：**GPU 推理必須**（CPU 含圖呼叫實測 204s/次不可用）；
  `OLLAMA_CONTEXT_LENGTH=8192`、`MAX_LOADED_MODELS=2`（3b 評分 +
  7b 驗證共存）。VRAM 不足時 ollama 自動 partial offload（實測 3b
  為 29%/71% CPU/GPU），不會弄倒 nav。
- `nav`：`user: "1000:1000"`（host 檔案所有權）、`HOME=/tmp`、
  `YOLO_CONFIG_DIR=/tmp/Ultralytics`、
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  ⚠️ `docker exec` 進容器**繞過 entrypoint**（拿到 conda base 的
  py3.13）；請用 `docker compose run --rm nav ...` 或在容器內
  `conda run -n habitat python ...`。

### 6GB VRAM 收支（RTX 4050 Laptop 實測）
| 佔用 | 大小 |
|---|---|
| habitat EGL + torch context | ~1.4 GB |
| YOLOE-11s fp16 推理 | ~0.7 GB |
| mobileclip 文字編碼（一次性，用畢釋放） | 0.6 GB 峰值 |
| qwen2.5vl:3b（ollama, partial offload） | ~3.9 GB |
| qwen2.5vl:7b（驗證時換入） | 依 offload 比例 |

## 17. Config 參考

| group | 檔案 | 關鍵欄位（預設） |
|---|---|---|
| agent | default | max_steps 500 / forward 0.25 / turn 30 / success_distance 0.1（paper mode 0.13）/ initial_scan / camera_height 0.88 / approach_stop_bbox_px 40000 / approach_max_steps 12 |
| detector | yoloe_small（6GB）/ yoloe（12GB） | weights / conf 0.3 / imgsz 512↔640 / half |
| scene_graph | default | keyframe 0.25m/30° / refine ≥3 每 3 / link 1.0m / assoc 0.4 / depth gate 0.5m / room_seg 每 10 kf |
| exploration | vlm / llm_text / nearest / random | top_n 5 / dedup 1m / min_cells 8 / subgraph 3m / 1 img/frontier / ≤4 frontier/call / prior 0.3 |
| llm | ollama / openai | base_url / text_model qwen2.5vl:3b / vlm_model / timeout 120s / max_image_px 512 |
| verification | on / off | min_obs 3 / min_score 0.45 / min_bbox_px 3000 / ring 0.8–2.0 / accept_conf 0.5 / **vlm_model qwen2.5vl:7b** |
| mapping | （內嵌） | res 0.05 / obstacle 0.1–1.5m / range 5m / stride 4 / inflate margin 0.07 |
| eval | hm3d_val / hm3d_val_mini | split / num_episodes / dataset_version v2 / save_viz |
| ablation | full / no_verify / paper_baseline / no_llm | defaults-list 覆蓋組合 |

執行入口與輸出格式見 [README.md](../README.md)；
資料下載見 [data/README.md](../data/README.md)。
