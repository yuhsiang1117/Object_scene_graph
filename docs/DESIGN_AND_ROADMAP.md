# Object Scene Graph ObjectNav — 設計文件與改進路線圖

> 深入的模組實作細節見 [ARCHITECTURE.md](ARCHITECTURE.md)；
> 測試策略與詳表見 [TESTING.md](TESTING.md)。本文件聚焦狀態與路線。

> 更新日期：2026-07-16。本文件記錄系統目前的設計、已驗證狀態、eval 除錯歷程、
> 已知問題，以及接下來的改進與測試規劃。

## 1. 專案目標與定位

以 `docs/RA_L__2025_.pdf`（RA-L 2025 投稿）為規格書，從零重建
**real-time object-based hierarchical 3D scene graph (3DSG) + LLM-guided
frontier exploration** 的 ObjectNav 系統，並整合三項 2025–2026 文獻回顧後
選定的現代化改進：

| # | 改進 | 動機（對應文獻） |
|---|---|---|
| A | **YOLOE 取代 YOLO-World**，偵測+分割一體、去除 SAM | YOLOE (ICCV 2025)；MSGNav 仍用 YOLO-World+SAM 而無法即時 |
| B | **VLM 多模態 frontier 評分**（keyframe 影像進 prompt） | MSGNav (2025)：純文字 scene graph 造成不可逆視覺資訊損失 |
| C | **Target verification / last-mile**（視點規劃 + VLM 驗證 + blacklist） | false positive 是 open-vocab ObjectNav 主要失敗源；CogNav 的驗證狀態 |

**論文敘事定位**（2025–2026 SOTA 比較後的結論）：不與 GPT-4o 雲端系統拚
SR，主打 **real-time、edge-deployable（6–12GB 消費級 GPU）、小型自架
LLM 可用**。評估對齊 HM3D-semantics **v0.2 / 6 類別 / ObjectNav v2
episodes**（近期 baseline 主流報告版本）。

## 2. 系統架構

### 2.1 資料流

```
Habitat sim ──> FrameData(rgb, depth, T_wc[OpenCV 慣例], intrinsics)   ← 唯一資料貨幣
    │                                                     （未來 ScanNet/真機只需換 loader）
    ├─ 每幀:    CostmapBuilder（向量化 raycast, 2D occupancy, 自動擴張）
    └─ keyframe（位移>0.25m 或旋轉>30°）:
         YoloeDetector（open-vocab det+seg, GPU, fp16）
           └→ ObjectLayer: 關聯（max(I/A1,I/A2) bbox近似 + 深度一致性閘）
                → 橢球初始化（mask矩 + 深度取樣反投影）
                → Wasserstein 多視角優化（scipy least_squares, 9參數, 最近10筆觀測）
                → 同類別 union-find 連結（L 型物件）
         VoronoiRoomSegmenter（距離變換+watershed, 每10 keyframe, 房間id穩定化）
         SceneGraph rebuild（building→room→object; near 邊在序列化時推導）
```

### 2.2 探索與導航（NavAgent FSM）

```
INIT(360°掃描) → EXPLORE ⇄ GOTO_FRONTIER
                    │ 發現候選目標
                    ▼
        GOTO_VERIFY_VIEW → VERIFYING ──accept──> APPROACH → STOP
                              │reject: blacklist → EXPLORE
```

- **Frontier**：free 鄰 unknown → 連通元件 → 去重；每 5 步節流一次選擇。
- **評分階梯**（config 切換，即 ablation）：`random` / `nearest` /
  `llm_text`（論文 baseline）/ `vlm`（改進 B：每 frontier 附 1 張最近且
  朝向它的 keyframe，≤4 frontiers/呼叫）。
- **AsyncScorer**：單 worker、in-flight 丟棄、錯誤退避 30s——**LLM 永不
  阻塞控制迴圈**（real-time 主張的關鍵；控制迴圈 FPS 與 LLM 延遲分開報告）。
- **選擇**：argmax Pᵢ/dᵢ，dᵢ 為 A* 真實路徑成本（top-5），未評分 frontier
  先驗 P=0.3。
- **驗證（改進 C）**：物件周圍 0.8–2.0m 環取樣視點 + Bresenham LOS 檢查 →
  走到視點 → 轉向面對物件 → VLM 以 best crop（主）+ live view（輔）判斷
  `{is_target, confidence}` → 拒絕即 blacklist。
- **終端接近（APPROACH，P1a）**：驗證通過後朝物件前進，每步用偵測器
  複查——可見且 bbox 夠大即停、可見但還小則續走一步、看不見則退回最後
  可見位姿停止。直接命中 HM3D 的 view_points 可見性定義，見 §6 P1a。

### 2.3 模組地圖（`src/osg/`）

| 套件 | 檔案 | 核心類別/職責 |
|---|---|---|
| core | types / geometry / profiler / config | FrameData、投影幾何、每模組計時、Hydra dataclass |
| perception | detector / keyframe | YoloeDetector（fp32 載入→fp16 推理；詞彙表排序正規化，per-episode no-op；mobileclip 用後即從 VRAM 釋放）、KeyframeStore |
| objects | ellipsoid / association / optimization / linking / object_layer | dual quadric 投影（退化檢查）、貪婪關聯、W2 優化、連結 |
| mapping | costmap / frontier / room_seg | 向量化批次 raycast（1620ms→38ms）、frontier 抽取、watershed 房間 |
| graph | scene_graph / serialize | 階層節點、`to_prompt_text()`（LLM 用）、`to_json()`（ScanNet eval 預留） |
| exploration | scorer / llm_scorer / vlm_scorer / async_scorer / selector | 評分階梯、async 包裝、P/d 選擇 |
| planning | planner / controller | A*（unknown 罰 3 倍可通行、60k 展開上限）、waypoint 追蹤 + 卡住偵測 |
| verification | viewpoint / verifier | 視點環取樣 + LOS、VLM 驗證 |
| agent | nav_agent | FSM、儀表化（stats + state_log） |
| sim / eval | habitat_env / runner / metrics / visualize | OpenGL→OpenCV 轉換、SR/SPL + timing.csv + 軌跡圖 |

### 2.4 Docker 環境

- `docker/Dockerfile`：CUDA 12.1 + miniforge py3.9 + habitat-sim 0.3.1
  headless + habitat-lab v0.3.1 + torch 2.4.1 cu121 + ultralytics 8.3 +
  CLIP tokenizer（預裝避免 AutoUpdate）。headless EGL 需
  `libopengl0 libglx0` 與 glvnd vendor json。
- `compose.yaml`：`ollama`（GPU、ctx 8192、KEEP_ALIVE 30m）+ `nav`
  （uid 1000 執行、PYTHONPATH 取代 pip install、原始碼 bind-mount）。
- **6GB VRAM 預算**（RTX 4050 Laptop 實測）：habitat+torch ~1.4GB、
  YOLOE-11s ~0.7GB、qwen2.5vl:3b 於 ollama ~3.9GB（29%/71% CPU/GPU
  partial offload）。eval 啟動時先 `keep_alive=0` 卸載 ollama，讓 YOLOE
  文字編碼独占 GPU 完成後再讓 ollama 回來。**CPU VLM 實測 204s/次含圖
  呼叫，不可用**；GPU 後約數秒。
- 12GB profile：`detector=yoloe` (11l/640px) + `qwen2.5vl:7b`，config 一鍵切換。

## 3. 目前驗證狀態（2026-07-15）

| 項目 | 狀態 |
|---|---|
| Docker build + headless EGL 渲染（真 HM3D 場景） | ✅ |
| 單元測試（55 個，合成資料免 GPU，含 nav_agent APPROACH 8 個） | ✅ <8s 全過 |
| Ollama 文字+視覺往返 | ✅ |
| HM3D minival（10 場景+semantic configs）+ ObjectNav v2 episodes | ✅ 已下載 |
| Sim 整合測試（stub detector 全 pipeline 1 episode） | ✅ |
| 端到端 eval（YOLOE+VLM 3 episodes 跑完、產出 summary/timing/viz） | ✅ |
| VLM frontier 評分實際運作 | ✅（GPU 後 0 錯誤） |
| SR > 0 | ✅ eval25：bed success=1 / SPL 0.295（paper mode 0.13m） |
| **嚴格 0.1m 門檻下的 SR** | ✅ P1a：bed dtg 0.015m（見 §6 P1a），首次通過嚴格門檻；chair/toilet 仍待 §6 P1b/P1c |
| **30-episode 全量（val_mini 全部）** | ✅ 已跑，SR=6.67%（2/30）、SPL=0.0205（見 §6 P1d）——**遠低於** 3-episode pilot 的 33.3%，證實先前反覆用同 3 個 episode 調參已隱性 overfit |

## 4. Eval 迭代記錄（除錯史與教訓）

| 輪 | 修正 | 結果/教訓 |
|---|---|---|
| 1–6 | Hydra GlobalHydra 衝突、YOLOE fp16 文字頭 dtype、CLIP AutoUpdate、outputs 權限、VRAM OOM | 環境層問題全清；教訓：**ultralytics checkpoint 是 fp16**、**掛載點擋 symlink**、**容器 root 檔案汙染 host** |
| 7 | 首次跑通 | SR 0/3、0.36 FPS；costmap 1.6s/幀（Python raycast）；VLM 100% context 爆掉；verify 92s/次 |
| 9 | costmap 向量化、影像減量 | costmap **38ms** ✅；VLM 改超時（CPU 推理 200s+ 不可行） |
| 10 | ollama 回 GPU、GOTO 不繞圈 | VLM **0 錯誤** ✅ FPS 2.2；新問題：verifier 過嚴，2/2 拒絕真目標 |
| 11–12 | verifier 面向物件+live view+寬容 prompt；啟動時卸載 ollama | 揭露**當前主問題**：agent 幾乎不移動（見 §5） |
| 13 | 加入 stats/state_log 儀表化 | 單 episode 診斷 run 被中止（使用者要求停止），**儀表化程式碼已就位** |

## 5. P0 已解決（2026-07-15）：movement deadlock 的四層根因

**首個 success 達成**：eval25（paper mode success_distance=0.13）
ep11 bed success=1 / SPL 0.295。探索→偵測→驗證→接近→停止全鏈貫通。

P0 的「agent 不移動」實際上是四個疊加 bug（`scripts/diag_movement.py`
以 stub detector + nearest scorer 隔離確診）：

1. **硬性膨脹斷開連通性**：agent 走到牆邊後，0.25m 膨脹把它所在的小
   口袋與地圖隔離（A* `no_path searched 300`）→ 膨脹改為軟性成本
   （inflate_penalty 8x），只有 occupied 不可通行。
2. **連坐封鎖**：一輪選擇失敗就把全部 frontier blacklist 50 步 → 只封
   鎖實際規劃失敗的候選。
3. **frontier id 不穩定**：每次抽取重新編號，id-keyed blacklist 無效
   → 改空間位置封鎖（0.6m 半徑）。
4. **看不見的低矮障礙**：agent 對著障礙帶以下的家具空推 → stuck 標記
   改 5x5 圓盤（單格會被 8-連通繞過）+ GOTO_FRONTIER 15 步無進展放棄網。

後續 debug 鏈（每輪一個 bug，均已修復並 commit）：GOTO_VERIFY_VIEW /
GOTO_TARGET 繞圈（加抵達判定+deadline）、候選品質閘門（碎片偵測觸發
80 步白跑）、3B 驗證器誤拒真目標（升 7B + 小圖放大 + describe-then-
decide + 拒絕時重問一次）、停止位姿（best-cam 回歸：回到最佳偵測的
相機位置——被證明可達且看得見物件）。

**診斷方法論**（比結論更值錢）：`prompt_lab.py` 用存檔影像離線迭代
VLM prompt；`verify_debug/` 存驗證證據影像；`state_log`+`agent_stats`
進 episodes.jsonl；`diag_movement.py` 隔離導航棧。

## 6. 改進規劃（優先順序）

### P1a — 「approach while visible」終端策略（已完成，2026-07-15）

**做了什麼**：終端接近從「走到固定距離/ring 就停」換成
`State.APPROACH`——驗證通過後朝物件前進，**每步用偵測器複查**：
- 可見且 bbox ≥ `agent.approach_stop_bbox_px`（預設 40000px²）→ 停
  （夠近、夠清楚，等同已站進 viewpoint 集合內部）；
- 可見但 bbox 還小 → 記錄此位姿為 `_approach_last_good_xy`、再前進一步；
- 看不見 → 若曾有 last_good_xy，**退回**該處停止（一步之遙的 3D 遮擋，
  2D costmap LOS 抓不到）；否則（從未看過）繼續朝物件前進到 deadline。

三個舊策略（追目標格、抵達判定、接觸式 nudge）全部卡在 dtg
0.107–0.147m 的成功圈邊緣——因為 HM3D 的 dtg 是量到 **view_points 集合**
（可見性定義的位姿）的測地距離，貼著物件反而衝出集合。bbox-可見性驅動
的停止條件直接命中這個定義。

**驗證數字**（eval-mini，paper mode 0.13m，與改動前同三個 episode 對照）：

| episode | 改動前 dtg | 改動後 dtg | 備註 |
|---|---|---|---|
| bed (ep11) | 0.107（success, SPL 0.295） | **0.015**（success, SPL 0.345） | **嚴格 0.1m 門檻下也會過**——本專案首次 |
| chair (ep4) | 0.121–0.147（一直失敗） | 0.153（失敗） | 噪音範圍內，兩種策略下都從未成功過 |
| toilet (ep7) | 500 步走不到（失敗） | 500 步走不到（失敗） | 不受影響，符合預期（探索效率問題，見 P1b） |

整體 SR 持平 1/3、**SPL 0.098→0.115**。淨效益明確。

實作：`src/osg/agent/nav_agent.py`（`State.APPROACH` 取代
`State.GOTO_TARGET`；`_do_approach`/`_follow_to`/`_best_target_detection`；
移除死碼 `_final_nudge_or_stop`/`_arrived_at_goal`/`_tried_viewpoints`）、
`core/config.py`（`approach_stop_bbox_px`/`approach_max_steps`）。
新增 [tests/unit/test_nav_agent.py](../tests/unit/test_nav_agent.py)
（8 個測試，直接呼叫 `_do_approach` 覆蓋四個分支：bbox 達標停止、可見
續走、視野遺失退回、從未可見時退避到前進、step/deadline/不可達邊界）
——這是專案第一份 `nav_agent.py` 單元測試，補上 TESTING.md 點名的空缺。

### P1b — 探索效率（ep7 型失敗，已解決導航部分，2026-07-15）

**診斷**：toilet episode（`agent_stats.frontier_give_up=11 / select_ok=12`）
用 `giveup_log`（新增，記錄每次放棄的 frontier 座標+agent 位置）抓到
根因——三個不同 frontier 的追逐全部死在**同一個座標**，45 步內完全沒有
移動。`WaypointController` 卡住偵測只標記前方一個 0.1m 半徑的小圓盤，
A* 仍能規劃「擦邊繞過」這個小標記、實際上還是穿過同一個門檻寬度瓶頸
的路徑，agent 換方向重試多次都撞在同一個物理瓶頸上。

同時查 habitat ground truth 發現：該 episode 3 個 toilet 實例中 2 個
明顯在不同樓層（y≈2.8–3.0 vs agent 起始 y≈0），是專案排除範圍內的
multi-floor 限制；只有 1 個同樓層實例（goal#2）理論可達。

**修正**：
1. `_select_new_frontier` 選到新 frontier 時沒有重置 give-up 計時器的
   參考點——真實 bug，已修，但**不是這裡的主因**（修正後 give-up 次數
   不變，`giveup_log` 顯示大部分放棄都發生在合理追逐之後、非選擇當下
   誤判）。
2. `WaypointController` 卡住標記從「前方一點、半徑 0.1m」擴大為「沿
   前進方向 3 個距離點（0.5×/1×/1.8× forward_m）、半徑 0.35m」，形成
   真正擋住門檻寬度瓶頸的一段障礙帶。

**結果**（同一 episode 重跑對照）：

| 指標 | 修正前 | 修正後 |
|---|---|---|
| `frontier_give_up` | 11 | **2** |
| `distance_to_goal` | 2.58 / 9.67（兩次不同噪音） | **0.24** |
| `final_xy` 位置 | 遠離同樓層 toilet | 幾乎貼上同樓層 toilet（goal#2）視點集合邊緣 |

探索/卡死問題基本解決。**新瓶頸**：`verify_calls: 0`——即使幾何上已
極度接近（dtg 0.24m），偵測器全程未曾產生一個通過品質閘門
（score≥0.45、bbox≥3000px²、obs≥3）的「toilet」候選。同時
`select_none: 30`（遠高於 `select_ok: 4`）顯示後段大量時間找不到可選
frontier，值得一併檢查是否為房間已探索完但目標視角一直沒對上。

**下一步**（P1b-2，待辦）：
- 檢查該 episode 是否有留存的 toilet 偵測但分數/bbox 不足以通過閘門
  （可能是浴室小、易遮擋物件的通性問題，非此 episode 特例）。
- 若確認是普遍問題，考慮依物件類別調整品質閘門（toilet/bathtub 等小型
  固定物件 vs chair/bed 等大型家具，用同一閾值可能不公平）。
- 也一併排查 `select_none=30` 高企的原因（frontier 真的枯竭，還是
  blacklist/give-up 累積過度保守）。

### P1c — 剩餘掃參與測試集

**驗證器離線評測集（已完成，2026-07-15）**：從整個 session 累積的
`outputs/*/verify_debug/` 中挑出 24 張真實影像（非合成），涵蓋全部 6
個 HM3D 類別，人工獨立標註 ground truth（不參考當時 VLM 自己的判斷），
存進版控的 [tests/fixtures/verify_bench/](../tests/fixtures/verify_bench/)
（`images/` + `labels.json`）。`scripts/verify_bench.py` 可離線重跑
（免 eval，~5 分鐘 vs 20–60 分鐘），`tests/integration/test_verify_bench.py`
（`@pytest.mark.gpu`）當回歸閘門。

**實測結果**（qwen2.5vl:7b，20 個非模糊案例）：
**accuracy=0.700、precision=0.846、recall=0.733**（TP=11 FP=2 FN=4 TN=3）
——比先前用 3 張精選參考圖做 prompt_lab 測試時的印象差很多，證明「先
測小樣本就下結論」的風險，也印證了先前「拒絕重問一次」設計決策的
必要性。具體發現：
- 同一張椅子的 4 個近乎相同裁切，2 張接受、2 張拒絕——**取樣噪音在
  「清楚的真陽性」上依然造成約 50% 不穩定**，不是邊界案例才會發生。
- 一張「紅白條紋柱子+粉色緞帶」的垃圾裁切被誤判為椅子（假陽性）——
  硬負例沒有想像中容易擋掉。
- 藍色馬桶（不尋常顏色、近距離裁切缺乏整體脈絡）被誤拒——假陰性。
- 沙發背面裁切被誤判成「木頭結構」而拒絕——VLM 描述有時系統性看錯
  材質紋理。
- 值得慶幸：4 張同一張「被誤標成 bed 的沙發」中，3 張被正確拒絕，只有
  1 張因取樣誤判被接受——確認 P1b 除錯時發現的「拒絕重問」雙面性
  （見上）不是常態，多數時候機制運作正常。

**待辦**：
- ~~`approach_stop_bbox_px` 依物件類別重新校準~~ ——**已用真實數據檢驗，
  結論是不該做**，見 P1e。

### P1d — 30-episode 全量結果（2026-07-15）：發現隱性 overfitting + 卡住標記迴歸

**動機**：3-episode pilot（chair/bed/toilet）在 P0–P1c 反覆被拿來調參
超過 25 輪，數字（SR 33.3%）很可能是對這 3 個特定 episode 過擬合，
不能代表真實泛化能力。擴大到 val_mini 全部 30 個 episode（不需新下載，
涵蓋 6 類：chair/bed/sofa/plant/tv_monitor/toilet）驗證。

**過程中發現的迴歸 bug（已修正）**：擴大樣本後第一輪跑到 7/30 時，
`ep10`（chair）的軌跡圖顯示 agent 在一個約 3m 的房間裡困了整整 500 步、
從未離開（`select_none=66/69`）。根因：`WaypointController` 的卡住
標記直接寫入 `costmap.grid` 且**沒有到期機制**，而一旦標記 OCCUPIED
就會擋住未來的 raycast、永遠無法被新的深度觀測自然消除——自我強化
的鎖死。P1b 把標記範圍從 0.1m 擴大到 0.35m×3 點後，這個鎖死效應被
放大到足以永久封死整個門。修正：每個標記帶上到期步數（60 步），過期
釋放回 UNKNOWN（不是 FREE，因為沒有新證據，讓 A* 用 penalty 重新嘗試）。
2 個回歸測試、59 單元測試 + sim 整合測試全過。**修正前後對照**（同一
episode）：`select_none` 66→21、`select_ok` 3→6、agent 從困死原地
變成能移動超過 1.3m 突破房間邊界。

**修正後的完整 30-episode 結果**（paper mode 0.13m，`configs/eval/hm3d_val_mini.yaml`）：

| 指標 | 數值 |
|---|---|
| SR | **6.67%**（2/30） |
| SPL | **0.0205** |
| mean dtg | 3.10m |
| mean steps | 330.6 |
| pipeline FPS | 1.41 |

按類別：chair 28.6%（2/7 成功，SPL 0.088，mean dtg 0.78m——明顯是我們
反覆調參最多的類別）；bed/plant/sofa/tv_monitor 全部 **0%**（5, 5, 5, 7
個 episode）；toilet 0%（僅 1 個，已知樓層限制）。

**關鍵訊號——「近失敗」比例遠高於成功率**：dtg < 0.5m 的 episode 有
**6 個（20%）**，但嚴格門檻下只算 2 個成功：
`sofa(dtg=0.16)`、`bed(dtg=0.16)`、`plant(dtg=0.20)`、`plant(dtg=0.43)`
都差一點點就過門檻。這代表 `approach_stop_bbox_px` 校準（P1c 原本列
為「待辦」）投報率可能很高——不需要新能力，只是停止時機/bbox 閾值沒
抓準，光是這項調整就有機會讓 SR 顯著提升。

**其他訊號**：
- `verification` 計時異常慢（mean 35.2s、max 44.6s）——遠比先前隔離
  測試時慢，懷疑是長時間連續跑（~4-5 小時）造成 VRAM/ollama 競爭累積
  （模型反覆換入換出），需要在 P2 效能項目一併排查。
- 同一個 episode（bed/ep11）在這次全量跑中**失敗**（41 步），但在 P1a
  單獨測試時曾達到 dtg=0.015 的完美結果——證實 LLM/VLM 溫度 + 探索
  隨機性造成的 run-to-run 變異很大，單次「成功」不能當成穩定能力的
  證明，這也是為什麼要擴大樣本數的原因。
- `select_none` 修正後在全部 30 個 episode 中沒有再出現病態比例
  （最糟 48:9≈5.3:1，遠不到修正前 66:3、86:1 的鎖死程度）。

### P1e — bbox 校準資料收集結果（2026-07-15）：假設被真實數據推翻

**動機**：P1d 發現 20% 的 episode 是「近失敗」，假設是
`approach_stop_bbox_px`（全域單一門檻 40000px²）沒有依物件類別校準。
在動手改之前，先替 `_do_approach` 加上 `approach_bbox_log`（記錄每次
偵測到的 bbox 面積）與 `approach_stop_reason`（bbox/retreat/deadline/
path_consumed），跑一次 30-episode 收集真實數據再下結論——本次 session
反覆學到「先猜測門檻、後被數據打臉」的教訓（`approach_stop_bbox_px`
本身、驗證器準確率都曾經歷這個過程），這次先驗證再動手。

**結果：假設不成立，不建議實作 per-category 門檻**。逐一比對全部 30 筆
`approach_bbox_log` + `approach_stop_reason` 後：

1. **chair 與 plant 全程從未觸發 `bbox` 停止**（0/12 個 episode）——
   bbox 最大值只到 8000–27000px²，遠低於 40000，不論成功或失敗皆然。
   對這兩類物件，門檻在實務上等同不存在，調整它不會改變任何行為。
2. **sofa/bed/tv_monitor 觸發 `bbox` 時一律是第一次檢查就爆表**
   （42325–116760px²，`bbox_log` 常常只有 1 筆），從未出現「卡在門檻
   邊緣」的情況——代表門檻同樣不是這些類別停止時機的決定因素。
3. **決定性反證**：bed 的一次成功（dtg=0.11, bbox=116760）與一次近
   失敗（dtg=0.16, bbox=115790）**bbox 數值幾乎相同**。若門檻是決定
   因素，這兩者不該一成一敗。真正的差異必然在別處——最可能是視點
   選擇（`ViewpointPlanner`）或物件橢球中心估計的幾何精度，而非
   bbox 視覺確認門檻。

**結論**：機械化實作 per-category `approach_stop_bbox_px` 會是沒有數據
支持的改動，不執行。真正值得投入的下一步（未開始）：直接比對我們的
停止位置與 habitat 該 episode 的 view_points 幾何邊界（P0 時期用過的
方法），確認近失敗的 0.15–0.2m 落差來源——物件位置估計偏差、視點
規劃器取樣半徑、還是格線/到達容差堆疊——這是全新的調查方向，範圍
不同於「校準 bbox」，留待下次決定是否進行。

實作：`src/osg/agent/nav_agent.py`（`approach_bbox_log`/
`approach_stop_reason` 儀表化）、10 個 nav_agent 測試更新驗證新欄位，
59 單元測試 + sim 整合測試全過。

### P1f — 收尾容差修正（2026-07-15）：找到並修正真正的落差來源

**方法**：`scripts/viewpoint_geometry_check.py` 直接查詢 habitat 該
episode 的實際 `view_points` 座標，跟我們的 `final_xy` 算歐氏（直線）
距離，對照 habitat 回報的（測地線）`distance_to_goal`。

**發現**：sofa(dtg=0.158/0.169)、bed(dtg=0.158)、plant(dtg=0.157) 這幾個
近失敗案例，我們的實際停止位置離最近 view_point 只有 **4.6–7.8 公分
（直線距離）**——幾乎就站在上面了，但 habitat 回報的測地線距離卻是
0.157–0.169m，剛好卡在 0.13m 門檻外。**落差穩定落在 0.09–0.11m**，
代表測地線路徑必須繞過附近的薄障礙物（牆角、家具邊緣）才能真正走到
那個視點格。

**這也推翻了 P1a 的假設**：「貼近物件會衝出視點集合」。bed 那次成功
（dtg=0.106）的直線落差（0.035）反而是這批案例裡最小的——代表**越
接近越可能成功**，不是越遠。P1a 移除的三個舊終端策略失敗的真正原因
應該是「瞄準錯的點」（物件粗估中心），不是「太近」本身。

**根因**：`AStarPlanner` 的預設 `goal_tolerance_m=0.3` 與
`WaypointController` 寫死的 0.2m 抵達判定，是為 frontier/verify-view
移動調校的（多走幾公分不影響效率），沿用到 APPROACH 的最終收尾段時，
就留下了足夠的餘裕讓測地線繞路吃掉成功半徑。

**修正**：`AStarPlanner.plan()` 與 `WaypointController.act()` 都加上
可選的逐次呼叫覆寫參數（`goal_tolerance_m` / `arrival_tol_m`，預設值
維持原本的建構時數值，frontier/verify-view 行為不變）。`NavAgent.
_follow_to`（只有 APPROACH 用）改用新設定
`agent.approach_goal_tolerance_m=0.12`、`agent.approach_arrival_tol_m=0.1`
——比預設 0.3/0.2 更緊，但仍高於 0.05m 的 costmap 解析度以維持穩健。

3 個新測試（planner/controller 逐次覆寫、nav_agent 用 spy 確認真的把
設定值傳到底層元件），62 單元測試 + sim 整合測試全過。

**驗證評估結果（30 episodes，`outputs/20260716_075940` vs P1e baseline
`outputs/20260715_204234`）**：

| | P1e（修正前） | P1f（修正後） |
|---|---|---|
| SR | 0.100 (3/30) | 0.133 (4/30) |
| SPL | 0.0261 | 0.0335 |

表面上有進步，但逐 episode 比對後發現**這個改善不能歸因於本次修正**：

1. **30 episode 中唯一翻盤的只有 1 個**（ep10/chair：P1e 失敗
   dtg=4.549、`approach_stop_reason=None`；P1f 成功 dtg=0.109、
   `stop_reason=deadline`）。但 P1e 那次**根本沒進入過 APPROACH 狀態**
   （`stop_reason=None` 代表探索/驗證階段就把步數耗盡），而本次修正
   只動了 `_follow_to`（僅 APPROACH 用）——代碼路徑上不可能是這次改的
   東西造成的。這個翻盤只能是探索階段的隨機性（VLM frontier 評分、
   ollama 取樣溫度）造成的路徑分岔。
2. **驗證了先前標記的「即時 bbox 觸發」缺口確實存在且影響最大宗**：
   ep13/sofa、ep5/bed、ep11/bed、ep12/bed 四個追蹤案例，`approach_bbox_log`
   長度都是 1（一進 APPROACH 第一次檢測就達標停止），兩次跑的
   steps/dtg/bbox 值**逐位元組相同**——證實這些案例從未呼叫過
   `_follow_to`，本次的容差修正對它們是空操作。
3. **唯一真正走過 `_follow_to` 多步的追蹤案例（ep4/plant）**：P1f 確實
   多繞了幾步（`approach_bbox_log` 從 2 筆變 11 筆，`stop_reason`
   從 retreat 變 deadline），但**最終 dtg 完全沒變**（都是 0.157）。
   代表就算給更緊的收尾容差、多走幾步，agent 也只是在同一個可通行邊界
   附近繞更久，最後停在同一個位置——這比較像是 costmap 膨脹半徑把
   agent 卡在離家具輪廓固定距離的地方（結構性的可通行邊界問題），
   不是容差不夠緊的問題。
4. **同批次也觀察到大幅度的執行間不確定性**：ep2/tv_monitor 從
   P1e 的 dtg=0.166（近失敗，bbox 觸發）變成 P1f 的 dtg=5.571（探索
   階段整個走岔，從未偵測到目標）；ep11/sofa 從 dtg=0.169 變 0.261。
   兩者都跟本次修正的程式碼路徑無關，純粹是 30 episode 規模下
   VLM-based frontier 評分的隨機性造成的路徑分岔。

**結論**：P1f 修正本身無害（62 單元測試 + sim 整合測試全過），但**這批
資料無法證明它對 SR/SPL 有實質貢獻**——唯一的翻盤案例與修正的程式碼路徑
無關，而唯一真正測試到修正效果的案例顯示容差放寬到 0.1m 也追不上結構性
的可通行邊界限制。**不建議照原計畫再去收緊 GOTO_VERIFY_VIEW 的抵達容差**
（`_follow_path` 目前仍用 0.2m/0.3m 預設）：同樣的機制（把容差調緊、逼
agent 多走幾步）在 ep4 這個唯一的直接證據上沒有改善最終距離，說明瓶頸
不在容差寬鬆，而在障礙物周圍的可規劃空間本身。若要再往這個方向查，
下一步應該是量測 costmap 膨脹半徑（`mapping.inflate_margin_m=0.07` +
`agent_radius=0.18`）造成的「離物件輪廓最近可站立格」距離，而不是繼續
調小 arrival tolerance 數值。另外，**這批資料也提醒：30 episode 規模下
跨次執行的比較必須考慮探索階段本身的隨機性**，未固定 seed 前，個別
episode 的前後對比說服力有限，只有「即時 bbox 觸發」這種在 APPROACH
之前完全走同一條路徑的案例才具備逐位元組可重現性。

### P1g — 站立距離量測（2026-07-16）：膨脹半徑假設也被推翻

**方法**：新增 `scripts/standoff_check.py`，重跑追蹤的近失敗 episode，在
episode 結束當下直接量測 agent 自己的即時 costmap：`final_xy`／APPROACH
目標點（`_nearest_free_xy`）到最近一個 sensed-OCCUPIED 格的距離
（`scipy.ndimage.distance_transform_edt`），對照
`inflate_radius_m = agent_radius(0.18) + inflate_margin_m(0.07) = 0.25m`
與 costmap resolution 0.05m。

**發現**：6 個追蹤 episode 中 5 個這次重跑探索路徑整個走岔（dtg 從原本
0.15–0.26 暴增到 1.1–7.5，再次印證 P1f 已記錄的探索階段隨機性），只有
**ep4/plant 精確重現**（dtg=0.158 vs 原本 0.157）。在這個唯一乾淨的樣本
上：`occ_from_final = occ_from_goal = 0.050m`——剛好等於 grid resolution，
agent **緊貼著 sensed 障礙物表面**，離 0.25m 的膨脹半徑還很遠。

補測 `ep11/bed`（原本因 script 的 dict-key 覆寫 bug 漏測，修正後補跑）
同樣**精確重現**（dtg=1.970，跟 P1e/P1f 兩次原始跑的數字完全一致），
`occ_from_final` 再次落在 **0.050m**——兩個逐位元組可重現的乾淨樣本，
兩個都顯示 agent 貼到 grid resolution 的距離，進一步坐實膨脹半徑不是
瓶頸的結論（這個 episode 本身是路徑耗盡的大範圍失敗，`occ_from_goal
=0.292` 顯示原始目標點離障礙物較遠，但 agent 耗盡路徑後仍能停在只
剩 0.05m 的位置，而非卡在 0.25m 邊界）。

**這推翻了 P1f 結尾寫的「costmap 膨脹卡住站立距離」假設**：`planner.py`
的膨脹本來就只是 A* 的 soft cost（8x 懲罰，見 `planner.py:65-71`
的既有註解），從未 hard-block，agent 實際上可以、也確實走到只剩一個
grid cell 的距離。（另外 5 個走岔樣本中有 2 個落在 0.25–0.29m，但那些
軌跡本身已不可信，无法歸因於膨脹半徑本身。）

**結論**：容差／膨脹半徑這條調查線到此為止——agent 已經逼近 grid
resolution 的物理極限，沒有更多空間可以透過調整這些參數擠出來。
P1f/P1g 加起來看，剩餘 ~0.09–0.11m 的測地線落差最可能來自兩個更難處理
的來源，而非任何容差/膨脹旋鈕：
1. 我們用深度感測到的「障礙物表面」跟 HM3D 用 ground-truth mesh 定義的
   `view_point` 本來就不是同一個參考基準；
2. 我們對目標物件位置的估計（ellipsoid centroid，來自帶雜訊的偵測+
   深度）本身可能偏離真實表面幾公分，導致 `_nearest_free_xy` 是繞著
   一個略微偏移的估計點在找最近格。

**後續方向建議**：不再往 APPROACH 容差/膨脹半徑調參數；優先順序應該是
(a) 解決探索階段的執行間隨機性（固定 RNG/ollama 取樣種子）以取得可信的
前後對比，或 (b) 轉去做 P2（效能，尤其驗證計時異常已兩次獨立確認
mean 33–35s）這種證據更扎實的項目。

### P1h — 偵測器 vs Ground Truth 比對（2026-07-16/17）：找出雜訊的真正來源

**動機**：懷疑「semantic segmentation 雜訊太大」是不是 SR 低的主因，且
可能是 P1g 剩餘測地線落差的另一個來源（物件位置估計偏移）。

**方法**：新增 `scripts/detector_gt_check.py`。啟用 habitat-sim 的
semantic sensor（`HabitatSimSemanticSensorConfig`）+ HM3D 的 annotated
scene dataset（`hm3d_annotated_basis.scene_dataset_config.json`，
HM3D-semantics v0.2 標註本來就存在，只是主 eval pipeline 沒開這個
sensor），跟 rgb/depth 用同解析度/hfov 對齊。關鍵發現：episode
`goals[i].object_id` **直接等於** semantic sensor 逐像素回傳的
`semantic_id`（實測驗證：`object_id=32` -> `category.name()="chair"`
一致），所以可以用 goal 的 object_id 集合直接切出「這個 episode 真正
算成功的目標實例」的 pixel-perfect mask，不需要自己猜測類別字串對應
關係。對每個取樣 frame，比對 YOLOE 的偵測（mask 由 `-seg` checkpoint
直接提供）跟這個 GT mask，分類成 TP / FN / FP-wrong-instance（偵測到
同類別但不是目標實例）/ FP-hallucination（偵測到的東西跟該類別完全
不重疊，純粹認錯）。

**踩到的資料集細節**：HM3D 的 raw 語意類別字串是 ObjectNav 類別的同義詞
集合，不是字面字串——例如 "plant" 這個 episode 類別，底層 goal 實例的
`category.name()` 其實是 `flowerpot`/`flower vase`/`decorative plant`，
不是 "plant" 本身。這代表任何直接拿 `episode.object_category` 字串去比
對 `sem_scene.objects` 類別名稱的做法（我一開始就是這樣寫，之後修正）
在 "chair" 這種類別上恰好對得上，但在 "plant" 上會完全比對失敗——已在
腳本裡改用「先查 goal 實例自己的 raw 類別字串，再用這組字串去找同類別
的其他實例」，避免這個陷阱。

**六類別小規模結果**（每類別 2–3 episodes，最多 300 步，每 3 步取樣一次）：

| category | recall | FP-halluc | mean IoU | mean offset(px) | @1.5m 換算 |
|---|---|---|---|---|---|
| chair | 83.3% | 2/111 (~2%) | 0.162 | 61.7 | ~20cm |
| bed | 62.5% | 0/30 (0%) | 0.246 | 51.2 | ~17cm |
| tv_monitor | 44.0% | 4/113 (~3.5%) | 0.306 | 51.9 | ~17cm |
| sofa（第一次跑，2 episodes） | 0% (n=1) | 38/76 (~50%) | – | – | – |
| sofa（第二次跑，3 episodes，多跑一集 ep10） | 95.0% (n=40) | 25/195 (~13%) | 0.326 | 52.1 | ~17cm |
| plant | 0% (n=8) | 8/200 (4%) | – | – | – |
| toilet | 無有效樣本（GT 全程未進入視野） | 0/100 | – | – | – |

**重要警示：recall/FP-halluc 的絕對數字本身不穩定**。同一組 sofa
episode（ep3+ep13）在兩次獨立跑中，recall 從 0% 跳到 95%、FP-halluc 比例
從 ~50% 跳到 ~13%——這跟 P1f/P1g 已經記錄的探索階段隨機性（VLM frontier
評分、ollama 取樣）完全一致：agent 這次走的路線恰好讓它更常正面看到
沙發，取樣到的 frame 組成就完全不同。**兩三個 episode 規模下，這些
百分比只能當作參考量級，不能當精確的類別能力比較**；IoU/offset 這兩個
「只在偵測正確時才計算」的量反而在兩次跑之間相對穩定（0.246–0.326、
51–62px），可信度較高。

**sofa 幻覺率的根因（crop 視覺檢查confirmed）**：把兩次跑總共 25+38=63
個 FP-hallucination 的 crop（帶 GT 實際類別標籤）全部 dump 出來看，
**68%（17/25，第二批同樣以此類別為主）都是同一類問題：把 armchair
（單人扶手椅／貴妃椅）誤認成 sofa**。實際看圖確認：ep3/ep13 裡反覆
被誤判的是**同一張圓潤無扶手分隔線、豹紋布套的貴妃椅**，agent 在同一
episode 裡從不同角度經過它時每次都被叫成 sofa（score 0.63–0.83，相當
自信）。少數非 armchair 的誤判（piano close-up、wall/window 邊緣裁切）
看起來才是真正隨機的雜訊，而且部分「wall/window」標籤本身是量測方法
的假象——mask 邊界裁切到背景較多時，多數決類別會跑掉，但視覺上主體
仍是同一張扶手椅（見 `ep3_step63_actual-wall_score0.72.png`）。

**根因很具體，且可驗證**：檢查 `core/config.py` 的 `DEFAULT_VOCABULARY`
（~40 個類別），**裡面沒有 "armchair"**。YOLOE 是開放詞彙偵測器，只能
從給定的詞彙表裡選標籤——看到一張扶手椅但詞彙表沒有 "armchair" 這個
選項時，"sofa" 顯然是模型認為最接近的候選字，於是系統性地把它分類
成 sofa。這不是模型能力不足，是**詞彙表設計問題**。

**結論與建議**：
1. Recall 的類別間巨大差異（plant/sofa 在單次跑可以是 0%，chair 可以是
   83%）目前無法用這批小樣本區分「類別真的很難認」還是「探索路徑剛好
   沒帶它靠近目標」——擴大到更多 episode/固定種子後才能真正回答。
2. sofa 的幻覺問題則已經有具體、可驗證的根因：**armchair 不在偵測詞彙
   表裡**。建議把 `armchair`（以及可能同樣會被錯認的 `loveseat`/
   `recliner`/`ottoman` 等相近家具類別）加進 `DEFAULT_VOCABULARY`，
   讓 YOLOE 有機會正確區分兩者，而不是被迫塞進最近的候選類別。這是一個
   低成本、證據充分、可以直接動手做的修正，跟 P1f/P1g 的導航容差調整
   是完全不同層級的問題（偵測輸入雜訊，而非路徑規劃/收尾精度）。
3. plant 類別的 0% recall 疑似跟其底層 GT 類別是 flowerpot/flower vase/
   decorative plant 這些同義詞有關——detector 對這些具體視覺樣式的辨識
   力可能本來就弱，值得比照 sofa 的做法把 crop dump 出來看，但屬於
   不同的後續調查（本次未做）。

### P2 — 效能（real-time 主張）
- 30-episode 全量兩次獨立測得 pipeline FPS 1.41–1.47，控制迴圈中位數
  ~480–710ms；目標 ≥5 FPS（優於論文的 RTX 3060 9.86 FPS 需在 12GB
  機器驗證）。
- **`verification` 計時異常且已兩次獨立確認**：mean 33.7–35.2s、max
  44.6–47.5s（P1d/P1e 兩次跑都測到，數字幾乎一致，排除單次偶發）。
  遠比先前隔離測試時慢，懷疑是長時間連續跑（4–8 小時）造成 VRAM/
  ollama 競爭累積（3B/7B 模型反覆換入換出，partial CPU/GPU offload
  比例可能隨時間惡化）。這是目前最大的單一效能瓶頸，優先度應提高。
- 剩餘熱點：object_layer mean 364ms（尖峰 3.4s，優化觸發時機）、
  frontier_extract/select 各 mean ~1000ms（已節流）、detector mean 82ms。
- 候選：A* 改 scipy/C 實作或 FMM；frontier 評分快取；房間分割增量化；
  ollama 常駐策略調整（避免長跑期間的模型換入換出，或定期重啟釋放）。

### P3 — 實驗與論文素材
- `--multirun +ablation=full,no_verify,paper_baseline,no_llm`（機制已就绪）。
- Full val（2000 episodes）在 12GB+ 機器跑（需下載 hm3d_val_v0.2，~30GB）。
- HM3D-OVON 評估（open-vocabulary 主張的直接證據）— 需新 episode loader。
- ScanNet R@d scene graph 精度（M7，`to_json()` 介面已預留）。
- 12GB profile 數據（yoloe-11l + qwen2.5vl:7b）。

## 7. 測試規劃

| 層級 | 內容 | 指令 | 頻率 |
|---|---|---|---|
| 單元 | 47 tests：橢球投影 round-trip、W2 恆等、關聯/深度閘、costmap raycast、frontier、房間分割、A*、P/d 選擇、serializer golden、controller、JSON 解析、async | `make test`（<5s） | 每次改動 |
| 偵測器隔離 | YOLOE 載入/詞彙表轉換/fp16 路徑 | 見 git log 中的 heredoc script，可固化為 `tests/integration @gpu` | detector 相關改動 |
| 整合 | 1 episode + stub detector（不依賴模型權重） | `make test-sim`（~4min） | pipeline 改動 |
| 端到端 | 3 episodes 真模型 + summary/timing/viz | `make eval-mini`（~30–60min） | 每輪修正後 |
| 診斷 | 單 episode + stats/state_log | `run_eval.py eval=hm3d_val_mini eval.num_episodes=1` | debug 時 |
| 回歸基準 | minival 全部 episodes 固定 seed | 待 P1 建立 baseline 數字後納入 | 里程碑 |

**測試紀律**（本次除錯的教訓）：
- 任何含迴圈的幾何程式碼必須有單元測試 + pytest `--timeout`（Bresenham
  無窮迴圈曾讓 pytest 空轉 4 小時）。
- 完整 log 必須落檔（`tail` 截斷曾讓 dtype 錯誤的呼叫點消失）。
- eval 前先跑隔離的元件 smoke（模型載入類錯誤 5 秒可測，別等 5 分鐘的
  場景載入）。

## 8. 指令速查

```bash
make build            # 建 nav image
make up               # 啟動 ollama
make pull-model       # 拉 qwen2.5vl:3b
make smoke            # EGL + LLM 往返
make test             # 單元測試
make test-sim         # sim 整合測試
make eval-mini        # 3-episode 端到端
# 資料（一次性；token 見 Matterport 帳號）
python scripts/download_data.py --username <ID> --password <SECRET> --uids hm3d_minival_v0.2
python scripts/download_data.py --episodes-only
# ablations
python scripts/run_eval.py --multirun +ablation=full,no_verify,paper_baseline,no_llm
```

輸出目錄：`outputs/<timestamp>/{summary.json, episodes.jsonl, timing.csv, viz/*.png}`。
`summary.json` 記錄 dataset version 與 config 指紋；`episodes.jsonl` 含
per-episode 的 llm/verify 統計、`agent_stats` 與 `state_log`（診斷用）。
