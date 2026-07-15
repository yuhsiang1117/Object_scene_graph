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
- 驗證器單獨評測集：把 verify_debug 影像整理成 20-30 張標注測試集，
  參數改動先過離線測試再進 eval。
- `approach_stop_bbox_px` 目前只有 2 個場景的樣本點，8–10 episodes 後
  應重新校準（不同物體類別的「夠近」bbox 差異可能很大——沙發 vs 馬桶）。
- 8–10 episodes 掃參；同時報告 0.1（標準）與 0.13（paper mode）兩組數字。

### P2 — 效能（real-time 主張）
- 目前控制迴圈中位數 ~400ms（2.5 FPS）；目標 ≥5 FPS（優於論文的 RTX
  3060 9.86 FPS 需在 12GB 機器驗證）。
- 剩餘熱點：frontier_select 789ms（A*×5，已節流）、object_layer 尖峰
  （優化觸發時機）、detector 66ms。
- 候選：A* 改 scipy/C 實作或 FMM；frontier 評分快取；房間分割增量化。

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
