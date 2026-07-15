# Object Scene Graph ObjectNav — 設計文件與改進路線圖

> 更新日期：2026-07-15。本文件記錄系統目前的設計、已驗證狀態、eval 除錯歷程、
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
        GOTO_VERIFY_VIEW → VERIFYING ──accept──> GOTO_TARGET → STOP
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
  走到視點 → 轉向面對物件 → VLM 以 live view（主）+ best crop（輔）判斷
  `{is_target, confidence}` → 拒絕即 blacklist。
- **終端接近**：目標 = 距物件中心最近的 FREE 格；單次規劃、路徑走完即
  STOP、100 步 deadline（防繞圈）。

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
| 單元測試（47 個，合成資料免 GPU） | ✅ 3.3s 全過 |
| Ollama 文字+視覺往返 | ✅ |
| HM3D minival（10 場景+semantic configs）+ ObjectNav v2 episodes | ✅ 已下載 |
| Sim 整合測試（stub detector 全 pipeline 1 episode） | ✅ |
| 端到端 eval（YOLOE+VLM 3 episodes 跑完、產出 summary/timing/viz） | ✅ 管線通、**SR 尚為 0** |
| VLM frontier 評分實際運作 | ✅（GPU 後 0 錯誤） |
| **SR > 0** | ❌ 進行中（見 §5） |

## 4. Eval 迭代記錄（除錯史與教訓）

| 輪 | 修正 | 結果/教訓 |
|---|---|---|
| 1–6 | Hydra GlobalHydra 衝突、YOLOE fp16 文字頭 dtype、CLIP AutoUpdate、outputs 權限、VRAM OOM | 環境層問題全清；教訓：**ultralytics checkpoint 是 fp16**、**掛載點擋 symlink**、**容器 root 檔案汙染 host** |
| 7 | 首次跑通 | SR 0/3、0.36 FPS；costmap 1.6s/幀（Python raycast）；VLM 100% context 爆掉；verify 92s/次 |
| 9 | costmap 向量化、影像減量 | costmap **38ms** ✅；VLM 改超時（CPU 推理 200s+ 不可行） |
| 10 | ollama 回 GPU、GOTO 不繞圈 | VLM **0 錯誤** ✅ FPS 2.2；新問題：verifier 過嚴，2/2 拒絕真目標 |
| 11–12 | verifier 面向物件+live view+寬容 prompt；啟動時卸載 ollama | 揭露**當前主問題**：agent 幾乎不移動（見 §5） |
| 13 | 加入 stats/state_log 儀表化 | 單 episode 診斷 run 被中止（使用者要求停止），**儀表化程式碼已就位** |

## 5. 當前未解問題（下一步的起點）

**主 blocker：agent 在 500 步內幾乎不移動。**

- 證據：ep4 軌跡圖（`outputs/20260715_010000/viz/ep4.png`）路徑只是起點
  旁一小坨；地圖呈細條輻射狀（單一視角的 line-of-sight）；物件偵測正常
  （掃描到 30+ 物件）。
- 主要假設（未驗證）：細窄 free space + 0.25m 障礙膨脹 → A* 起點
  nudge 失敗或 frontier 路徑全數規劃失敗 → 所有 frontier 進入 50 步
  blacklist → EXPLORE 原地打轉循環。
- **驗證手段已備好**：`NavAgent.stats`（plan_ok/plan_fail/select_none/
  select_ok）與 `state_log`（狀態轉換序列）已寫入 episodes.jsonl，跑
  `eval.num_episodes=1` 即可確診。
- 次要疑點：verifier 對 chair 的 2/2 拒絕在 live-view 修正後尚未重新驗證
  （eval 11/12 都死在更早的環節）。

## 6. 改進規劃（優先順序）

### P0 — 讓 SR > 0（movement bug）
1. 跑單 episode 看 stats：確認 plan_fail / select_none 比例。
2. 若確認規劃失敗：
   - 縮小膨脹半徑（0.25→0.15m）或改用「膨脹後不可行才退回未膨脹」的
     兩段式規劃；
   - `_nudge_free` 搜尋窗從 0.3m 放大到 1m；
   - frontier 目標點改取「frontier 質心最近的可規劃 free 格」。
3. 檢查 costmap free space 是否過細：目視 raycast 覆蓋、必要時調
   `depth_stride`（4→2）或障礙帶高度參數。
4. 重驗 verifier（live view 版本）對 chair/bed 的接受率。

### P1 — SR 調參與魯棒性（P0 解除後）
- 逐 episode 檢查 viz + state_log，分類失敗型態（探索不足/誤偵測/停止誤差）。
- 停止距離策略再校準（habitat success 是「到 goal viewpoint 的測地距離
  < 0.1m」，非物件中心歐氏距離）。
- 調 `assoc_score_thresh`、`accept_confidence`、`unscored_prior`。
- 8–10 episodes 的 minival 小規模掃參。

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
