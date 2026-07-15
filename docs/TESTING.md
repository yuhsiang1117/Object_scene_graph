# 測試策略、現況與改進

> 對應版本：2026-07-16。架構背景見
> [ARCHITECTURE.md](ARCHITECTURE.md)。

## 1. 測試金字塔

| 層級 | 依賴 | 耗時 | 指令 | 使用時機 |
|---|---|---|---|---|
| 單元（55 tests / 15 檔） | 無 GPU、無資料、合成輸入 | ~8s | `make test` | 每次改動 |
| 元件隔離 | GPU + 權重 / ollama | 秒~分 | 見 §4 | 模型相關改動 |
| Sim 整合 | habitat + HM3D minival | ~4min | `make test-sim` | pipeline 改動 |
| 端到端 eval | 全部 | 30–60min | `make eval-mini` | 每輪修正驗收 |
| 診斷 | 全部 | ~10min | `run_eval.py eval=hm3d_val_mini eval.num_episodes=1` | debug |

原則：**問題要在最便宜的層級被抓到**。25 輪 eval 除錯的最大教訓是
「模型載入類錯誤 5 秒可測，不要等 5 分鐘的場景載入去發現它」。

## 2. 單元測試詳表（tests/unit/）

### 共用 fixture（conftest.py）
- `intrinsics`：640×480、fx=fy=320 的相機。
- `make_camera(position, look_at)`：構造 OpenCV 慣例的 `T_wc`
  （預設 up = 世界 −y，配合 y-up 世界 + y-down 相機）。
- `draw_ellipse_mask(h, w, mu, semi_axes, angle)`：畫填滿橢圓 mask。
- `make_frame(...)`：常數深度的合成 FrameData。

### 幾何 / 物件層

| 檔案 | 測試 | 驗證內容 |
|---|---|---|
| test_ellipsoid.py（6） | 投影正確性 | 正對/偏心/側視三種相機下投影中心與尺寸合理 |
| | cheirality | 相機後方 → `project()` 回 None |
| | 初始化 round-trip | 合成偵測 → `init_from_detection` → 中心誤差有界 |
| | mask 矩 | `ellipse_from_mask` 對合成橢圓恢復 μ 與半軸 |
| test_wasserstein.py（3） | `sqrtm_2x2_spd` | S@S == M（對角與一般 SPD） |
| | 零殘差恆等 | GT 橢球對自身觀測殘差 <1e-4 |
| | 優化收斂 | 擾動初值（+0.3m、1.5×半軸）+ 5 合成視角 → 中心誤差 <5cm |
| test_association.py（4） | 重偵測關聯 | 同位姿兩次偵測 → 1 track、n_obs=2 |
| | 異類別分離 | chair/table 同位置 → 2 tracks |
| | 深度閘 | 同影像位置、深度差 2 倍 → 拒絕關聯 |
| | 候選過濾 | min_obs / label 正規化 / blacklist 全路徑 |

### Mapping / 規劃

| 檔案 | 測試 | 驗證內容 |
|---|---|---|
| test_costmap.py（3） | raycast 標記 | 合成深度：中途 FREE、牆端 OCCUPIED、相機後方 UNKNOWN |
| | 自動擴張 | 出界更新後座標往返一致 |
| | 膨脹 | `inflated(r)` 依半徑正確 dilate |
| test_frontier.py（3） | 開口偵測 | 帶 8 格門縫的房間 → 恰 1 個 frontier、質心在縫上 |
| | min_cells 濾噪 | 1 格 unknown 針孔（4 格 frontier）被濾、門縫（8 格）保留 |
| | 質心去重 | 相近 frontier 依 dedup_m 合併 |
| test_room_seg.py（3） | 雙房分割 | 3×3m×2 + 0.4m 門 → 恰 2 房、兩側 id 不同 |
| | id 穩定 | 地圖增長後重跑，舊房 id 不變 |
| | 不過度分割 | 6×6m 開放空間 → 恰 1 房 |
| test_planner.py（4） | 直線成本 | 2m 直線 cost ≈ 2.0 |
| | 繞牆 | 中央牆 → cost > 2.5 |
| | 不可達 | 目標被 OCCUPIED 全包 → fail |
| | unknown 可通行 | 目標在未知區仍可規劃（frontier 前提） |
| test_selector.py（3） | P/d 權衡 | 近低分 vs 遠高分的兩個方向 |
| | blocked 過濾 | 被封鎖候選跳過 |
| | 無分數先驗 | 空分數表仍能選（prior 0.3） |
| test_controller.py（4） | 對齊前進 | 面向 waypoint → forward |
| | 轉向方向 | **+z 在右 → turn_right**（habitat 轉向慣例的回歸鎖） |
| | 抵達 | 最後 waypoint 0.2m 內 → None |
| | 卡住標記 | 2 次無位移 forward → stuck + 前方格 OCCUPIED |

### 探索 / 驗證 / LLM

| 檔案 | 測試 | 驗證內容 |
|---|---|---|
| test_async_scorer.py（3） | 不阻塞 | 0.3s 慢 scorer，request() <0.1s 返回、稍後 latest() 有分數 |
| | in-flight 丟棄 | 併發第二請求被拒、底層只呼叫一次 |
| | 例外隔離 | scorer 拋錯 → n_errors+1、latest() 空、迴圈不死 |
| test_viewpoint.py（3） | 開放房間 | 物件方塊周圍找到 0.7–2.1m 視點 |
| | 尊重牆 | 三面圍牆 → 視點必在開口側（LOS） |
| | 全未知 | 無 FREE 格 → None |
| test_serialize.py（3） | golden 文字 | 房間層級 + near 註記 + Hallway 段落逐字比對 |
| | 空圖 | "(no objects mapped yet)" |
| | JSON round-trip | to_json 可序列化、欄位齊全 |
| test_llm_parsing.py（5） | `extract_json` | 純 JSON / 夾雜文字 / 無 JSON 拋錯 |
| | `_parse_scores` | 分數 clamp 到 [0,1] + 過濾未知 id / 無 "scores" wrapper 也可解析 |
| test_nav_agent.py（8） | `_do_approach` 四分支（P1a） | bbox 達標即停 / 可見未達標則續走+記錄 last_good_xy / 忽略非目標類別偵測 / 視野遺失後退回 last_good_xy（不佔用前進步數）/ 從未可見時退避前進（steps_left 遞減）/ step 預算耗盡即停 / deadline 到即停 / 目標被 OCCUPIED 包圍無法規劃即停 |

## 3. Sim 整合測試（tests/integration/）

`test_one_episode.py`（`@pytest.mark.sim`、timeout 600s）：
真 habitat episode + `StubDetector`（隔離模型權重）+ NearestScorer，
100 步上限；斷言：跑得完、costmap 覆蓋 >100 格、habitat metrics 齊全。
資料未掛載時自動 skip。
定位：**驗 pipeline 接線，不驗智慧**——它抓 config/座標/API 層的破壞，
抓不到行為退化（行為靠 eval-mini + 診斷工具）。

## 4. 元件隔離測試（未固化為 pytest，模式已建立）

| 元件 | 方法 | 曾抓到的 bug |
|---|---|---|
| YoloeDetector | 容器內 heredoc：init → set_vocabulary → detect → 再 set_vocabulary（新類別） | fp16 文字頭 dtype、詞彙表順序誤判、AutoBackend 半精度快取 |
| VLM 驗證 | `scripts/prompt_lab.py <img> <target> ...`：存檔影像 × 多種 prompt 離線比較 | 3B 誤拒真目標、小圖解析度懸崖、describe-then-decide 有效性 |
| 導航棧 | `scripts/diag_movement.py`：stub detector + nearest scorer 200 步，輸出逐步狀態/動作分布/plan 統計/plan probe（逐 frontier 失敗原因）/costmap+膨脹視覺化/`diag_costmap.npz` | P0 全部四個 bug、give-up 網有效性 |

**建議固化**（P1）：把 heredoc 寫成 `tests/integration/test_detector.py
(@gpu)`；把 `verify_debug/` 累積的影像整理成 20–30 張標注集 +
`tests/offline/test_verifier_bench.py`（驗證器參數改動先過離線集）。

## 5. 執行期診斷工具

| 工具 | 位置 | 內容 |
|---|---|---|
| `episodes.jsonl` | `outputs/<run>/` | 每 episode：success/spl/dtg/steps/control_fps、llm_calls/errors/`llm_last_error`、verify_calls/rejections、`agent_stats`（plan_ok/fail、select_ok/none、frontier_give_up）、`state_log`（前 40 個狀態轉換）、`final_xy` |
| `timing.csv` | 同上 | 每模組 count/mean/median/max ms（控制迴圈 vs 決策延遲分離） |
| `viz/ep*.png` | 同上 | top-down 軌跡 + 物件 + costmap |
| `verify_debug/` | 同上 | 每次驗證的證據影像 + VLM 回應 JSON |
| `summary.json` | 同上 | SR/SPL/per-category + config 指紋 + dataset version |

## 6. 測試紀律（25 輪 eval 除錯的教訓）

1. **迴圈必設邊界**：任何含迴圈的幾何/搜尋程式必須有單元測試 +
   pytest `--timeout`（手寫 Bresenham 曾讓 pytest 100% CPU 空轉 4 小時）。
   格線走訪一律用 `core/geometry.bresenham`。
2. **完整 log 落檔**：eval 輸出重導到檔案，別只 `tail` 進終端
   （截斷曾讓 dtype 錯誤的呼叫端消失，多花一輪重跑）。
3. **隔離優先**：模型/prompt 問題先用存檔輸入離線重現
   （prompt_lab 一次 5 秒 vs eval 一次 20 分鐘）。
4. **儀表先行**：行為異常先加計數器/狀態log再猜
   （P0 的 select_none:34 一眼定位）。
5. **對照組要對齊**：episode_id 跨場景不唯一——任何資料集比對用
   scene+id 複合鍵（曾誤判 runner 跑錯目標）。
6. **容器操作精準**：`docker rm -f`（全域 filter）曾誤殺進行中的
   eval；砍容器前先確認名單。`docker exec` 繞過 entrypoint
   （conda env 不生效），一律 `docker compose run --rm nav`。
7. **確定性優先於重試**：驗證第一次呼叫 temp=0；重試才升 temp
   （邊界翻面是取樣噪音，先固定住再談統計）。
8. **一次一個變因**：每輪 eval 只帶一組修正 + 一個 commit，
   25 輪的因果鏈才追得回來。

## 7. 改進路線（測試面）

- [ ] 驗證器離線基準集：`verify_debug/` 影像 → 標注 →
      準確率/誤拒率報表；verifier 任何改動先過此集。
- [ ] `FakeDetector`（habitat semantic sensor → GT 偵測）：
      隔離「感知品質」與「導航/決策」的 eval 變因。
- [ ] minival 全 30 episodes 固定 seed 回歸基準
      （P1 調參穩定後鎖 SR/SPL 底線數字）。
- [ ] 探索覆蓋率測試：N 步後 coverage_cells 下限
      （抓 give-up 參數退化）。
- [ ] CI：單元測試無依賴可直接跑；sim/gpu 測試以 marker 隔離。
- [ ] `record_frames.py` + `run_offline_pipeline.py` 的離線軌跡
      迴歸（M1 工具，目前少用；感知層改動時比 sim 快一個量級）。
