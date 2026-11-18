# Method

> **Superseded as a draft.** The paper is final and transcribed at [`PAPER.md`](../PAPER.md);
> the English draft this annotates (`METHOD_SECTION.md`) has been deleted. This copy is kept
> only for the Chinese review commentary interleaved with it, which exists nowhere else.
> Where it disagrees with `PAPER.md` or with the code, it is the one that is wrong.

We address object-goal navigation from an **open-vocabulary** query: the agent
receives an RGB-D stream with pose and a target named in free text, and must
navigate to an instance of it and stop. Its only persistent representation is an
object-level 3D scene graph it builds itself, online, from that stream. Building
one poses three demands that a closed-set semantic map does not meet. First, the
query may name an object no fixed-vocabulary detector can label, so the front end
must be able to admit instances it cannot name. Second, an object's 3D extent is
not observable from a single view, so the representation must accumulate partial
and oblique observations into one geometric object rather than a point. Third,
the graph must record **what supports what** — that the mug is on *that* table —
because a room centroid is both too coarse to act on and too coarse to be wrong
about. The graph is maintained per storey, since in our benchmark the target is
frequently not on the storey the agent starts on.

The same graph is then used a second time, in a scene that has since changed. The
difficulty is no longer that the map is incomplete but that it is **wrong**: it
asserts positions the agent will walk to and find empty. A wrong map turns one
question into three — where objects are, whether they are still there, and where
to look once they are not — and the two halves of the method are not independent,
because the representation built for the first is chosen for what the third
demands. The projected ellipsoid that associates a detection to a track is the
same projection that decides whether the object *should* have been visible; the
support relation the hierarchy records is the relation the search posterior is
defined over.

**Pipeline overview.** The system takes an RGB-D stream with pose, maintains a
persistent object-level 3D scene graph, and emits one discrete action per
simulation step. Each step begins by estimating which storey the agent is on and
updating that storey's occupancy costmap from depth (§2.4); the more expensive
perception runs only on **keyframes** (§2.1). A keyframe is passed to two
detection channels — one named, one class-agnostic — whose outputs are fused into
a single detection set; each segmentation mask is back-projected through depth to
initialise a dual-quadric ellipsoid, associated with the existing tracks, and
refined once enough views have accumulated (§2.2). Tracks that clear the
admission gates (§2.3) become object nodes of the scene graph, hanging beneath
the rooms segmented from the costmap and the containers qualified by category and
geometry together, with storeys joined by the stair edges the agent has actually
traversed (§2.4–2.6). The same keyframe also updates each track's **presence
belief**, gated on whether the object should have been visible in this frame had
it still been there, so that a detection or an absence moves the belief only when
the frame could have settled the question (§3). On the decision side, each round
arbitrates two kinds of goal under one utility-per-cost index — mapped support
surfaces, which carry a semantic prior, and unexplored frontiers, which are
purely geometric — and hands the winner to the same drive state (§4); when the
belief mass lies on another storey, the same quantity is maximised once more at
the storey level (§4.3). The planner then produces a path and the state machine
emits an action. If a track clears the candidate gates the agent enters its
approach and stops on arrival; and an arrival that finds nothing is itself fed
back as negative evidence, so that navigation rewrites the map rather than merely
consuming it (§3.2).

All constants quoted below are taken from the configuration that
`+experiment=mf5_osg_on_ascent_map` actually composes, and each is attributed to
the module it lives in.

---

## 1. Notation

| 符號 | 意義 | 來源 |
|---|---|---|
| $\mathcal{F}_t$ | 時刻 $t$ 的 frame：RGB $I_t$、depth $D_t$、內參 $K$、外參 $T^{cw}_t$ | `core/types.FrameData` |
| $\mathcal{K}$ | keyframe 集合；$\mathcal{F}_t \in \mathcal{K}$ 當位移 $>\tau_d$ 或轉角 $>\tau_\theta$ | `perception/keyframe.py` |
| $z_k=(\ell_k, s_k, M_k, B_k)$ | 第 $k$ 個 detection：label、score、mask、bbox | `core/types.Detection` |
| $\mathcal{O}=\{o_i\}$ | object track 集合（object layer） | `objects/object_layer.py` |
| $Q_i^{*}$ | track $i$ 的 dual quadric（ellipsoid 表示） | `objects/ellipsoid.py` |
| $(\mathbf{t}_i, \mathbf{a}_i, R_i)$ | ellipsoid 的中心、半軸、旋轉 | 同上 |
| $C_i^{*}$ | $Q_i^{*}$ 投影得到的 dual conic；等價於 2D Gaussian $\mathcal{E}_i=(\boldsymbol{\mu}_i,\Sigma_i)$ | 同上 |
| $\varphi_i \in \{0,1\}$ | floor key（storey 身分，非垂直序） | `mapping/floor_stack.py` |
| $\ell^\star$ | 查詢目標的類別字串 | — |
| $L_i$ | track $i$ 的 presence log-odds；$p_i=\sigma(L_i)$ | `objects/presence.py` |
| $E_i\in\{0,1\}$ | expectation：此 frame「若物體仍在，是否應看見」 | 同上 |
| $Z_i\in\{0,1\}$ | 此 frame 是否偵測到 track $i$ | 同上 |
| $r_i$ | recall $P(Z{=}1\mid \text{present}, \text{view})$ | 同上 |
| $q$ | false-alarm rate $P(Z{=}1\mid \text{absent})$ | 同上 |
| $\rho_i$ | identity rejection 計數 | `agent/candidate.py` |
| $f_j$ | 第 $j$ 個 frontier（cluster） | `mapping/frontier.py` |
| $P_j$ | frontier 的 relevance score | `exploration/selector.py` |
| $d_j$ | 由 planner 給出的 geodesic path cost | `planning/planner.py` |
| $x$ | 一個 search candidate（frontier 或 mapped surface） | `exploration/search_belief.py` |
| $b(x), d(x), c(x)$ | belief、per-visit detection probability、cost | 同上 |
| $\lambda(x)$ | inspection log 的存活因子 | 同上 |

$\sigma(\cdot)$ 為 logistic function。世界座標採 Habitat 慣例（$y$ 向上），
ground plane 為 $(x,z)$，記作 $\Pi$。

---

## 2. 3DSG Construction

場景圖為四層階層 $\text{floor} \to \text{room} \to \text{container} \to \text{object}$
（`graph/scene_graph.py`）。以下依建構順序描述。

### 2.1 Perception frontend

**Keyframe 選取。** 為避免對每個模擬步都執行偵測，僅在相對於上一個 keyframe
的位移 $\|\Delta \mathbf{t}\| > \tau_d$ 或旋轉 $\|\mathrm{rotvec}(\Delta R)\| > \tau_\theta$
時觸發，取 $\tau_d = 0.25\,\mathrm{m}$、$\tau_\theta = 30^\circ$。由於 agent 的
離散動作恰為 $0.25\,\mathrm{m}$ 前進與 $30^\circ$ 轉向，實測 keyframe 數約為步數的
$0.91$–$0.96$ 倍。

**Open-vocabulary 偵測。** 主偵測器為 YOLOE-11L-seg，以固定的 47 類 vocabulary
加上查詢字串進行 text-conditioned 偵測，輸入解析度 $1280$，全域門檻 $s \ge 0.35$。
關鍵在於它輸出**分割遮罩** $M_k$ 而非僅有 bbox：物體點雲由 $M_k$ 經 depth 反投影
取得，若以矩形代替遮罩，物體後方的牆面會被納入同一個 quadric。

**Class-agnostic region proposal。** 為涵蓋偵測器無法命名的物體，在每個 keyframe
額外執行 FastSAM-s 取得至多 64 個 class-agnostic region，經
$[200,\,0.02\,|I|]$ 的面積帶過濾後，以 MobileCLIP-S2 編碼並與查詢字串比對餘弦
相似度，超過 $\tau_{\mathrm{admit}}=0.24$ 者以固定 score $0.5$ 併入 detection 集合。
此類 observation 在 track 上獨立記錄（`proposal_only`），其晉升為 candidate 的門檻
與 named detector track 分開。

### 2.2 物體構成：ellipsoid 表示、關聯與多視角組合

每個 object track 以一個 dual quadric ellipsoid 表示，由中心 $\mathbf{t}$、
半軸 $\mathbf{a}=(a,b,c)$ 與旋轉 $R$ 決定：其對偶二次曲面為
$Q^{*} = Z\,\mathrm{diag}(a^2,b^2,c^2,-1)\,Z^{\top}$，其中
$Z=[\,R\ \ \mathbf{t}\,;\ \mathbf{0}^{\top}\ 1\,]$；經投影矩陣
$P = K\,[R^{cw}\mid \mathbf{t}^{cw}]$ 得對偶圓錐 $C^{*} = P\,Q^{*}P^{\top}$，
正規化後可由其直接讀出二維高斯參數 $\mathcal{E}=(\boldsymbol{\mu},\Sigma)$。
此表示法及其投影推導沿用 VOOM [cite]，本文不再重述。

選擇 ellipsoid 而非點或 bounding box，其效益在於**同一個投影被使用兩次**：
$\mathcal{E}$ 既是本節資料關聯的比對量，也是 §3.1 判定「若物體仍在原處，
此 frame 是否應當看見它」的依據。一個點沒有範圍可供投影；一個軸對齊的 box
在斜視角下無法給出正確的成像輪廓。兩者都無法支撐 §3 的 presence 通道，
而 presence 是本系統處理動態場景的全部基礎。

**初始化。** 新 track 由單一視角建立：取分割遮罩 $M_k$ 的二階矩得二維橢圓，
其中心像素以橢圓內取樣深度的均值反投影為 $\mathbf{t}$，半軸依 $z/f$ 自像素
縮放至公尺，$R$ 對齊當前相機座標軸，半軸裁剪於 $[0.01,\,3.0]\,\mathrm{m}$。
此初始化刻意粗糙——它只需落在下述關聯閘之內即可，其精度由多視角精煉承擔。

**關聯。** 為避免重複建立物體，每個 detection 須先與既有 track 比對。此步驟
沿用 VOOM：將每個已建立的 ellipsoid 投影至當前 keyframe，與 detection 的遮罩
橢圓以 normalized Gaussian-Wasserstein 相似度比對，以 bounding-box 重疊為
空間閘，採貪婪匹配 [cite]。我們在此僅加入一道**類別一致性檢查**，該檢查為
VOOM 所無。

此步驟並不解決大型物體，而我們主張它也不應被要求解決。沙發或桌子在深度可用的
距離內經常超出視野（深度被裁切於 $0.5$–$5.0\,\mathrm{m}$，超出者無從反投影），
因此一個 keyframe 看見它的一端、下一個 keyframe 看見另一端；兩個遮罩橢圓描述的
本就是同一物體的**不同部位**，被判為不相似是正確的。那一幀確實只看見了半張沙發，
強行使其關聯，等同於要求該度量去推論它從未觀測到的部分。我們因此**容許大型物體
碎裂為數個 track，並於下游重聚**——在物體層以 linking（見下），在圖層以隨物體
尺度縮放的 container 合併（§2.4）。此設計的代價可被量測：對照 HM3D 自身的語義
標註，固定半徑的合併規則在 00829 留下 16 個 `bed` 節點，而該場景僅有一張床；
改用尺度感知的合併後收斂至 7 個，其餘為偵測器將 sofa 與 bench 誤標為 bed，
屬於命名問題而非碎裂問題。

**多視角精煉。** 當一個 track 累積 $\ge 3$ 個 observation 後，對其 9 個參數
$\boldsymbol{\theta}=(\mathbf{t},\,\log\mathbf{a},\,\mathrm{rotvec}(R))$
最小化投影橢圓與觀測橢圓之間的 Bures 代理殘差（VOOM 目標函數 [cite]，每個
observation 貢獻 5 維殘差）。僅取最近 10 個 observation 以界定殘差數量，以
`scipy.optimize.least_squares`（trf，$\mathrm{max\_nfev}=50$）求解；問題規模
極小，不需要通用 graph optimizer。

**該目標函數不能直接照搬至此。** 它僅透過**視差**約束深度，而 ObjectNav 的
agent 是正面走向物體、而非繞行拍攝，視角弧因而很窄：一次精煉可使中心沿視線
滑動數公尺，而二維重投影誤差仍然很低。實測顯示此情形**惡化 3D 中心的次數多於
改善**，其誤差尾端達 $+1$ 至 $+5\,\mathrm{m}$。我們因此要求精煉結果
$\hat{\boldsymbol{\theta}}$ 須通過採納條件方予寫回：

$$
\hat{\boldsymbol{\theta}} \ \text{採納} \iff
\underbrace{\hat{\boldsymbol{\theta}} \in \mathbb{R}^{9}\ \text{有限}}_{\text{數值}}
\ \wedge\
\underbrace{\mathcal{C}(\hat{\boldsymbol{\theta}}) \le \tfrac{1}{2}\|r_0\|^2 + 10^{3}}_{\text{退化幾何}}
\ \wedge\
\underbrace{\|\hat{\mathbf{t}} - \mathbf{t}\| \le \Delta_{\max}}_{\text{視差無法支持的位移}},
\qquad \Delta_{\max}=0.5\,\mathrm{m},
$$

其中 $r_0$ 為精煉前的殘差向量。任一條件不過，即保留原 ellipsoid。第三項是其中
最關鍵的：它並非保守設定，而是將一個為繞行軌跡設計的 SLAM 目標函數移植到一條
**不繞行**的軌跡上所必需的約束。

**Linking。** 有些物體單一 ellipsoid 無法涵蓋——L 形沙發，或分成兩段被看見的床。
我們在物體層以 union-find 重聚這些碎片：

$$
i \sim j \iff
\ell_i = \ell_j \ \wedge\ \varphi_i = \varphi_j \ \wedge\
\|\mathbf{c}_i - \mathbf{c}_j\| < d_{\mathrm{link}} \ \wedge\
\big|\,t^{\mathrm{last}}_i - t^{\mathrm{last}}_j\,\big| \le \Delta t_{\max},
$$

$d_{\mathrm{link}} = 1.0\,\mathrm{m}$；令 $\mathcal{L}_i$ 為 $\sim$ 之遞移閉包下
$o_i$ 所屬的連通分量，則導航與場景圖所使用的物體中心為該分量各中心的平均：

$$
\mathbf{c}(o_i) \;=\; \frac{1}{|\mathcal{L}_i|}\sum_{j \in \mathcal{L}_i} \mathbf{c}_j .
$$

第四項條件——兩個 track 須在時間相近的 frame 中被觀測過——是必需的，其理由與
§3 直接相關。Linking 的用途是重聚**一個物體的數個片段**，而片段必然是同時被
看見的；一個物體與它自己的舊影則不是。若無此閘，一個移動了半公尺的馬克杯，
其陳舊 track 與新 track 會落在 $d_{\mathrm{link}}$ 之內而被併合，$\mathbf{c}(o_i)$
於是報在「它曾經在」與「它現在在」的中點——一個沒有馬克杯的位置，而該位置
**沒有任何觀測能夠否證它**。§3 的 presence filter 正是靠著「走過去、發現空的」
來修正地圖；一個既非舊位置也非新位置的中點，會使該機制永遠收不到證據。

### 2.3 Admission gates

一個 track 需同時滿足下列條件才進入場景圖並可被查詢：
觀測數 $n_i \ge$ `min_obs`、最佳偵測分數 $s_i \ge 0.35$、最佳 bbox 面積
$\ge 1200\,\mathrm{px}$、累積證據 $\ge$ `min_evidence`。這些門檻是針對
**detector 輸出**校準的；`proposal_only` 的 track 沒有對應量，改以獨立的
proposal 門檻判定。

### 2.4 空間層

**Costmap。** 每層樓一張 $0.05\,\mathrm{m}$ 解析度的 2D occupancy grid，值域
$\{-1,0,100\}$（unknown / free / occupied）。落在 $[\,y_{\mathrm{floor}}+0.15,\;
y_{\mathrm{floor}}+1.5\,)$ 高度帶內的深度點標記為 occupied；grid 隨探索自動擴張。
另可選擇性維護一張 per-cell 的**最低**觀測高度層 $H$（取最小值而非平均，
以免桌面遮蔽其下的地板），供樓梯偵測使用。

**Floor estimation。** 樓層由兩個訊號估計（`mapping/floors.py`）：
主訊號為 agent 自身站立高度 $y_{\mathrm{agent}} = \text{cam}_y - h_{\mathrm{cam}}$，
次訊號為觀測地面點的高度直方圖峰值（僅能預先登記「看得到但還沒去過」的樓層，
不得推翻主訊號）。新樓層需與既有樓層相距 $\ge 1.8\,\mathrm{m}$ 方可登記；此值
刻意大於 HM3D 中樓梯平台的典型高度（$\le 1.1\,\mathrm{m}$）而小於樓層間距
（$2.5$–$3.4\,\mathrm{m}$），用以排除把平台誤登記為樓層的失敗模式。
Floor id 為**穩定 id**：後發現的樓層取新 id，不重編既有 id，因為 per-floor costmap、
room id、frontier blacklist 皆以此為鍵。

**Room segmentation。** 在每層的 free space 上以形態學侵蝕（`room_erode_iters=6`，
門寬 $2.0\,\mathrm{m}$）分離連通區域，保留 $\ge 60$ cells 者為 room node。
Room id 以 `floor_id * 1e6 + local_id` 編碼，確保跨樓層唯一。

**Container / support 關係。** container 層回答「物體放在哪個表面上」，
由類別閘與幾何閘的**交集**決定：類別須屬於 `CONTAINER_CATEGORIES`
（table、desk、counter、shelf、cabinet、dresser、nightstand、bed、sofa…），
且 ellipsoid 須具備真實的水平頂面與最小面積。僅以類別（或 CLIP 相似度）認定
會讓任何帶有家具標籤的偽陽性升格為永久 anchor，這正是幾何閘要排除的。

### 2.5 階層與序列化

最終圖結構為

$$
\text{floor}(\varphi) \;\to\; \text{room}(\varphi, r) \;\to\; \text{container}(o_c) \;\to\; \text{object}(o_i),
$$

object node 為 object layer 上的輕量 view（圖本身不持有物體狀態）。
object-object 的 *near* 邊在序列化時按需推導。整張圖連同每個 track 的
presence 狀態、樓層高度與連通性可序列化為 schema-v2 snapshot，供跨 episode
重載。

### 2.6 樓梯與跨樓層連通性

上式為**包含關係**樹，而樓梯不被任何節點包含——它**連接**兩個節點。因此樓梯
不是第五種節點，而是 floor 層之上的一組**邊**：

$$
\mathrm{StairEdge} = \big(\varphi_{\mathrm{from}},\ \varphi_{\mathrm{to}},\
\mathbf{x}_{\mathrm{entry}},\ \mathbf{x}_{\mathrm{exit}},\ t,\ n\big),
$$

於 agent 實際完成一次樓層轉換時寫入（`mapping/floor_stack.py`），並隨 snapshot
持久化為 `connectivity`。把樓梯做成節點需要一種違反其餘各層包含語義的節點型別，
且會使「梯段屬於哪個房間」成為一個無解的問題。

邊的**證據**分三處，皆不入圖：

1. **costmap（稠密、per-storey）。** `stair_mask` 與 up/down 命中累積
   （`mapping/stairs.py:StairDetector`）。確認的 cell 寫為 FREE 並豁免於
   §2.4 的障礙物 stamp；此處刻意保守——`docs/history/INVESTIGATION.md` 記錄了每一次
   放寬 costmap 障礙物寫入的嘗試，失去的真實幾何都多於換得的。
2. **flight（由高度層即時導出，不儲存）。** 高度**嚴格介於**本層與次層之間的
   連通分量，保留其高度跨距 $\ge 1.0\,\mathrm{m}$ 者：桌面是高原、跨距為零，
   斜坡面積過大而由 `max_cells` 排除，而連接兩層樓的梯段依定義必然跨越一公尺
   以上。其**最低階即為梯腳**，無論偵測器遮罩認為梯腳在哪——實測在 00821 上
   該遮罩的目標距真實梯腳 2.7 m。另有 **portal**：高度層中距本層
   $1.8$–$4.0\,\mathrm{m}$ 的 patch。兩者問的是不同問題：flight 問「能從哪裡
   *走上去*」，portal 問「從這裡能*看見*哪一層」。
3. **object layer 中的 `stairs` track。** 與其他物體一樣受 §2.3 的 admission
   gate 約束（$n_i \ge 2$、evidence $\ge 1.0$）。

三者的採用順序為 **flight > `stairs` track > portal**，此順序由量測而非偏好決定：
在 100-episode v1 run 上，YOLOE 的 `stairs` 類別僅在 **15% 的多樓層 episode**
中觸發（12 tracks / 100 episodes）——觸發時品質良好（中位分數 $0.60$、中位
$26$ 次觀測），但過於稀疏，不足以作為樓層轉換的閘。因此**幾何主導、語義確認**。
使用此順序的樓層決策見 §4.3。

此分工的效果是：**圖記錄已證實的連通、costmap 記錄可通行性、flight 記錄下一個
目標點**。三者失效的方式不同，因此可以分別被否證——一條 `StairEdge` 不會因為
某次偵測器沉默而消失，一個 flight 也不會因為從未被走過就被當成不存在。

---

## 3. Presence Belief

物體離開後，地圖中的位姿可能過時。每個物體 track 因此維護一個
**presence belief**，表示它仍在原位的證據。關鍵是區分「沒有看見該位置」
與「看清該位置卻找不到物體」（見下圖）。

![可見性判斷與 presence 更新](../figures/presence_belief.svg)

*圖： (a–c) 近處遮擋不給負證據，背景可見且偵測器沉默時降低 belief，
重新偵測則提高 belief。(d) Keyframe detector 與抵達時擇一使用的觀察者送入
同一有界更新。(e) 依設定的可靠度畫出的示意 log-odds 軌跡；VLM 與 detector
fallback 不在同一次抵達中同時計分。*

### 3.1 以可見性閘住證據

令 $L_i$ 為 track 的有界 log-odds，$p_i=\sigma(L_i)$ 為 presence 分數。
$Z_i=1$ 表示 detection 與投影的 track 重疊，且一個 detection 最多只歸給一個
track；$E_i=1$ 表示此 frame 若物體仍在，偵測器應有機會看見它。以觀測條件下
的 recall $r_i$ 和 false-alarm rate $q$ 更新：

$$
\Delta L_i =
\begin{cases}
\log(r_i/q), & Z_i=1 \quad \text{(sighting)},\\[1ex]
\log((1-r_i)/(1-q)), & Z_i=0,\ E_i=1 \quad \text{(informative miss)},\\[1ex]
0, & Z_i=0,\ E_i=0 \quad \text{(uninformative)}.
\end{cases}
$$

**$E_i$ 閘住 miss，不閘住 sighting。** 物體的橢球必須投影到可讀、夠大的影像
區域，且落在偵測器可工作的距離內，miss 才有意義。同一個用於 track
association 的投影也給出預期深度帶。沿中心射線，物體近側邊界為
$z_{\mathrm{near}}=z_c-e_i-\delta$；若投影區域中夠多有效深度樣本**比它近**，
代表有東西擋在前面，故 $E_i=0$。深度**比預期物體遠**則不被遮擋判據排除：
這時可能看見後方背景，偵測器的沉默正是物體被移走的負證據。這個單邊測試
使遮蔽與移除獲得不同的更新。$r_i$ 由成像大小、距離與視角決定；模型擬合和
各閾值留在實作細節中。

### 3.2 有界更新與抵達證據

累積的證據採非對稱截斷：

$$
L_i \leftarrow \operatorname{clip}(L_i+\Delta L_i,\ L_{\min},L_{\max}),
\qquad |L_{\min}|>L_{\max}.
$$

較低的正向上限避免過去多次 sighting 讓可變的世界變得「幾乎確定」；較深的
負向下限能壓低已被推翻的位姿，但重新偵測仍能恢復 track。因為截斷，
$p_i$ 是決策分數，並非完全校準的 posterior probability。

整段 approach 的偵測器沉默，或 VLM 對局部影像的回答，也能以各自的
$(r,q)$ 送入同一更新。在可檢查的視角抵達地圖指定位置而未找到目標時，
抵達本身給一次負向讀數；視角不可讀則沒有 absence 證據。模型呼叫失敗本身
不提供 VLM 讀數，但通過 expectation 的 detector scan 仍可作為抵達讀數。
圖 (d–e) 顯示兩者如何以不同權重更新同一狀態，導航因而能修訂它曾依循的
地圖。

---

## 4. 搜尋先驗與統一選擇

先檢查地圖中的目標 track。低 presence 使過時位姿退出直接候選；其餘已命名
track 按 presence 排序，分數飽和時以累積的 track evidence 破同分。
Presence 不回答**是不是目標**：錯誤目標也可能真的在原位，故多次抵達卻判為
非目標時，另以 identity rejection count 排除該 track。這些候選 gate 在選擇
搜尋目標之前處理。

舊位姿被 presence 證據推翻後，agent 在**檢查已建圖的支撐表面**與
**開拓未知空間**之間選擇。兩者都是由同一 planner 計價、交給同一 drive
state 執行的導航目標。

### 4.1 兩類搜尋候選

Frontier 是當前樓層可達 free space 與 unknown space 的邊界。
其 relevance $P_j$ 結合探索先驗、information gain 與朝向連續性；
在本評估設定下，這些項目都是幾何量，不使用目標類別。已建圖表面 $x$
來自 scene graph 的 container node，得到相對分數

$$
b(x)=a(x)\,
\alpha(\ell^\star,\ell_x)^\kappa\,
\exp\!\left(-\|\mathbf c_x-\mathbf c^{\mathrm{last}}\|/L\right).
$$

本層的選擇式只納入當前樓層的 surface；是否換樓由 §4.3 決定。
$a(x)$ 是**二元的幾何承載條件**：無法承接目標的表面不進入候選。
$\alpha$ 是類別 affinity，並由 $\kappa<1$ 軟化；proximity 從目標最後
被相信的位置量起，而非從 agent 量起，因為行走距離另由代價項承擔。
$b(x)$ 用於排序，並非校準的機率。與 frontier 比較前，surface 分數以
最高值正規化至固定 mass $\bar b(x)$；**之後**才乘上 inspection 衰減，
以免正規化每輪把已被檢查的表面重新抬高。

### 4.2 共同選擇與持續檢查

令 $c(g)$ 為 planner 到候選 $g$ 的 geodesic cost，$d(x)$ 為檢查表面
$x$ 時看見目標的機會，$\lambda(x)$ 為先前檢查後留下的持續 belief。
選擇式為

$$
g^\star=\arg\max_{g\in\mathcal F\cup\mathcal S} U(g),
\qquad
U(g)=
\begin{cases}
\beta P_j/c(f_j), & g=f_j\in\mathcal F,\\[1ex]
\bar b(x)\lambda(x)d(x)/c(x), & g=x\in\mathcal S.
\end{cases}
$$

$\beta$ 表示未知空間相對於合理的已知表面值多少。候選先以便宜的
relevance 排序、截斷，再讓少數倖存者付出真實 planning 代價；
勝者進入同一導航流程。檢查表面後，若未找到目標，只衰減而非刪除：

$$
\lambda(x)\leftarrow\lambda(x)\bigl(1-d(x)\bigr).
$$

此值跨 attempt 保留。遠距離一瞥不必否證整個表面；近距離的負向檢查
則大幅降低其價值。當已建圖表面逐漸失分，可達 frontier 仍可提供恢復搜尋。

### 4.3 跨樓層策略

每個 surface 有穩定的樓層 key，剩餘分數也可用來提出換樓請求。
本設定採每層合格 surface 的**平均**分數：

$$
M_\varphi=
\frac{1}{|\mathcal S_\varphi|}
\sum_{x\in\mathcal S_\varphi}\bar b(x)\lambda(x)d(x),
\qquad
\varphi^\star=\arg\max_{\varphi:|\mathcal S_\varphi|>0} M_\varphi.
$$

平均值避免家具較多的樓層只因候選數量多而勝出。換樓須以 margin
勝過當前樓層；若此層仍有被相信、尚未實際檢查的目標 track，則沿用
§4 開頭的 presence gate 暫停請求。失敗抵達可否證一層；時間閘避免
過晚或頻繁切換。通過後先走高度層萃取的 **flight**，其次使用偵測的
stairs track，最後才使用幾何 portal（§2.6）。完成跨層後才將
StairEdge 寫入 graph。
