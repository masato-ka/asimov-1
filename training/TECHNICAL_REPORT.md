# Asimov-1 歩行ポリシー学習 Technical Report

mjlab を用いた Asimov-1 の二足歩行ポリシー(速度追従)学習パイプラインの構築記録。作業日: 2026-09-21〜22。

- 対象: `sim-model/xmls/asimov_1.xml`(MuJoCo MJCF)
- タスク ID: `Asimov-Velocity-Flat`(平地での速度指令追従)
- 実装場所: `training/`(本ドキュメントと同じディレクトリ)

![Asimov-1 のシミュレーション上での歩行](assets/asimov-1-walking-high.gif)

## 目次

1. [シミュレーション環境の解析結果](#1-シミュレーション環境の解析結果)
2. [歩行ポリシーの学習方針と実験結果](#2-歩行ポリシーの学習方針と実験結果)
3. [作成したスクリプトの利用方法](#3-作成したスクリプトの利用方法)
4. [既知の制限と今後の課題](#4-既知の制限と今後の課題)

---

## 1. シミュレーション環境の解析結果

### 1.1 ソフトウェア環境

| 項目 | 内容 |
|---|---|
| 学習フレームワーク | mjlab 1.6.0(MuJoCo Warp + Isaac Lab 流の manager-based API) |
| 物理エンジン | mujoco 3.11.0 / mujoco-warp 3.11.0(mjlab が `~=3.11.0` に固定) |
| RL ライブラリ | rsl-rl-lib 5.4.2(PPO。mjlab が固定。mjlab が統合しているのは rsl_rl のみ) |
| 深層学習 | torch 2.11.0+cu128 |
| ログ | wandb 0.28.2(`<0.29` に制限。§3.5 参照) |
| Python | 3.12 |
| GPU | NVIDIA RTX 5060 Ti(16 GB, Blackwell / sm_120)。CUDA 12.8 用の torch が必要なため `pytorch-cu128` index を指定 |

### 1.2 MJCF(`sim-model/xmls/asimov_1.xml`)の構造

| 項目 | 内容 |
|---|---|
| ルート | `pelvis_link` に `freejoint`(floating base)。初期位置 z = 0.630 m |
| 関節 | 23 個の hinge。脚 12(左右 × hip pitch/roll/yaw、knee、ankle pitch/roll)、腕 10(左右 × shoulder pitch/roll/yaw、elbow、wrist yaw)、腰 1(waist yaw) |
| ボディ | 25(pelvis + 脚 12 + waist + 首 2 + 腕 10) |
| アクチュエータ | **なし**(`sim-model/README.md` に「学習側の Python で定義する」と明記) |
| keyframe | **なし** |
| センサー | `imu_in_pelvis` サイトに `imu_ang_vel`(gyro)、`imu_lin_vel`(velocimeter)、`imu_lin_acc`、`imu_quat`、`root_angmom`(subtreeangmom)。関節センサー・足裏の力センサーはなし |
| 足の接触 | 各足 4 個の球(半径 5 mm、`foot_capsule` クラス、`condim=3`、`friction=0.6`)+ `left_foot` / `right_foot` サイト |
| その他の衝突 | `collision` クラスは `condim=6`、`priority=1`。隣接ボディ 24 組は `<contact><exclude>` で除外済み |
| 物理設定 | `<option>` は「スタンドアロンビューア用」とコメントされており、学習側で設定する前提 |
| 首 | `neck_yaw_link` / `neck_pitch_link` はメッシュのみで**関節が存在しない**(README の「首 2 DOF」は MJCF に未反映) |
| `passive_upper` クラス | `damping=15, stiffness=80, springref=0`。定義はあるが**どの関節も使っていない** |

センサー名(`imu_lin_vel`、`imu_ang_vel`、`root_angmom`)や足の命名(`{left,right}_foot{1..}_collision`、`left_foot`、`*_ankle_roll_link`)は mjlab 同梱の Unitree G1 設定と同じ規約で、MJCF は mjlab のテンプレートに載せることを想定して作られている。

### 1.3 関節の性質

**armature(反映ロータ慣性)と可動域**(すべて MJCF の値):

| 関節 | armature | 可動域 [rad] |
|---|---|---|
| hip pitch | 0.095625 | -2.09 〜 1.0(右は軸が逆で -1.0 〜 2.09) |
| hip roll | 0.11 | ±0.785 |
| hip yaw | 0.038 | ±0.785 |
| knee | 0.0339552 | 0 〜 1.5(右は -1.5 〜 0) |
| ankle pitch | 0.0565056 | ±0.35 |
| ankle roll | 0.0565056 | ±0.1 |

**トルク・速度上限**(MJCF に無いため `sim-model/urdf/asimov_1.urdf` から取得、左右同値):

| 関節 | effort [Nm] | velocity [rad/s] |
|---|---|---|
| hip pitch | 45 | 12.57 |
| hip roll | 45 | 3.98 |
| hip yaw | 28 | 5.45 |
| knee | 45 | 12.25 |
| ankle pitch | 40 | 9.32 |
| ankle roll | 17 | 9.32 |

**実装時に問題になった MJCF の性質**

1. **左右で軸の向きが逆**: 左 hip pitch / knee / ankle pitch は軸 +y、右は -y。同じ物理的な動きでも関節値の符号が左右で反転する(例: 膝の屈曲は左 +a、右 -a)。正規表現で左右を一括設定すると右膝が可動域外になる。
2. **肘の `ref`**: 左右の肘だけ `ref=±0.785398`(他はすべて 0)。CAD の腕姿勢は qpos = ref に対応する。`passive_upper` の `springref=0` をそのまま使うと復元角が可動域の端になるため、肘は `springref=ref` に設定した。
3. **MJCF 内の地面とライト**: MJCF は独自の `floor` 平面とライトを持つ。mjlab では terrain が地面を提供するため、そのままだと平面が二重になり足の接触が倍になる。`get_spec()` で除去した(MJCF 自体は変更していない)。
4. **`MjsJoint` の `stiffness` / `damping` は 3 要素配列**(この MuJoCo バージョン)。線形項のみ使う。

### 1.4 立位の検証(`training/scripts/check_standing.py`)

- 脚をすべて 0(直立)にしたときの足裏は世界 z ≈ 0.005 m(脚長 ≈ 0.625 m)。ただし膝が可動域の端で特異姿勢のため、学習用の初期姿勢には使えない。
- ゼロ入力で PD だけを効かせても直立は維持できない(約 1.5 秒で転倒)。ankle kp = 65 Nm/rad は質量 35 kg の倒立振子を受動的に支える剛性(概算 mgh ≈ 170 Nm/rad)に届かないため、想定どおり。バランスは方策の役割。
- 初期姿勢は全身重心を足の支持領域の中心に合わせて決めた。上半身の質量が股関節より後ろにあるため、体幹を直立させると重心が足の中心より約 3 cm 後ろになる。グリッド探索(重心が足の中心 ±6 mm、膝 ≥ 0.3 rad、足首角を最小化)で次の姿勢を採用した。

| 項目 | 値(左側。右側は符号反転) |
|---|---|
| hip pitch | -0.25 rad(太ももを前へ) |
| knee | +0.30 rad |
| ankle pitch | -0.17 rad(足裏を水平に保つ) |
| 体幹(pelvis)の前傾 | 0.12 rad(約 7°) |
| pelvis 高さ | 0.62 m(足が接地する高さ 0.6171 m + 数 mm) |
| 肘 | `ref` の値(±0.785) |

### 1.5 Menlo 歩行ガイドとの対応

参考: <https://docs.menlo.ai/guides/locomotion-training>

- 一致: 脚 12 関節、関節命名、armature の値(hip pitch 0.095625、knee 0.0339552、ankle 0.0565056)。よってガイドの数値はこの MJCF に適用できると判断した。
- 相違(今回は**スコープ外**として現状の MJCF のまま扱った):
  - ガイドは足首を **RSU パラレルリンク機構**(2 モーター)、脚を **7 DOF(受動つま先関節あり)** と記述しているが、MJCF は足首 pitch / roll を独立した単純ヒンジとしてモデル化しており、つま先はない。モーター A/B の混合は将来のファームウェア / デプロイ層の課題。
  - ガイドの ankle 効果トルク上限などの数値は URDF の値と一致しない箇所がある。
  - `manual.asimov.inc` の Locomotion / API ページは調査時点で接続できなかった(HTTP 522)。

---

## 2. 歩行ポリシーの学習方針と実験結果

### 2.1 方針(決定事項)

1. **行動空間は脚のみ 12 DOF**。腕・腰・首はポリシーの対象外。
2. 足首・つま先は**現状の MJCF のまま**(RSU リンクや受動つま先を追加しない)。
3. アクチュエータは **mjlab 標準の PD 位置アクチュエータ**(`BuiltinPositionActuatorCfg`)。モーターの詳細モデルは使わない。
4. mjlab 同梱の **velocity タスクをテンプレート**として使い、G1 設定と同様に「生成してから上書き」する(`make_velocity_env_cfg()`)。上書きが困難なら `ManagerBasedRlEnvCfg` を直接組む方針だったが、その必要は生じなかった。

### 2.2 ロボット定義(`training/src/mjlab_asimov/robot/asimov_constants.py`)

- **脚 12 関節**: PD 位置アクチュエータ。kp = 65、kd = 5(Menlo ガイドの system identification の値)、effort 上限は URDF の関節別の値。アクチュエータの指令遅延は 0〜1 物理ステップ(ガイドの `delay_min_lag=0, delay_max_lag=1`)。ソフト可動域は 0.9 倍。
- **腕 10 + 腰 1 = 11 関節**: アクチュエータなし。MuJoCo ネイティブの関節バネ・ダンパ(stiffness 80 / damping 15。`passive_upper` クラスと同値)で保持。ポリシーの行動空間にも観測にも含めない。
- **action scale**: 関節別に `0.30 × effort_limit / stiffness`(Menlo)。hip pitch / roll と knee で 0.208、hip yaw 0.129、ankle pitch 0.185、ankle roll 0.078。
- **衝突**: `CollisionCfg` で足は `condim=3, friction=0.6, priority=1`、それ以外の衝突カプセルは `condim=1, priority=0`(MJCF の `condim=6` は接地を想定しないリンクには過剰なため)。

### 2.3 環境設定(`training/src/mjlab_asimov/tasks/asimov_velocity_env_cfg.py`)

| 項目 | 設定 |
|---|---|
| 物理 / 方策の周期 | 200 Hz / 50 Hz(timestep 0.005 s、decimation 4)。エピソード長 20 s |
| 地形 | 平地(`plane`)。地形スキャン・地形カリキュラムは削除 |
| コマンド | 固定範囲(カリキュラムなし)。vx ∈ [-0.8, 0.8] m/s、vy ∈ [-0.6, 0.6] m/s、wz ∈ [-0.6, 0.6] rad/s。テンプレートの standing 10%・heading 30%・forward-only 20% を維持 |
| 終了条件 | 20 s のタイムアウト、70° 以上の傾き(`fell_over`) |

**観測(非対称 actor-critic)**

| グループ | 内容 | 次元 |
|---|---|---|
| actor | base 角速度(3)、重力投影(3)、速度コマンド(3)、脚関節位置(12)、脚関節速度(12)、前回の行動(12) | 45 |
| critic | actor の項目 + base 線速度(特権)、足の高さ・滞空時間・接地・接地力 | 60 |

- actor に base 線速度は含めない(実機で測定できないため)。Menlo ガイドの actor 入力(45 次元)と一致。
- ノイズは Menlo の値: 角速度 ±0.01、重力投影 ±0.05、関節位置 ±0.01 rad、関節速度 ±0.1 rad/s。
- 関節位置・速度には 0〜1 ステップの観測遅延を入れた。ガイドは CAN のポーリング順に応じた 3 段階(0〜2 / 0〜1 / 0)だが、Asimov-1 の実機構成が不明なため一律 0〜1 のプレースホルダーとした。

**ドメインランダム化**(Menlo の方針: 実機とシミュレーションで実際にばらつく量だけ)

| 項目 | 範囲 |
|---|---|
| エンコーダのオフセット | ±0.02 rad(startup) |
| PD ゲイン(kp, kd) | ×0.9 〜 ×1.1(reset) |
| 足の摩擦 | 1.0 〜 1.5(startup、足のジオメトリのみ) |
| 押し外乱 | 水平 ±0.5 m/s、4〜8 秒ごと |
| アクチュエータ遅延 | 0〜1 物理ステップ |
| **ランダム化しないもの** | 質量、リンク長、重力(ガイドが安定性を損なうと明記)。テンプレートの `base_com` も削除 |

**報酬(最終値)**

| 項目 | 重み | 評価内容 |
|---|---|---|
| `track_linear_velocity` | +2.0 | xy 速度の指令追従(指数カーネル、std = √0.25) |
| `track_angular_velocity` | +2.0 | yaw 角速度の指令追従(std = √0.5) |
| `upright` | +1.0 | 骨盤の姿勢が直立に近いほど加点 |
| `pose` | +1.0 | ホーム姿勢からの偏差。`exp(-mean(err²/std²))`。歩行中の std は下表 |
| `air_time` | +0.5 | 足の滞空時間が 0.05〜0.5 s の範囲にある足の本数(0〜2)を毎ステップ加点。高さは見ない。指令が大きいときのみ |
| `foot_swing_height` | -2.0 | **着地の瞬間だけ**、遊脚中の最高到達高さが目標 0.1 m から外れた量 `(peak/0.1 - 1)²` を罰 |
| `foot_clearance` | -2.0 | 各足の `|足の高さ - 0.1 m| × 足の水平速度` を罰(動いている足のみ) |
| `foot_slip` | -0.1 | 接地中の足の水平速度² |
| `body_ang_vel` | -0.08 | 骨盤の roll / pitch 角速度(Menlo) |
| `angular_momentum` | -0.03 | 全身角運動量(Menlo。MJCF の `root_angmom` を使用) |
| `dof_pos_limits` | -1.0 | 脚関節のソフト可動域超過 |
| `action_rate_l2` | -0.03 | 行動の変化量 |
| `torques` | -5e-5 | 関節トルク² |
| `soft_landing` | -1e-5 | 着地時の衝撃力 |
| `self_collisions` | -1.0 | 自己衝突(力 10 N 以上) |

`pose` の歩行時 std(rad):

| 関節 | 値 | 備考 |
|---|---|---|
| hip pitch | 0.6 | G1 の値 0.3 の 2 倍 |
| knee | 0.7 | G1 の値 0.35 の 2 倍 |
| ankle pitch | 0.5 | G1 の値 0.25 の 2 倍 |
| hip roll / hip yaw | 0.15 | G1 と同じ |
| ankle roll | 0.1 | G1 と同じ |

停止時(`std_standing`)は全関節 0.05。

**PPO(rsl_rl)**: actor / critic とも MLP `512 → 256 → 128`(ELU、観測正規化あり)。学習率 1e-3(adaptive、目標 KL 0.01)、clip 0.2、gamma 0.99、lambda 0.95、エントロピー係数 0.01、5 エポック × 4 ミニバッチ、`num_steps_per_env=24`。1500 イテレーション(RTX 5060 Ti、4096 環境で約 15 分、約 16 万 steps/s)。

### 2.4 実験の経過

**初回学習(ベースライン)**: テンプレートの報酬(G1 相当)を Asimov-1 に合わせて調整した設定で 1500 イテレーション学習した。

- 転倒 0、エピソード長は上限の 1000 ステップに到達。
- 追従は指令の約 80〜88%。停止指令では足踏みせず、移動時は左右がほぼ同頻度(約 1.7 Hz)で接地、両足が同時に浮く時間は 0(跳ねていない)。
- **問題**: 足の持ち上げが小さい(すり足気味)。目視でも、学習ログの `Metrics/peak_height_mean`(約 3.4 cm、目標 10 cm)でも確認できた。

**原因の当たり付け**: `air_time` は無効(重み 0)で、足を空中に置くこと自体への報酬がなかった。持ち上げを促すのは罰則 2 つだけで、特に `foot_swing_height` は着地時のみ・1 回あたり上限 0.25 で、着地は 1 ステップあたり約 0.07 回のため、1 ステップあたり最大でも約 0.02(追従報酬の約 1%)にしかならない。一方、足を上げるコストは大きく、学習ログでは `torques` -0.184、`action_rate_l2` -0.379、`pose` 0.82 / 1.0 と、`foot_clearance`(-0.104)より大きかった。

**実験**: 1 回に変更を積み上げる形で 4 回(各 1500 イテレーション、同一設定・シード 1 つ)。

| 実験 | 変更内容(前の実験に追加) |
|---|---|
| E1 | `foot_swing_height` を -0.25 → -2.0 |
| E2 | `air_time` を 0 → 0.5 |
| E3 | `torques` を -2e-4 → -5e-5、`action_rate_l2` を -0.1 → -0.03(2 つ同時に変更) |
| E4 | `pose` の std を hip pitch / knee / ankle pitch で 2 倍(0.3/0.35/0.25 → 0.6/0.7/0.5) |

### 2.5 評価結果

評価方法(`training/scripts/eval_policy.py`): 固定の速度指令で 64 環境 × 8 秒(最初の 1 秒は除外)。ノイズ・押し外乱・ドメインランダム化なし。遊脚の最高高さは足サイトの地面からの高さ(センサー値)で、着地ごとの平均。

**遊脚の最高高さ [cm]**

| 指令 | ベースライン | E1 | E2 | E3 | **E4** |
|---|---|---|---|---|---|
| 前進 0.4 m/s | 2.9 | 3.5 | 3.5 | 3.5 | **6.8** |
| 前進 0.8 m/s | 5.4 | 6.0 | 6.7 | 6.7 | **10.8** |
| 後退 -0.4 m/s | 3.0 | 4.0 | 3.7 | 3.2 | 5.2 |
| 横移動 0.4 m/s | 2.2 | 3.3 | 3.1 | 2.9 | 5.4 |
| 旋回 0.5 rad/s | 1.8 | 2.8 | 2.4 | 2.4 | 4.6 |

**実測の指令追従**(指令 → 実測。旋回は wz [rad/s]、その他は速度 [m/s])

| 指令 | ベースライン | E1 | E2 | E3 | **E4** |
|---|---|---|---|---|---|
| 前進 0.4 | 0.33 | 0.33 | 0.34 | 0.36 | **0.37** |
| 前進 0.8 | 0.69 | 0.69 | 0.70 | 0.71 | **0.74** |
| 後退 -0.4 | -0.32 | -0.33 | -0.34 | -0.34 | -0.34 |
| 横移動 0.4 | 0.30 | 0.34 | 0.31 | 0.32 | 0.31 |
| 旋回 0.5 | 0.40 | 0.30 | 0.48 | 0.45 | 0.48 |

**両足が接地している時間の割合**(前進 0.4 / 0.8)

| ベースライン | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| 0.45 / 0.28 | 0.52 / 0.31 | 0.26 / 0.15 | 0.19 / 0.11 | 0.16 / 0.07 |

- 全条件・全実験で転倒 0、両足が同時に空中にいる割合 0(跳ねなし)。
- 学習ログの `fell_over`(押し外乱・ノイズあり)は、ベースライン 0.0、E1 0.125、E2 0.042、E3 0.0、E4 0.083(イテレーション最終値。値の正規化は mjlab の定義による)。

### 2.6 考察

- **足が上がらない主因は `pose` 報酬**だった。E1〜E3(遊脚罰則の 8 倍、`air_time` の追加、トルク・行動変化の罰の軽減)では前進 0.4 m/s の高さが 3.5 cm から動かず(前進 0.8 も 6.0〜6.7 cm の範囲)、報酬の重みを変えても改善しなかった。`pose` の許容幅を広げた E4 で初めて 6.8 / 10.8 cm に達した。
- 診断(E3 のポリシー、前進 0.4 m/s): 膝は 0.14〜0.90 rad と大きく曲げるが、股関節 pitch の振れ幅は 0.28 rad と小さく、太ももを前へ振り上げる動きが少なかった。トルクの使用率は最大でも hip roll の約 0.65、hip pitch は約 0.1 で余裕があり、行動値のクリップもないため、トルク・行動範囲は原因ではない。`pose` 報酬(重み 1.0、歩行時は膝 std 0.35 で 0.6 rad 動くと大きく減点)が股関節・膝の動きを抑えていたと解釈できる。
- E1〜E3 の効果は**各 1 回・1 シードの結果で、1 cm 未満の差は有意とは言えない**。ただし `air_time` の追加(E2)は接地時間の割合を大きく下げ、旋回の追従を 0.30 → 0.48 rad/s に改善した。E3 は追従がわずかに改善した。これらを歩容に悪影響がなかったため設定に取り込み、E4 の基礎とした。
- E4 の前進 0.8 m/s で 10.8 cm と、目標 0.1 m をわずかに超えている。高すぎる場合は目標高さの調整が可能。

### 2.7 現在の設定と E1〜E4 の関係

E1〜E4 の設定はすべて `asimov_velocity_env_cfg.py` に取り込み済みで、現在の既定値が E4 である。ベースラインを再現する場合は、CLI で次を上書きし、`pose` の std は同ファイルの `walking_std` を (0.3, 0.15, 0.15, 0.35, 0.25, 0.1) に戻す。

```bash
uv run asimov-train Asimov-Velocity-Flat \
  --env.rewards.foot-swing-height.weight -0.25 \
  --env.rewards.air-time.weight 0.0 \
  --env.rewards.torques.weight -2e-4 \
  --env.rewards.action-rate-l2.weight -0.1
```

---

## 3. 作成したスクリプトの利用方法

すべてリポジトリルートで実行する。

**環境構築(初回のみ。学習を実行する前に必要)**

```bash
# 1. 依存パッケージのインストール
uv sync

# 2. wandb にログイン(学習ログを wandb に送るため、学習前に実行しておく)
uv run wandb login
```

学習ログは既定で wandb に送られるため、`wandb login` は学習を始める前に済ませておく。wandb を使わない場合は、学習時に `--agent.logger tensorboard` を指定すれば `wandb login` は不要(§3.5)。

### 3.1 ファイル構成

```
training/
├── TECHNICAL_REPORT.md                       本ドキュメント(日本語版)
├── TECHNICAL_REPORT_en.md                    英語版
├── src/mjlab_asimov/
│   ├── robot/asimov_constants.py             ロボット定義(アクチュエータ、初期姿勢、衝突)
│   ├── tasks/__init__.py                     タスク登録(Asimov-Velocity-Flat)
│   ├── tasks/asimov_velocity_env_cfg.py      環境設定(観測・報酬・DR・コマンド)
│   ├── tasks/asimov_rl_cfg.py                PPO / ロガー設定
│   └── scripts/
│       ├── train_cli.py                      asimov-train / asimov-play の実体
│       └── play_keyboard.py                  asimov-play-keyboard の実体
├── scripts/
│   ├── check_standing.py                     初期姿勢・モデル構築の確認(CPU)
│   └── eval_policy.py                        固定指令での数値評価
└── tests/                                    静的テスト(GPU 不要、23 件)
```

ルート `pyproject.toml` のコマンド: `asimov-train`、`asimov-play`、`asimov-play-keyboard`。mjlab 標準の `train` / `play` は同梱タスクしか探索しないため、`asimov-*` はタスクを登録してから mjlab の本体を呼ぶラッパーになっている。

### 3.2 学習

```bash
uv run asimov-train Asimov-Velocity-Flat --env.scene.num-envs 4096 --agent.run-name <実験名>
```

| オプション | 説明 |
|---|---|
| `--env.scene.num-envs N` | 並列環境数。4096 で約 16 万 steps/s(RTX 5060 Ti)、1024 で約 8 万 steps/s |
| `--agent.max-iterations N` | イテレーション数(既定 10000)。1500 で約 15 分(RTX 5060 Ti、4096 環境) |
| `--agent.run-name NAME` | run の表示名。ログのディレクトリ名と wandb の run 名に付く |
| `--agent.logger {wandb,tensorboard}` | ログ出力先(既定 wandb) |
| `--env.rewards.<項目>.weight X` | 報酬の重みの上書き(例: `--env.rewards.foot-swing-height.weight -1.0`)。項目名の `_` は `-` |
| `--env.rewards.<項目>.params.<キー> X` | 報酬パラメータの上書き(例: `...foot-swing-height.params.target-height 0.08`) |
| `--agent.upload-model False` | wandb へのチェックポイントのアップロードを止める |
| `--help` | すべてのオプションを表示(報酬・イベント・PPO 設定も CLI から上書き可能) |

出力:

- ログとチェックポイント: `logs/rsl_rl/asimov1_velocity/<日時>_<実験名>/`(`.gitignore` 済み)。`model_<iter>.pt` は 50 イテレーションごとに保存され、保存のたびに ONNX(`*.onnx`)も出力される。
- 進行状況は標準出力に表示される(`Mean reward`、`Mean episode length`、各報酬項、`Metrics/peak_height_mean` など)。

### 3.3 再生

**mjlab 標準のビューア**(指令はランダム。ビューアの操作は mjlab 標準)

```bash
uv run asimov-play Asimov-Velocity-Flat --checkpoint-file logs/rsl_rl/asimov1_velocity/<run>/model_1499.pt --num-envs 1
```

`DISPLAY` があればネイティブの MuJoCo ビューア、なければブラウザ(viser)ビューアで開く(`--viewer native|viser` で指定可能)。ビューアには指令の矢印(緑=目標、青=実測)が表示される。wandb から取得する場合は `--wandb-run-path <entity>/asimov1-locomotion/<run_id>` が使える。

**キーボードで指令を操作**

```bash
uv run asimov-play-keyboard --checkpoint logs/rsl_rl/asimov1_velocity/<run>/model_1499.pt
```

キーは、コマンドを起動した**ターミナルに入力する**(ビューアのウィンドウではない)。

| キー | 動作 |
|---|---|
| ↑ / ↓ | 前進 / 後退(vx ± step) |
| ← / → | 左 / 右へ旋回(wz ± step。← が正) |
| A / E | 左 / 右へ横移動(vy ± step。A が正) |
| SPACE | すべての指令を 0 にクリア |
| Ctrl-C | 終了 |

- 1 回押す(またはオートリピート)ごとに `--step`(既定 0.1)だけ変化する。指令は学習時の範囲(vx ±0.8、vy ±0.6、wz ±0.6)に制限される。現在の指令は端末に `cmd vx=+0.30 vy=+0.00 wz=+0.00` と表示される。
- **キーをビューアで受け取らない理由**: MuJoCo のビューアは A〜Z の 26 文字すべてと SPACE、左右矢印などを表示切替・操作に割り当てている(W=ワイヤーフレーム、X=テクスチャ、S=影、D=静的ボディ など)。ウィンドウ内のキーで操作すると見た目が変わってしまうため、端末から読み取る方式にした。ビューア側のキー(SPACE で一時停止など)は標準のまま使える。
- 標準入力が端末でない場合(パイプ等)はエラーで終了する。

### 3.4 評価・確認

```bash
# 固定指令での数値評価(停止/前進/後退/横移動/旋回、遊脚高さ・歩数・接地割合)
uv run python training/scripts/eval_policy.py --checkpoint logs/rsl_rl/asimov1_velocity/<run>/model_1499.pt

# 初期姿勢・モデルの確認(CPU。基準高さ、重心位置、転倒までの推移)
uv run python training/scripts/check_standing.py

# 静的テスト(ロボット定義、キー入力処理)
uv run pytest training/tests -q
```

### 3.5 wandb

- 事前に `uv run wandb login` でログインしておく(手順は §3 冒頭の環境構築)。既定の送信先は、ログインしたアカウントの **`asimov1-locomotion`** プロジェクト(タグ: `asimov1`, `velocity`, `flat`)。チーム側に送る場合は環境変数 `WANDB_USERNAME` に entity を指定する。
- 送信せずに動作だけ確認したい場合は `WANDB_MODE=offline`(ローカルに保存)、ログを取らない場合は `--agent.logger tensorboard`。
- **wandb のバージョン制限**: mjlab 1.6.0 が固定する `rsl-rl-lib 5.4.2` は `wandb.Settings(start_method="thread")` を呼ぶが、wandb 0.29 以降はこの引数を廃止している(0.28.0 までは受け付ける)。そのためルート `pyproject.toml` で `wandb>=0.22.3,<0.29` に制限している。mjlab / rsl-rl-lib を更新するときは見直すこと。
- 検証はオフラインで実施(プロジェクト名、タグ、指標、チェックポイントの記録を確認)。クラウドへの実送信は未確認。

---

## 4. 既知の制限と今後の課題

**未確認・未測定**

- 歩容の見た目(膝を過度に曲げる不自然な動きがないか)は、E4 について目視評価が済んでいない。
- 押し外乱・ノイズ・ドメインランダム化下での安定性は数値評価していない(評価は外乱なし)。学習ログの `fell_over` は E4 で 0.083 と、E3 の 0.0 より高かった。
- E1〜E4 は各 1 回・1 シードの結果で、小さな差(1 cm 未満)は有意とは言えない。
- キーボード操作は擬似端末で動作を確認したが、ビューアと並べた実際の操作感は確認待ち。

**シミュレーションと実機の差**

- 足首は独立した pitch / roll ヒンジで、実機の RSU パラレルリンク(モーター A/B の混合、バックラッシュ、連成ダイナミクス)や受動つま先を再現していない。デプロイ時に別途マッピング層が必要。
- 腕・腰は受動バネで固定しており、歩行中の腕振りによる角運動量の補償は学習されない。
- 観測遅延は一律 0〜1 ステップのプレースホルダー(Menlo は CAN の順序に応じた 3 段階)。
- MJCF の `noslip_iterations=5`・`impratio=10`・楕円摩擦円錐は mjlab の設定に反映していない(mjlab は `noslip` を持たず、`impratio` / `cone` はテンプレートの既定値のまま)。

**設定のうち出発点として置いた値**

- 報酬・DR の数値は Menlo ガイド(別ロボット / 旧バージョンの値を含む可能性)と、mjlab の G1 設定からの転用が中心。Asimov-1(35 kg / 1.2 m)向けの再チューニングが前提。
- Menlo の値から意図的に変えた点: `torques` -2e-4 → -5e-5、`action_rate` を -0.03 に軽減、`pose` の std を G1 の 2 倍(矢状面の関節のみ)、足の摩擦 DR は Menlo の 1.0〜1.5(mjlab テンプレートは 0.3〜1.2)。
- テンプレートにない機能(その場旋回コマンドを一定割合で強制サンプリングする仕組み)は入れていない。

**今後の候補**

- E4 の目視評価と、外乱下での評価(押し外乱の強さを変えた評価)。
- 目標高さ(現在 0.1 m)の調整。前進 0.8 m/s では 10.8 cm と目標を超えている。
- 複数シードでの再現性確認。
- 首の関節追加や RSU 機構のモデル化(MJCF の拡張)、デプロイ(ONNX エクスポートは学習時に自動出力される)。
