# 给 WF-TRON1B (mjlab) 加入「摔倒自恢复」— 设计与落地方案

> 参照 robot_lab Go2W (`main@500399e`) 的自恢复机制，移植到你的
> `Leixinjonaschang/wheeled_legged_mjlab@develop`。
> 结论先行，再给逐文件、可粘贴的改法，最后是风险与验证策略。

---

## 0. TL;DR

**机制可以直接移植，数学约定完全一致**（mjlab 的 `projected_gravity_b` 与 robot_lab 同号：直立 `pg_z≈-1`，倒置 `pg_z≈+1`，`gravity_vec_w=[0,0,-1]`；`reset_root_state_uniform` 也支持 roll/pitch）。

但有一个**本质区别决定成败**:

- **Go2W 是四轮四足**，静态稳定，倒了用 4 条腿（12 DOF）撑地翻身，准静态就能起来 → robot_lab 那套「全姿态 reset + 关终止 + `upward` 奖励 + 上正门控」直接奏效。
- **WF-TRON1B 是两轮双足**（每侧 abad/hip/knee 3 DOF + 1 轮 = 8 DOF），是**动态平衡**的轮式倒立摆。摔倒后可用来翻身的驱动权限少得多，翻正后还要**重新建立动态平衡**。这更接近 Cassie/Digit 的 get-up 难度，不是 Go2W 那种「顺手就起来」。

所以我的建议不是把四要素照抄 100%，而是：**做一个后向兼容的 `recovery` 变体配置**（不动你现有训练），在其中移植四要素，并针对双足加三处特化：① 上正门控（robot_lab 的灵魂，你现在完全没有）；② 部分翻倒 reset（不是 100% 随机姿态）；③ 翻正进度 shaping。分阶段验证。

还有一个**必须修的隐患**:你现在的 `upright` 奖励用的是 `exp(-‖pg_xy‖²/σ²)`,**只看倾斜大小、对翻转对称** —— 机器人**完全倒扣**时 `pg_xy≈0`,该奖励≈1.0(满分)。用它做恢复会把「四脚朝天」当成「站好了」。恢复必须换成**带 z 符号**的项(robot_lab 的 `upward=(1-pg_z)²`)。

---

## 1. robot_lab Go2W 到底怎么做的（四要素 + 一个灵魂）

代码位置：`config/wheeled/unitree_go2w/rough_env_cfg.py` + 共享 `velocity_env_cfg.py` / `mdp/rewards.py`。

| # | 要素 | Go2W 的具体设置 | 源码位置 |
|---|---|---|---|
| 1 | **全姿态 reset** | `randomize_reset_base`: `roll/pitch/yaw ∈ (-3.14, 3.14)`, `z ∈ (0,0.2)`, 六轴速度 `(-0.5,0.5)` | `unitree_go2w/rough_env_cfg.py:109-126` |
| 2 | **摔倒不终止** | `terminations.illegal_contact = None`; `rewards.is_terminated.weight = 0` | 同上 `:136, :223` |
| 3 | **密集上正奖励** | `upward = square(1 - projected_gravity_b[:,2])`,weight `1.0`（B2/B2W 用 3.0） | `mdp/rewards.py:608-613` |
| 4 | **回合中途推扰** | `push_by_setting_velocity`, 每 10~15s, `x/y∈(-0.5,0.5)` | `velocity_env_cfg.py` EventCfg |

**灵魂（robot_lab 报告里没强调，但代码里到处都是）—— 上正门控 (upright gate):**

`mdp/rewards.py` 里**几乎每一个** locomotion 奖励和惩罚末尾都乘了：

```python
reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
```

- 直立 `pg_z≈-1` → `clamp(1,0,0.7)/0.7 = 1.0`（门开，正常拿速度奖励）
- 侧躺 `pg_z≈0` → `0`（门关）
- 倒置 `pg_z≈+1` → `0`（门关）

含义：**摔在地上时，走路奖励和大部分惩罚全部归零，此刻唯一有梯度的信号就是没被门控的 `upward`**。于是策略被逼着「先翻正 → 门打开 → 再拿速度奖励」。这就是它不需要状态机、不需要课程也能学会自恢复的原因。`track_lin_vel_xy_exp`（`rewards.py:34`）、`base_height_l2`（`:643`）、`flat_orientation_l2`（`:686`）等等都带这个门。

**你要移植的其实是「要素 3 + 灵魂」这一对**，要素 1/2/4 只是让机器人有机会摔、且摔了不被打断。

---

## 2. 你的 mjlab 现状逐项对照（阻碍在哪）

文件：`src/wheeled_legged_mjlab/tasks/velocity/config/wf_tron1b/env_cfgs.py`

| robot_lab 要素 | 你现在的状态 | 是否阻碍恢复 |
|---|---|---|
| 全姿态 reset | `reset_base` 只随机 `yaw∈(-π,π)`, `z∈(0.01,0.05)`,**没有 roll/pitch** → 永远正着生成 | ⛔ 需要加 roll/pitch |
| 摔倒不终止 | `fell_over = bad_orientation(limit 65°→85°)` **摔就终止**;`illegal_contact`(非轮 geom 触地)**也终止** | ⛔⛔ 两个都得关 |
| 上正奖励 | `upright = exp(-‖pg_xy‖²/σ²)`,**对翻转对称、倒扣得满分** | ⛔ 需换带 z 符号的项 |
| 上正门控 | **完全没有**。所有奖励/惩罚都无条件生效 | ⛔⛔⛔ 最关键的缺口 |
| 推扰 | `push_robot` 15~15.5s,已含 6 轴(roll/pitch/yaw) | ✅ 已具备,甚至更强 |
| 恢复容忍课程 | 已有 `fell_over_limit_angle`(65°→85°),docstring 写「recovery tolerance」 | ⚠️ 思路对,但只是放宽终止锥,没真正让它恢复 |

另外几个**会和恢复打架的项**(它们隐含「轮子在地上、机身直立」):

- `base_height` weight **-50**, target 0.82。倒地时 height≈0.17,误差²×50 ≈ **每步 -21**,会淹没 `upward`。
- `flat_orientation` -3、`base_ang_vel_xy` -0.15、`stand_still` -2。
- `illegal_ground_contact`(**惩罚** -1.0)、`self_collisions` -0.1:恢复时机身/腿必然触地,这项持续扣分。
- `wheel_distance` / `wheel_air_time_balance` / `soft_landing` / rough 里的 `wheel_*`:全部假设轮子是接触点,倒地时是垃圾信号。

**这些正是 robot_lab 用门控一并解决的**——门关时它们全归零。你没有门控,所以要么加门控,要么在 recovery 变体里手动关掉/调权重。

一个好消息:mjlab 核心 `reset_root_state_uniform` 用 `quat_from_euler_xyz(samples[3,4,5])` 生成朝向增量,**原生支持 roll/pitch**,加范围即可,无需改核心。

---

## 3. 核心差异与风险（think harder 的部分）

### 3.1 两轮双足 ≠ 四轮四足
Go2W 恢复是**准静态翻身**问题;TRON1B 恢复是**「翻身 + 重新起摆平衡」**问题。风险:
- 倒在背/侧面时,两条腿能提供的翻正力矩有限,可能存在**大量姿态学不会翻**(kinematically 困难甚至不可行)。
- 100% 随机姿态 reset 会让**绝大多数算力耗在地上打滚**,把本就难的动态平衡主任务带崩。robot_lab 敢用 100% 是因为四足恢复太容易。

→ 对策:**部分翻倒 reset**(下面 4.1),而不是无脑照抄 `(-3.14,3.14)`。

### 3.2 `upright` 奖励的翻转对称性(必修 bug)
`mdp.upright` 用 `‖pg_xy‖²`,倒扣时 `pg_xy≈0` 给满分。恢复训练必须用**单调区分上下**的量:

- robot_lab: `upward=(1-pg_z)²` → 直立 4、侧躺 1、倒扣 0。**保留这个,天然是恢复梯度**。

### 3.3 稀疏性
对四足 `upward` 够用;对双足,从「四脚朝天」到「第一次翻过来」可能太稀疏。→ 加一个**翻正进度** shaping(`Δ(-pg_z)`,只奖励朝上正方向变化,4.4),把稀疏地形填密。

### 3.4 终止的两难
关掉 `fell_over` 后,机器人无法靠「快速终止」逃避 -50 的 height 惩罚 —— 但如果 height/wheel 惩罚不门控,地上每步狂扣分,`upward(max=4)` 根本压不住,策略可能学出「贴地不动最省」的退化解。**所以门控不是可选项,是恢复能不能学出来的前提**。

---

## 4. 落地改法（后向兼容的 `recovery` 变体，逐文件可粘贴）

设计原则:**不碰你现有的 flat/rough 训练**。仿照 robot_lab「每机器人一个 cfg」的做法,给 `make_*` 加一个 `recovery: bool=False` 开关,串到各 manager,并注册一个新任务 `...-Recovery-...`。

### 4.1 Events — 部分翻倒 reset
`env_cfgs.py::make_events` 增加参数,并在 recovery 时把 `reset_base` 换成「一部分 env 全姿态、其余正常」。最简单的实现:两个 reset 事件按环境切分,或直接用一个更宽的范围 + 提高 z。推荐先用**整体加宽 + 抬高 z**跑通,再上分桶。

```python
def make_events(*, depth: bool = False, recovery: bool = False) -> dict[str, EventTermCfg]:
    events = { ... }  # 原样保留

    if recovery:
        # 全姿态生成:让相当比例的回合从侧躺/仰躺/倒扣开始。
        # z 抬到能容纳「躺平」的高度(基座默认 z≈0.966,躺平时质心离地≈轮半径0.127)。
        events["reset_base"].params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.3),
            "roll":  (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw":   (-math.pi, math.pi),
        }
        events["reset_base"].params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
        }
        # 腿关节也给更大初值,避免总从同一构型翻
        events["reset_leg_joints"].params["position_range"] = (-0.6, 0.8)
    return events
```

> **更稳的做法(强烈建议,4.6 讲原因):部分翻倒。** 只让 ~30% 环境全姿态、70% 正常起,防止主任务被拖垮。mjlab 的 event 是按 `env_ids` 调用的,可以写一个自定义 reset 包一层,对随机选中的子集用大 roll/pitch、其余用原范围。示例(放进你的 `mdp/events.py`):
> ```python
> def reset_root_state_partial_fallen(env, env_ids, pose_range, velocity_range,
>                                     fallen_pose_range, fallen_fraction=0.3,
>                                     asset_cfg=_DEFAULT_ASSET_CFG):
>     from mjlab.envs.mdp.events import reset_root_state_uniform
>     n = len(env_ids)
>     fallen_mask = torch.rand(n, device=env.device) < fallen_fraction
>     up_ids   = env_ids[~fallen_mask]
>     fall_ids = env_ids[fallen_mask]
>     if up_ids.numel():
>         reset_root_state_uniform(env, up_ids, pose_range, velocity_range, asset_cfg)
>     if fall_ids.numel():
>         reset_root_state_uniform(env, fall_ids, fallen_pose_range, velocity_range, asset_cfg)
> ```

### 4.2 Terminations — 关掉「摔倒终止」
```python
def make_terminations(*, rough: bool, recovery: bool = False) -> dict[str, TerminationTermCfg]:
    terminations = {
        "non_finite_physics": TerminationTermCfg(func=mdp.non_finite_physics),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    if not recovery:
        terminations["fell_over"] = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": FELL_OVER_LIMIT_ANGLE_INITIAL},
        )
        terminations["illegal_contact"] = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "illegal_ground_contact"},
        )
    # recovery: 只保留 time_out(20s) 和数值保护;摔了不打断,给它时间翻身
    if rough:
        terminations["out_of_terrain_bounds"] = TerminationTermCfg(
            func=mdp.out_of_terrain_bounds, params={"margin": 1.5}, time_out=True,
        )
    return terminations
```

### 4.3 Curriculum — 去掉 `fell_over_limit_angle`
它引用了 `fell_over` 项,recovery 下该项已删,会报错;去掉即可(rough 的 `terrain_levels` 保留)。
```python
def make_curriculum(*, rough: bool, recovery: bool = False) -> dict[str, CurriculumTermCfg]:
    curriculum = {}
    if not recovery:
        curriculum["fell_over_limit_angle"] = CurriculumTermCfg( ... )  # 原样
    if rough:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp.terrain_levels_vel, params={"command_name": COMMAND_NAME})
    return curriculum
```
> 也可以**反过来玩课程**:recovery 下把 roll/pitch 的 reset 范围从小到大 ramp(先 ±30° 再 ±180°),比一步到位更稳。这是把现有「recovery tolerance」思路做实。

### 4.4 Rewards — 新增 `upward` + 门控 helper + 翻正进度
在 `mdp/rewards.py` 顶部加两个函数:

```python
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def _upright_gate(env, asset_cfg=_DEFAULT_ASSET_CFG, hi: float = 0.7):
    """robot_lab 灵魂:直立→1,侧躺/倒扣→0。用来门控所有 locomotion 奖惩。"""
    asset = env.scene[asset_cfg.name]
    pg_z = asset.data.projected_gravity_b[:, 2]      # 直立≈-1,倒扣≈+1
    return torch.clamp(-pg_z, 0.0, hi) / hi

def upward(env, asset_cfg=_DEFAULT_ASSET_CFG):
    """带 z 符号的上正奖励(robot_lab 版):直立4,侧躺1,倒扣0。不门控。"""
    asset = env.scene[asset_cfg.name]
    return torch.square(1.0 - asset.data.projected_gravity_b[:, 2])

class righting_progress:
    """翻正进度 shaping:只奖励 (-pg_z) 朝上正方向的增量,给稀疏恢复填密。"""
    def __init__(self, cfg, env):
        del cfg
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    def __call__(self, env, asset_cfg=_DEFAULT_ASSET_CFG, max_progress: float = 0.1):
        up = -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]  # [-1,1]
        prog = torch.clamp(up - self._prev, 0.0, max_progress) / max_progress
        prog = prog * self._has_prev.float()
        self._prev = up.detach(); self._has_prev[:] = True
        return prog
    def reset(self, env_ids):
        self._prev[env_ids] = 0.0; self._has_prev[env_ids] = False
```

**门控现有 locomotion 项**。为了后向兼容,给要门控的奖励加一个可选参数,默认关闭(不影响你现有训练):

```python
def track_linear_velocity(env, std, command_name,
                          asset_cfg=_DEFAULT_ASSET_CFG,
                          upright_gate_hi: float | None = None):     # 新增
    ...  # 原逻辑不变
    reward = torch.exp(-lin_vel_error / std**2)
    if upright_gate_hi is not None:                                  # 新增两行
        reward = reward * _upright_gate(env, asset_cfg, upright_gate_hi)
    return reward
```
对以下项做同样的两行改造(和 robot_lab 门控的集合一致):
`track_linear_velocity`、`track_angular_velocity`、`track_heading`、`base_height_l2`、`flat_orientation_l2`、`stand_still`、`wheel_distance`、`wheel_air_time_balance`、`soft_landing`、`self_collision_cost`(尤其 `illegal_ground_contact` 那条)以及 rough 的 `wheel_*` 全家。

然后 `make_rewards(recovery=True)` 里给这些项塞 `"upright_gate_hi": 0.7`,并加入 `upward` / `righting_progress`,同时把和恢复打架的强惩罚**在门控之外再降权**(双保险):

```python
def make_rewards(*, rough: bool, recovery: bool = False):
    rewards = { ...原样... }
    if recovery:
        GATE = 0.7
        for name in ["track_linear_velocity","track_angular_velocity","track_heading",
                     "base_height","flat_orientation","stand_still","wheel_distance",
                     "wheel_air_time_balance","soft_landing","illegal_ground_contact",
                     "self_collisions"]:
            if name in rewards:
                rewards[name].params["upright_gate_hi"] = GATE
        # 恢复驱动力(不门控)
        rewards["upward"] = RewardTermCfg(func=mdp.upward, weight=1.0,
                                          params={"asset_cfg": SceneEntityCfg(ROBOT_ENTITY)})
        rewards["righting_progress"] = RewardTermCfg(func=mdp.righting_progress, weight=1.0,
                                          params={"asset_cfg": SceneEntityCfg(ROBOT_ENTITY),
                                                  "max_progress": 0.1})
        # 门控之外再兜底:height 惩罚太狠,降一档,避免地上打滚被爆扣
        rewards["base_height"].weight = -10.0
        # 现有 upright(xy 对称)在恢复期没意义甚至误导,弱化
        rewards["upright"].weight = 0.2
    return rewards
```

> 注意:`upright`(类)、`variable_posture`(类)、`wheel_air_time_balance`(类)是**有状态的类奖励**,给它们加门控要在 `__call__` 末尾乘 `_upright_gate`,别忘了它们的 `reset()`。`righting_progress` 我按类写了并带 `reset`,和你 `heading_progress` 的风格一致。

### 4.5 Metrics — 恢复成功率 / 耗时
你已有 `MetricsManager`。加两个指标,方便判断到底学没学会(robot_lab 恰恰缺这个):
```python
def recovery_success_rate(env, asset_cfg=_DEFAULT_ASSET_CFG, upright_thresh: float = 0.9):
    up = -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return (up > upright_thresh).float().mean()   # 当前直立比例
```
再配合一个「从倒地到首次直立的步数」计数器(类似 `velocity_direction_deviation` 的有状态写法)统计 time-to-recover。

### 4.6 Play/eval 变体
`apply_play_overrides` 现在会 `pop("push_robot")` 并把 `fell_over` 设到最终角度 —— recovery 下 `fell_over` 已不存在,直接跳过即可。做一个 recovery-play:保留全姿态 reset、不终止,可视化「丢下去→自己爬起来」。

### 4.7 注册新任务
`make_env_cfg` 先把 `recovery` 透传给 `make_events/make_rewards/make_terminations/make_curriculum`,
在 `env_cfgs.py` 末尾加工厂函数:
```python
def wf_tron1b_flat_recovery_env_cfg(play=False):
    return make_env_cfg(rough=False, play=play, recovery=True)
def wf_tron1b_rough_recovery_env_cfg(play=False):
    return make_env_cfg(rough=True, play=play, recovery=True)
```
然后在 `src/wheeled_legged_mjlab/__init__.py` 里(和现有 8 个任务同样的 `register_mjlab_task` 写法)加:
```python
from ...env_cfgs import wf_tron1b_flat_recovery_env_cfg  # 补 import

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-WF-Tron1B-Recovery",
    env_cfg=wf_tron1b_flat_recovery_env_cfg(),
    play_env_cfg=wf_tron1b_flat_recovery_env_cfg(play=True),
    rl_cfg=wf_tron1b_ppo_runner_cfg(),          # 复用现有 PPO runner
    runner_cls=WheeledLeggedVelocityOnPolicyRunner,
)
```
命名沿用你现有 `Mjlab-Velocity-{Flat,Rough}-WF-Tron1B[...]` 约定,加 `-Recovery` 后缀即可。

---

## 5. 推荐的分阶段验证（别一步到位）

1. **Flat + 部分翻倒(30%) + 门控 + upward,先不加 righting_progress。**
   看 `recovery_success_rate` 是否随训练上升、主任务(正常起的 70%)速度跟踪是否没崩。这是最小可行验证。
2. 若翻正学得慢/学不动 → 加 `righting_progress`,或把 fallen_fraction 用课程从 0.1 爬到 0.4,或按 4.3 用 roll/pitch ramp 课程。
3. **Flat 稳了再上 rough。** rough 的 `wheel_*` 奖励务必都门控,否则地形 + 倒地双重噪声。
4. 全程盯两条曲线:`upward`/`righting_progress`(该涨)和 `track_linear_velocity`(主任务不该塌)。二者拉扯时,调 fallen_fraction 和 GATE 的 `hi`。

---

## 6. 备选方案:如果统一策略学不动（双足很可能遇到）

robot_lab 明确**不做**独立 get-up,但那是四足红利。若 TRON1B 统一策略卡住,退而求其次:

- **两阶段/双策略 + 相位切换:** 一个 `getup` 策略专练「任意姿态→直立静止」(奖励只有 upward + 关节安全 + 触地软着陆,无速度命令),一个 locomotion 策略照旧;运行时用 `-pg_z` 阈值切换。训练解耦,各自好收敛。
- **参考轨迹/AMP:** 录一段(或脚本生成)双足起身 motion,用 mimic/AMP 引导起身相 —— 对高 DOF 双足往往比纯 reward-shaping 稳(参考 robot_lab 里 G1 的 BeyondMimic/AMP 思路,只不过目标动作换成 get-up)。
- **课程放宽终止(你已有的路子做实):** 不删 `fell_over`,而是把 `limit_angle` 从 85° 一路 ramp 到 180°,同时逐步引入 fallen reset —— 平滑地把「摔倒终止」过渡到「摔倒恢复」。

---

## 7. 风险与注意事项清单

- **数值/穿模:** 全姿态 + 抬高 z 后,倒扣姿态可能初始穿模。你已有 `clear_non_finite_sim_data` 和 `non_finite_physics` 兜底;若仍报错,调低 fallen z 上限或加一步 settle。
- **`nconmax/njmax`:** 倒地时接触点激增,rough 的 `nconmax=256/njmax=512` 可能不够,留意 MuJoCo contact 溢出警告,必要时调大。
- **门控 `hi=0.7` 的含义:** 沿用 robot_lab,`-pg_z>0.7`(倾斜 < ~45°)才算「基本直立、开门」。可调:想让门更早开就调大 hi。
- **`base_height` 权重:** -50 是为动态平衡调的,恢复期必须门控 + 兜底降权,否则主导一切。
- **别忘了有状态奖励的 `reset()`:** 加门控/新写的类奖励一定实现 `reset(env_ids)`,否则跨回合状态串味。
- **没有现成 checkpoint 能保证学会:** 和 robot_lab 一样,「配置能跑」≠「学得会稳定起身」。双足尤其要靠上面的指标 + 可视化确认,而不是假设。

---

## 附:关键源码坐标（便于你核对）

**robot_lab (`500399e`)**
- `config/wheeled/unitree_go2w/rough_env_cfg.py:109-126`(reset)、`:215`(upward w=1)、`:223`(illegal_contact=None)
- `mdp/rewards.py:22-48`(track+门控)、`:608-613`(upward)、`:643/:686`(base_height/flat 门控)
- `velocity_env_cfg.py` EventCfg(reset_base 默认只 yaw；push 10~15s)、TerminationsCfg(illegal_contact)

**你的 mjlab (`develop`)**
- `config/wf_tron1b/env_cfgs.py`:`make_events:517-579`(reset 只 yaw / push)、`make_rewards:669-924`、`make_terminations:927-947`(fell_over/illegal)、`make_curriculum:950-968`
- `mdp/rewards.py`:`upright` 类 `:383-476`(xy 对称,需换)、`track_linear_velocity:235`、`base_height_l2:528`
- `mdp/terminations.py:illegal_contact:66`、`mdp/curriculums.py:fell_over_limit_angle:96`

**mjlab 核心 (`mjlab@main`,pip 版 1.3.0)**
- `envs/mdp/events.py:reset_root_state_uniform`(支持 roll/pitch)、`push_by_setting_velocity`
- `envs/mdp/terminations.py:bad_orientation` = `acos(-pg_z) > limit_angle`
- `entity/entity.py:809`:`gravity_vec_w=[0,0,-1]`；`entity/data.py:584`:`projected_gravity_b`
