# Visual-CTS / Vision-CTS 复现文档

## 0. 复现目标

本文档用于复现论文 **LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots** 中的视觉版 CTS 框架，即论文所称的 **Vision-based Concurrent Teacher-Student Learning Architecture**。用户可将本文档直接交给 Codex，作为实现 `Visual-CTS / Vision-CTS` 的需求说明。

复现重点不是完整复刻论文所有工程细节，而是实现一个结构忠实、训练闭环完整、可在 Isaac Lab / rsl_rl / PPO 框架中落地的版本。核心包括：

1. teacher-student 并发训练；
2. teacher 使用 privileged information + height map；
3. student 使用 proprioception history + depth image；
4. student 通过两个 estimator 预测 privileged information 和 height map；
5. actor / critic 在 teacher 和 student 之间共享；
6. supervised reconstruction loss 与 PPO loss 联合训练；
7. LIPM-guided stable reward；
8. stable-tracking reward fusion；
9. decoupled velocity tracking reward；
10. double critic：locomotion critic + stable critic；
11. deployment 时只保留 student path。

---

# 1. 总体思想

Vision-CTS 是在原 CTS，也就是 Concurrent Teacher-Student Reinforcement Learning 的基础上加入视觉输入。

原 CTS 的特点是：

* teacher group 和 student group 同时训练；
* 两组 agent 在同一个 PPO 训练过程中并发 rollout；
* teacher 和 student **共享同一个 actor/policy**；
* teacher 和 student **共享 value/critic 网络**；
* teacher 直接使用仿真中的 privileged observation；
* student 只使用部署可获得的 observation；
* student estimator 通过监督学习去模仿 teacher latent 或重建 privileged state；
* 不需要先训练 teacher 再蒸馏 student，而是在一个阶段里同时训练。

这篇论文的改动是：

* teacher 增加 terrain heightmap；
* student 增加 egocentric depth image；
* student 不仅估计 privileged information，还估计 heightmap；
* 使用视觉 estimator 形成 terrain latent；
* 最终 actor 同时依赖 proprioception 和 latent representation；
* 训练中引入稳定性优先的 reward 设计；
* 使用 double critic 分别学习 locomotion reward 和 stable reward。

论文把这个视觉增强版本称为 **Vision-CTS**。用户提到的 **Visual-CTS** 可以理解为同一方法。

---

# 2. 训练时的信息流

## 2.1 两类环境组

训练时将 parallel environments 分成两组：

```text
Teacher Group: 50%
Student Group: 50%
```

论文使用 2048 个并行环境，teacher:student = 1:1。因此可以设置：

```python
num_envs = 2048
num_teacher_envs = 1024
num_student_envs = 1024
```

实现中可以用一个 boolean mask：

```python
is_teacher_env: Tensor[bool]  # shape: [num_envs]
```

或者将 env index 前半部分作为 teacher，后半部分作为 student。

---

## 2.2 Teacher group 的输入

Teacher group 在训练时可以访问：

```text
e_t: privileged information
h_t: surrounding terrain heightmap
o_t: policy 所需的普通 proprioception observation
```

其中 `e_t` 是部署时不可获得、但仿真中可直接获得的信息。根据论文表 I，teacher 端相关信息包括：

```text
Privileged / GT information:
- ground-truth proprioception, dim 27
- base linear velocity, dim 3
- joint torque, dim 6
- joint acceleration, dim 6
- feet contact force, dim 6
- external force, dim 3
```

注意：论文表 I 的排版略有歧义。较稳妥的实现方式是：

```text
o_t = 部署可获得的 proprioceptive observation
e_t = 除 o_t 以外、仿真可获得但部署不可获得的 privileged state
```

如果希望严格贴近论文，则可将 `e_t` 定义为以下拼接：

```python
e_t = concat([
    base_lin_vel_gt,        # [3]
    joint_torque,           # [num_actions = 6]
    joint_acceleration,     # [num_actions = 6]
    feet_contact_force,     # [6] or [num_feet * 3], depending implementation
    external_force,         # [3]
    optional_gt_proprio,    # [27], if following table literally
])
```

Heightmap：

```text
h_t: 441-dimensional height map
```

441 通常对应：

```text
21 x 21 local height grid
```

建议实现为：

```python
heightmap_dim = 441
heightmap_shape = (21, 21)
```

---

## 2.3 Student group 的输入

Student group 在训练和部署时只能使用：

```text
o_t: proprioceptive observation
d_t: egocentric depth image
```

根据论文表 I，student observation 为：

```text
o_t:
- base angular velocity, dim 3
- base orientation, dim 3
- joint positions, dim 6
- joint velocities, dim 6
- velocity command, dim 3
- last action, dim 6
Total: 27
```

因此：

```python
proprio_dim = 27
```

Depth image：

```text
d_t: 80 x 60 depth image
```

建议实现时转成 channel-first：

```python
depth: Tensor  # [B, 1, 80, 60] or [B, 1, 60, 80]
```

具体宽高顺序要在工程中统一。论文只写 `80 x 60`，没有说明 PyTorch tensor layout。

---

# 3. Action space 和低层 PD 控制

论文的 action 是 6 维：

```text
a_t ∈ R^6
```

表示对 nominal joint pose 的 joint position offset。

策略输出：

```python
action = policy(obs)  # [B, 6]
```

低层 target joint position：

```python
q_target = q_nominal + action_scale * action
```

论文没有给出 `action_scale`，需要作为 config 项。建议默认：

```yaml
action_scale: 0.25
```

但需要根据机器人关节限位和已有 baseline 调整。

PD 控制器参数论文给出：

```text
Kp = 40.0
Kd = 2.5
```

控制 torque：

```python
tau_cmd = Kp * (q_target - q) + Kd * (0 - q_dot)
```

如果 Isaac Lab 已经通过 actuator model 处理 PD，可以只输出 joint position target，并在 actuator config 里设置 stiffness/damping。

---

# 4. 网络结构

## 4.1 总体结构

Vision-CTS 中有四类网络模块：

```text
Teacher side:
1. Privileged Encoder
2. Heightmap Encoder

Student side:
3. Privileged Estimator
4. Heightmap Estimator

Shared:
5. Policy / Actor
6. Locomotion Critic
7. Stable Critic
```

Teacher 和 student 的关键区别是 latent 的来源不同：

```text
Teacher latent:
    l_t^t,e = teacher_privileged_encoder(e_t)
    l_t^t,h = teacher_heightmap_encoder(h_t)

Student latent:
    l_t^s,e, e_hat_t = student_privileged_estimator(o_t^H)
    l_t^s,h, h_hat_t = student_heightmap_estimator(o_t^H, d_t)
```

最终根据 env group 选择 latent：

```python
if teacher_env:
    l_t = concat(l_t_t_e, l_t_t_h)
else:
    l_t = concat(l_t_s_e, l_t_s_h)
```

Actor 输入：

```python
actor_input = concat(o_t, l_t)
```

Critic 输入：

```python
critic_input = concat(e_t, h_t, l_t)
```

部署时：

```text
只使用 student path:
o_t history + depth image -> student estimators -> latent -> actor -> action
```

---

# 5. Teacher encoders

## 5.1 Privileged Encoder

输入：

```python
e_t: [B, privileged_dim]
```

输出：

```python
l_t_t_e: [B, priv_latent_dim]
```

论文只说是 encoder，没有给出具体结构。建议默认：

```python
PrivilegedEncoder = MLP(
    input_dim=privileged_dim,
    hidden_dims=[256, 128],
    output_dim=priv_latent_dim,
    activation=ELU
)
```

建议 config：

```yaml
priv_latent_dim: 32
priv_encoder_hidden_dims: [256, 128]
```

---

## 5.2 Heightmap Encoder

输入：

```python
h_t: [B, 441]
```

输出：

```python
l_t_t_h: [B, height_latent_dim]
```

论文只说 heightmap encoder，没有给出结构。可以实现为 MLP：

```python
HeightmapEncoder = MLP(
    input_dim=441,
    hidden_dims=[256, 128],
    output_dim=height_latent_dim,
    activation=ELU
)
```

或者将 heightmap reshape 成 `[B, 1, 21, 21]` 后用小 CNN。为了简洁复现，建议先用 MLP。

建议 config：

```yaml
height_latent_dim: 32
height_encoder_hidden_dims: [256, 128]
```

---

# 6. Student estimators

Student side 有两个 estimator：

```text
1. Privileged Estimator
2. Heightmap Estimator
```

二者都是 encoder-decoder 结构。它们不仅输出显式估计值，还输出 latent representation。

---

## 6.1 Proprioception history

Student estimator 使用 proprioceptive history：

```python
o_t^H = [o_{t-H+1}, ..., o_t]
```

论文没有给 history length。建议设为可配置项：

```yaml
history_len: 10
proprio_dim: 27
```

将 history flatten：

```python
proprio_hist_flat = o_hist.reshape(B, H * proprio_dim)
```

即：

```python
o_hist: [B, H, 27]
proprio_hist_flat: [B, H * 27]
```

---

## 6.2 Student Privileged Estimator

目标：从 proprioception history 中估计 privileged information。

输入：

```python
o_hist: [B, H, 27]
```

Encoder：

```python
l_t_s_e = priv_estimator_encoder(o_hist_flat)
```

Decoder：

```python
e_hat_t = priv_estimator_decoder(l_t_s_e)
```

输出：

```python
l_t_s_e: [B, priv_latent_dim]
e_hat_t: [B, privileged_dim]
```

建议结构：

```python
PrivilegedEstimatorEncoder = MLP(
    input_dim=history_len * proprio_dim,
    hidden_dims=[256, 128],
    output_dim=priv_latent_dim,
    activation=ELU
)

PrivilegedEstimatorDecoder = MLP(
    input_dim=priv_latent_dim,
    hidden_dims=[128, 256],
    output_dim=privileged_dim,
    activation=ELU
)
```

---

## 6.3 Student Heightmap Estimator

目标：从 proprioception history 和当前 depth image 中估计 terrain heightmap。

输入：

```python
o_hist: [B, H, 27]
d_t: [B, 1, 80, 60]
```

论文说：

* MLP 提取 proprioception history feature；
* CNN 提取 depth image feature；
* 拼接二者；
* 经过 GRU 捕捉 temporal dependencies；
* 得到 latent `l_t_s_h`；
* MLP decoder 重建 heightmap `h_hat_t`。

### 6.3.1 Proprio branch

```python
proprio_feature = proprio_mlp(o_hist_flat)
```

建议：

```python
ProprioHistoryEncoder = MLP(
    input_dim=history_len * proprio_dim,
    hidden_dims=[256, 128],
    output_dim=proprio_feature_dim,
    activation=ELU
)
```

默认：

```yaml
proprio_feature_dim: 64
```

### 6.3.2 Depth branch

```python
depth_feature = depth_cnn(d_t)
```

论文没有给 CNN 结构。建议默认：

```python
DepthCNN:
    Conv2d(1, 16, kernel_size=5, stride=2)
    ELU
    Conv2d(16, 32, kernel_size=3, stride=2)
    ELU
    Conv2d(32, 32, kernel_size=3, stride=2)
    ELU
    Flatten
    Linear(flatten_dim, depth_feature_dim)
    ELU
```

默认：

```yaml
depth_feature_dim: 64
```

### 6.3.3 GRU fusion

拼接 proprio feature 和 depth feature：

```python
fusion_feature = concat(proprio_feature, depth_feature)
```

论文说通过 GRU 捕捉 temporal dependencies。这里有两种实现方式：

### 方式 A：GRU 接收 sequence

更符合“temporal dependencies”的说法。

对每个 timestep 的 proprio 和 depth feature 构造 sequence：

```python
fusion_seq: [B, T, fusion_dim]
```

但论文只明确使用当前 depth image `d_t`，没有明确 depth history。因此可以将 depth feature repeat 到 history length，或者只让 GRU 处理 proprio history。这会增加实现复杂度。

### 方式 B：GRUCell 作为 recurrent memory

更适合在线部署。

```python
gru_input = concat(proprio_feature, depth_feature)  # [B, fusion_dim]
h_gru = GRUCell(gru_input, h_prev)
l_t_s_h = Linear(h_gru)
```

部署时维护每个 env 的 GRU hidden state。episode reset 时清零。

建议优先实现方式 B。

默认：

```yaml
fusion_dim: proprio_feature_dim + depth_feature_dim
gru_hidden_dim: 128
height_latent_dim: 32
```

### 6.3.4 Heightmap decoder

```python
h_hat_t = height_decoder(l_t_s_h)
```

输出：

```python
h_hat_t: [B, 441]
```

建议：

```python
HeightmapDecoder = MLP(
    input_dim=height_latent_dim,
    hidden_dims=[128, 256],
    output_dim=441,
    activation=ELU
)
```

---

# 7. Latent alignment 和 reconstruction loss

论文的 student estimator loss 为：

```text
L_rec =
    MSE(l_t^s,e, l_t^t,e)
  + MSE(l_t^s,h, l_t^t,h)
  + MSE(h_hat_t, h_t)
  + MSE(e_hat_t, e_t)
```

实现：

```python
loss_latent_priv = mse(l_s_e, l_t_e.detach())
loss_latent_height = mse(l_s_h, l_t_h.detach())
loss_recon_priv = mse(e_hat, e_t)
loss_recon_height = mse(h_hat, h_t)

loss_rec = (
    loss_latent_priv
    + loss_latent_height
    + loss_recon_priv
    + loss_recon_height
)
```

建议 teacher latent 在 supervised loss 中 detach：

```python
l_t_e.detach()
l_t_h.detach()
```

理由：

* teacher latent 是 student 的模仿目标；
* 如果不 detach，teacher encoder 会被 student reconstruction loss 拉动，可能破坏 PPO 学到的 latent 表征。

但是，teacher encoder 本身仍可接收 PPO gradient，因为 teacher actor path 使用了 teacher latent。

最终总 loss：

```python
loss_total = loss_ppo + rec_coef * loss_rec
```

建议：

```yaml
rec_coef: 1.0
```

如果 reconstruction loss 过大，调低：

```yaml
rec_coef: 0.1
```

---

# 8. Actor / Policy

Actor 是 teacher 和 student 共享的。

输入：

```python
actor_input = concat(o_t, l_t)
```

其中：

```python
o_t: [B, 27]
l_t: [B, priv_latent_dim + height_latent_dim]
```

输出：

```python
action_mean: [B, 6]
```

如果使用 PPO Gaussian policy：

```python
dist = Normal(action_mean, action_std)
action = dist.sample()
```

建议结构：

```python
Actor = MLP(
    input_dim=27 + priv_latent_dim + height_latent_dim,
    hidden_dims=[512, 256, 128],
    output_dim=6,
    activation=ELU
)
```

动作输出建议用 `tanh` 或在环境中 clip：

```python
action = clip(action, -1.0, 1.0)
```

---

# 9. Double Critic

论文使用两个 critic：

```text
1. Locomotion Critic
2. Stable Critic
```

输入都包括：

```python
critic_input = concat(e_t, h_t, l_t)
```

输出：

```python
V_loco_t: [B, 1]
V_stable_t: [B, 1]
```

建议结构：

```python
Critic = MLP(
    input_dim=privileged_dim + 441 + priv_latent_dim + height_latent_dim,
    hidden_dims=[512, 256, 128],
    output_dim=1,
    activation=ELU
)
```

两个 critic 不共享最后输出头。可以实现为：

```python
self.loco_critic = MLP(...)
self.stable_critic = MLP(...)
```

或者共享 trunk + 两个 head：

```python
critic_trunk = MLP(...)
loco_value_head = Linear(...)
stable_value_head = Linear(...)
```

为了简单，建议两个独立 critic。

---

## 9.1 Double critic 的 PPO return 设计

论文说 reward 分成两组：

```text
stability group:
    r_stable

locomotion group:
    r_loco_lin
    r_loco_reg
```

但论文总 reward 又包含：

```text
r = r_stable + r_stable * r_loco_lin + r_loco_reg
```

因此实现时建议分成两个 reward stream：

```python
r_stable_stream = r_stable
r_loco_stream = r_stable * r_loco_lin + r_loco_reg
r_total = r_stable_stream + r_loco_stream
```

然后分别计算 GAE：

```python
adv_stable, ret_stable = compute_gae(
    rewards=r_stable_stream,
    values=V_stable,
    dones=dones,
)

adv_loco, ret_loco = compute_gae(
    rewards=r_loco_stream,
    values=V_loco,
    dones=dones,
)
```

Policy advantage：

```python
adv_total = adv_stable + adv_loco
```

Value loss：

```python
value_loss = mse(V_stable, ret_stable) + mse(V_loco, ret_loco)
```

PPO policy loss：

```python
policy_loss = clipped_ppo_loss(logprob, old_logprob, adv_total)
```

总 PPO loss：

```python
loss_ppo = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

---

# 10. LIPM-guided stable reward

论文使用 LIPM 生成 desired CoM position，并用它构造 stable reward。

## 10.1 Desired CoM position

论文公式：

```text
p_hat_com = p_ZMP + (z / g) * k_p * (v_cmd_xy - v_xy)
```

其中：

```text
p_hat_com: desired CoM position in xy plane
p_ZMP: zero moment point position
z: intercept of centroidal motion constraint plane
g: gravity
k_p: feedback gain
v_cmd_xy: commanded linear velocity in xy plane
v_xy: actual linear velocity in xy plane
```

实现：

```python
p_hat_com_xy = p_zmp_xy + (z / gravity) * kp_lipm * (v_cmd_xy - v_xy)
```

需要配置：

```yaml
kp_lipm: 1.0
gravity: 9.81
```

论文没有说明如何在仿真中计算 `p_ZMP`。可选实现：

### 方式 A：用接触力计算 ZMP

如果可以获得足端接触力和接触点，则可以计算 ZMP。

### 方式 B：用 stance foot position 近似

对 point-foot biped，简单实现可以使用当前接触脚位置：

```python
if left_foot_contact and not right_foot_contact:
    p_zmp_xy = left_foot_pos_xy
elif right_foot_contact and not left_foot_contact:
    p_zmp_xy = right_foot_pos_xy
elif both_contact:
    p_zmp_xy = weighted_average_by_normal_force(left_foot, right_foot)
else:
    p_zmp_xy = previous_p_zmp_xy
```

建议先用方式 B，工程上更稳定。

---

## 10.2 Constraint plane intercept reward

论文要求：

```text
zc should match robot upright standing height
normal vector is unconstrained
```

也就是只约束 constraint plane 的 intercept，不强行约束平面法向量。这样机器人可以根据地形调整身体姿态，而不是被固定 CoM height 限死。

实现中可近似为：

```python
z_error = z_nominal - z_actual
```

其中：

```python
z_nominal = nominal standing CoM height
z_actual = current CoM height relative to local support plane
```

简化实现：

```python
z_actual = com_pos_z - terrain_height_under_com
z_error = z_nominal - z_actual
```

配置：

```yaml
nominal_com_height: <robot-specific>
```

---

## 10.3 Roll / pitch angular velocity penalty

论文希望满足 LIPM 的 zero angular momentum assumption，因此惩罚 roll/pitch angular velocity：

```python
omega_e = [roll_rate, pitch_rate]
```

如果 base angular velocity 是 body frame：

```python
roll_rate = base_ang_vel_b[:, 0]
pitch_rate = base_ang_vel_b[:, 1]
```

---

## 10.4 Stable reward 公式

论文公式：

```text
r_stable = exp(
    - ||p_hat_com - p_com||^2
    - |z_c - z|
    - (|theta_dot_roll| + |theta_dot_pitch|)
)
```

实现：

```python
p_error = p_hat_com_xy - com_pos_xy
z_error = z_nominal - z_actual
omega_error_l1 = abs(roll_rate) + abs(pitch_rate)

r_stable = torch.exp(
    - torch.sum(p_error ** 2, dim=-1)
    - torch.abs(z_error)
    - omega_error_l1
)
```

注意：

```python
r_stable ∈ (0, 1]
```

---

# 11. Stable-Tracking Reward Fusion

论文使用 RFM 思想：稳定性优先于速度跟踪。

基础融合公式：

```text
r_t = r_stable + r_stable * r_vel
```

梯度性质：

```text
∂r / ∂r_stable = 1 + r_vel
∂r / ∂r_vel = r_stable
```

含义：

* stable reward 永远有效；
* velocity reward 只有在 stable reward 足够高时才强；
* 当机器人不稳定时，策略更倾向于先恢复姿态，而不是强行追踪速度。

---

# 12. Decoupled Velocity Tracking Reward

传统速度追踪：

```text
r_vel = exp(-α ||v_cmd - v||^2)
```

问题：

* 当 RFM 把 velocity reward 乘以 `r_stable` 后，如果机器人不稳定，速度追踪梯度会整体变弱；
* 机器人可能不再保持方向，只是停下来；
* 论文因此把线速度追踪拆成 direction tracking 和 magnitude tracking。

---

## 12.1 Direction tracking

使用 cosine similarity：

```text
D(v_cmd_xy, v_xy) =
    dot(v_cmd_xy, v_xy) / (||v_cmd_xy|| * ||v_xy||)
```

方向误差：

```python
direction_error = D - 1.0
```

因为：

```text
D ∈ [-1, 1]
D - 1 ∈ [-2, 0]
```

实现：

```python
eps = 1e-6
cmd_norm = torch.norm(v_cmd_xy, dim=-1)
vel_norm = torch.norm(v_xy, dim=-1)

D = torch.sum(v_cmd_xy * v_xy, dim=-1) / (
    cmd_norm * vel_norm + eps
)

direction_error = D - 1.0
r_dir = torch.exp(4.0 * direction_error)
```

当 command speed 很小时，需要特殊处理：

```python
if ||v_cmd_xy|| < command_threshold:
    direction reward can be disabled or set to 1.0
```

建议：

```yaml
command_threshold: 0.1
```

---

## 12.2 Magnitude tracking

速度大小误差：

```python
mag_error = - (cmd_norm - vel_norm) ** 2
r_mag = torch.exp(4.0 * mag_error)
```

---

## 12.3 论文表 II 的实现形式

论文正文公式和表 II 略有不完全一致。

正文公式倾向于：

```text
r = r_stable + r_stable * r_loco_lin + r_loco_reg
```

这会让 direction 和 magnitude 都被 `r_stable` 调制。

但论文解释说：

```text
stability primarily modulates speed magnitude while maintaining directional control
```

表 II 也显示：

```text
direction tracking weight: 0.5
magnitude tracking weight: 0.5 * r_stable
```

因此更建议按照表 II 和文字解释实现：

```python
r_lin_dir = 0.5 * torch.exp(4.0 * direction_error)
r_lin_mag = 0.5 * r_stable * torch.exp(4.0 * mag_error)

r_loco_lin = r_lin_dir + r_lin_mag
```

这样当机器人不稳定时：

* magnitude tracking 被压低；
* direction tracking 仍然保留；
* 机器人会倾向于朝正确方向慢下来，而不是完全放弃方向。

---

# 13. Locomotion regularization rewards

根据论文表 II，除 linear velocity tracking 外，还有以下 regularization reward。

## 13.1 Yaw angular velocity tracking

```python
r_ang_vel = 0.5 * torch.exp(
    -4.0 * (yaw_rate_cmd - yaw_rate) ** 2
)
```

---

## 13.2 Vertical velocity penalty

```python
r_lin_z = -2.0 * v_z ** 2
```

---

## 13.3 Joint acceleration penalty

```python
r_joint_acc = -2.5e-7 * torch.sum(q_ddot ** 2, dim=-1)
```

---

## 13.4 Joint power penalty

```python
r_joint_power = -2.0e-5 * torch.sum(
    torch.abs(tau) * torch.abs(q_dot),
    dim=-1
)
```

---

## 13.5 Joint torque penalty

```python
r_joint_torque = -1.0e-4 * torch.sum(tau ** 2, dim=-1)
```

---

## 13.6 Action rate penalty

```python
r_action_rate = -0.01 * torch.sum(
    (a_t - a_t_minus_1) ** 2,
    dim=-1
)
```

---

## 13.7 Action smoothness penalty

论文表中写法为：

```text
||a_t - 2a_{t-1} - a_{t-2}||^2
```

但常见 second-order action smoothness 是：

```text
||a_t - 2a_{t-1} + a_{t-2}||^2
```

建议实现时提供 config：

```yaml
action_smoothness_use_standard_second_difference: true
```

默认使用常见形式：

```python
r_action_smoothness = -0.01 * torch.sum(
    (a_t - 2.0 * a_t_minus_1 + a_t_minus_2) ** 2,
    dim=-1
)
```

如果要严格复刻论文表格，则改成：

```python
a_t - 2.0 * a_t_minus_1 - a_t_minus_2
```

---

## 13.8 Collision penalty

```python
r_collision = -1.0 * n_collision
```

---

## 13.9 Joint limit penalty

```python
r_joint_limit = -2.0 * n_limitation
```

---

# 14. 总 reward

建议采用表 II 版本：

```python
r_stable_term = 1.0 * r_stable

r_loco_lin = (
    0.5 * exp(4 * direction_error)
    + 0.5 * r_stable * exp(4 * magnitude_error)
)

r_loco_reg = (
    r_ang_vel
    + r_lin_z
    + r_joint_acc
    + r_joint_power
    + r_joint_torque
    + r_action_rate
    + r_action_smoothness
    + r_collision
    + r_joint_limit
)

r_total = r_stable_term + r_loco_lin + r_loco_reg
```

如果要严格遵循正文公式，则可实现另一个选项：

```python
r_total = r_stable + r_stable * r_loco_lin_raw + r_loco_reg
```

建议在 config 中提供：

```yaml
reward_fusion_mode: "table_ii"  # or "eq10"
```

默认：

```yaml
reward_fusion_mode: "table_ii"
```

---

# 15. Domain Randomization

论文表 III 给出的 domain randomization：

```yaml
domain_randomization:
  payload_mass:
    range: [-1.0, 3.0]
    unit: kg

  center_of_mass_shift:
    x_range_cm: [-3.0, 3.0]
    y_range_cm: [-2.0, 2.0]
    z_range_cm: [-3.0, 3.0]

  friction_coefficient:
    range: [0.4, 1.2]

  restitution_coefficient:
    range: [0.25, 0.75]

  joint_kp_scale:
    range: [0.8, 1.2]

  joint_kd_scale:
    range: [0.8, 1.2]

  motor_strength_scale:
    range: [0.8, 1.2]

  system_delay_ms:
    range: [0.0, 20.0]

  camera_position_noise_mm:
    range: [-10.0, 10.0]

  camera_pitch_noise_deg:
    range: [-1.0, 1.0]

  camera_fov_noise_deg:
    range: [-1.0, 1.0]
```

---

# 16. Training setup

论文训练设置：

```yaml
num_envs: 2048
teacher_student_ratio: 1.0
teacher_envs: 1024
student_envs: 1024
simulator: Isaac Lab
algorithm: PPO
terrain_curriculum: true
training_time_reference: about 12 hours on RTX 4090
```

Terrain curriculum 论文没有给具体参数，只说类似 Rudin et al. 的 terrain curriculum。可以复用 Isaac Lab / legged_gym 常见 curriculum：

```text
If robot succeeds:
    move to harder terrain level
If robot fails or walks too short:
    move to easier terrain level
```

---

# 17. Deployment setup

部署时只保留 student path：

```text
Input:
    o_t history
    current depth image d_t

Forward:
    l_s_e = privileged_estimator_encoder(o_t^H)
    l_s_h = heightmap_estimator_encoder(o_t^H, d_t)
    l_t = concat(l_s_e, l_s_h)
    action = actor(concat(o_t, l_t))

Output:
    6D joint position offset
```

不使用：

```text
e_t
h_t
teacher privileged encoder
teacher heightmap encoder
stable critic
locomotion critic
decoder heads
```

论文部署频率：

```text
policy frequency: 50 Hz
depth processing: 30 Hz asynchronous
camera: Intel RealSense D435i
compute: Intel NUC
```

工程建议：

* policy thread 50 Hz；
* depth thread 30 Hz；
* depth feature 或 latest depth frame 使用 lock-free buffer / mutex buffer；
* 如果某一帧 depth 没更新，policy 使用上一帧 depth；
* GRU hidden state 在机器人启动时清零；
* fall reset 或 episode reset 时清零。

---

# 18. 推荐代码结构

建议 Codex 按以下结构实现：

```text
visual_cts/
    configs/
        visual_cts.yaml

    models/
        mlp.py
        depth_cnn.py
        visual_cts_actor_critic.py
        estimators.py

    rewards/
        lipm_stable_reward.py
        velocity_tracking_reward.py
        regularization_rewards.py

    algorithms/
        ppo_visual_cts.py
        rollout_storage_visual_cts.py

    envs/
        observations.py
        heightmap.py
        depth_preprocess.py
        domain_randomization.py

    deployment/
        export_student_policy.py
        student_policy_runtime.py

    tests/
        test_observation_shapes.py
        test_estimator_shapes.py
        test_reward_shapes.py
        test_teacher_student_switch.py
        test_deployment_no_privileged_input.py
```

---

# 19. 关键 PyTorch 伪代码

## 19.1 Model forward

```python
class VisualCTSActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.teacher_priv_encoder = MLP(
            cfg.privileged_dim,
            cfg.priv_encoder_hidden_dims,
            cfg.priv_latent_dim,
        )

        self.teacher_height_encoder = MLP(
            cfg.heightmap_dim,
            cfg.height_encoder_hidden_dims,
            cfg.height_latent_dim,
        )

        self.student_priv_estimator = PrivilegedEstimator(cfg)
        self.student_height_estimator = HeightmapEstimator(cfg)

        latent_dim = cfg.priv_latent_dim + cfg.height_latent_dim
        actor_input_dim = cfg.proprio_dim + latent_dim
        critic_input_dim = cfg.privileged_dim + cfg.heightmap_dim + latent_dim

        self.actor = MLP(
            actor_input_dim,
            cfg.actor_hidden_dims,
            cfg.num_actions,
        )

        self.stable_critic = MLP(
            critic_input_dim,
            cfg.critic_hidden_dims,
            1,
        )

        self.loco_critic = MLP(
            critic_input_dim,
            cfg.critic_hidden_dims,
            1,
        )

        self.log_std = nn.Parameter(torch.zeros(cfg.num_actions))

    def forward_train(
        self,
        o_t,
        o_hist,
        depth,
        e_t,
        h_t,
        is_teacher,
        gru_state=None,
    ):
        # Teacher latents
        l_t_e = self.teacher_priv_encoder(e_t)
        l_t_h = self.teacher_height_encoder(h_t)

        # Student latents and reconstructions
        l_s_e, e_hat = self.student_priv_estimator(o_hist)
        l_s_h, h_hat, next_gru_state = self.student_height_estimator(
            o_hist,
            depth,
            gru_state,
        )

        # Switch module
        l_e = torch.where(is_teacher[:, None], l_t_e, l_s_e)
        l_h = torch.where(is_teacher[:, None], l_t_h, l_s_h)
        latent = torch.cat([l_e, l_h], dim=-1)

        actor_input = torch.cat([o_t, latent], dim=-1)
        action_mean = self.actor(actor_input)

        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        critic_input = torch.cat([e_t, h_t, latent], dim=-1)
        v_stable = self.stable_critic(critic_input)
        v_loco = self.loco_critic(critic_input)

        aux = {
            "l_t_e": l_t_e,
            "l_t_h": l_t_h,
            "l_s_e": l_s_e,
            "l_s_h": l_s_h,
            "e_hat": e_hat,
            "h_hat": h_hat,
            "next_gru_state": next_gru_state,
        }

        return dist, v_stable, v_loco, aux

    def act_student(self, o_t, o_hist, depth, gru_state=None):
        l_s_e, _ = self.student_priv_estimator(o_hist)
        l_s_h, _, next_gru_state = self.student_height_estimator(
            o_hist,
            depth,
            gru_state,
        )

        latent = torch.cat([l_s_e, l_s_h], dim=-1)
        actor_input = torch.cat([o_t, latent], dim=-1)
        action_mean = self.actor(actor_input)

        return action_mean, next_gru_state
```

---

## 19.2 Reconstruction loss

```python
def compute_reconstruction_loss(aux, e_t, h_t):
    loss_latent_priv = F.mse_loss(
        aux["l_s_e"],
        aux["l_t_e"].detach(),
    )

    loss_latent_height = F.mse_loss(
        aux["l_s_h"],
        aux["l_t_h"].detach(),
    )

    loss_recon_priv = F.mse_loss(
        aux["e_hat"],
        e_t,
    )

    loss_recon_height = F.mse_loss(
        aux["h_hat"],
        h_t,
    )

    return (
        loss_latent_priv
        + loss_latent_height
        + loss_recon_priv
        + loss_recon_height
    )
```

---

## 19.3 Reward computation

```python
def compute_stable_reward(
    com_pos_xy,
    p_zmp_xy,
    v_cmd_xy,
    v_xy,
    z_actual,
    z_nominal,
    roll_rate,
    pitch_rate,
    kp_lipm,
    gravity=9.81,
):
    p_hat_com_xy = p_zmp_xy + (z_actual / gravity) * kp_lipm * (
        v_cmd_xy - v_xy
    )

    p_error = p_hat_com_xy - com_pos_xy
    z_error = z_nominal - z_actual
    omega_error_l1 = torch.abs(roll_rate) + torch.abs(pitch_rate)

    r_stable = torch.exp(
        - torch.sum(p_error ** 2, dim=-1)
        - torch.abs(z_error)
        - omega_error_l1
    )

    return r_stable
```

```python
def compute_decoupled_velocity_reward(
    v_cmd_xy,
    v_xy,
    r_stable,
    eps=1e-6,
):
    cmd_norm = torch.norm(v_cmd_xy, dim=-1)
    vel_norm = torch.norm(v_xy, dim=-1)

    cosine = torch.sum(v_cmd_xy * v_xy, dim=-1) / (
        cmd_norm * vel_norm + eps
    )

    direction_error = cosine - 1.0
    magnitude_error = - (cmd_norm - vel_norm) ** 2

    r_dir = 0.5 * torch.exp(4.0 * direction_error)
    r_mag = 0.5 * r_stable * torch.exp(4.0 * magnitude_error)

    r_lin = r_dir + r_mag

    return r_lin, {
        "r_dir": r_dir,
        "r_mag": r_mag,
        "direction_error": direction_error,
        "magnitude_error": magnitude_error,
    }
```

---

# 20. PPO training loop

```python
for iteration in range(num_learning_iterations):

    for step in range(num_steps_per_env):

        obs = env.get_observations()

        o_t = obs["proprio"]             # [B, 27]
        o_hist = obs["proprio_history"]  # [B, H, 27]
        depth = obs["depth"]             # [B, 1, 80, 60]
        e_t = obs["privileged"]          # [B, privileged_dim]
        h_t = obs["heightmap"]           # [B, 441]
        is_teacher = obs["is_teacher"]   # [B]

        dist, v_stable, v_loco, aux = model.forward_train(
            o_t=o_t,
            o_hist=o_hist,
            depth=depth,
            e_t=e_t,
            h_t=h_t,
            is_teacher=is_teacher,
            gru_state=gru_state,
        )

        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)

        next_obs, _, dones, info = env.step(action)

        rewards = compute_all_rewards(info, action)

        storage.add(
            obs=obs,
            action=action,
            logprob=logprob,
            v_stable=v_stable,
            v_loco=v_loco,
            r_stable_stream=rewards["r_stable_stream"],
            r_loco_stream=rewards["r_loco_stream"],
            r_total=rewards["r_total"],
            dones=dones,
            aux=aux,
        )

    storage.compute_returns_and_advantages(
        last_v_stable,
        last_v_loco,
    )

    for batch in storage.mini_batches():

        dist, v_stable, v_loco, aux = model.forward_train(...)

        policy_loss = compute_ppo_policy_loss(
            dist=dist,
            actions=batch.actions,
            old_logprobs=batch.old_logprobs,
            advantages=batch.adv_total,
        )

        value_loss = (
            F.mse_loss(v_stable, batch.ret_stable)
            + F.mse_loss(v_loco, batch.ret_loco)
        )

        entropy_loss = dist.entropy().sum(dim=-1).mean()

        rec_loss = compute_reconstruction_loss(
            aux,
            e_t=batch.e_t,
            h_t=batch.h_t,
        )

        loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_loss
            + rec_coef * rec_loss
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
```

---

# 21. Config 草案

```yaml
visual_cts:
  num_envs: 2048
  teacher_student_ratio: 1.0

  proprio_dim: 27
  num_actions: 6

  depth:
    width: 80
    height: 60
    channels: 1
    normalize: true
    clip_min: 0.0
    clip_max: 5.0

  heightmap:
    dim: 441
    shape: [21, 21]

  history_len: 10

  latent:
    priv_latent_dim: 32
    height_latent_dim: 32

  model:
    activation: elu

    priv_encoder_hidden_dims: [256, 128]
    height_encoder_hidden_dims: [256, 128]

    priv_estimator_encoder_hidden_dims: [256, 128]
    priv_estimator_decoder_hidden_dims: [128, 256]

    proprio_feature_dim: 64
    depth_feature_dim: 64
    gru_hidden_dim: 128

    actor_hidden_dims: [512, 256, 128]
    critic_hidden_dims: [512, 256, 128]

  action:
    action_scale: 0.25
    clip: 1.0
    pd_kp: 40.0
    pd_kd: 2.5

  reward:
    reward_fusion_mode: table_ii
    kp_lipm: 1.0
    nominal_com_height: null
    command_threshold: 0.1

    stable_weight: 1.0
    lin_direction_weight: 0.5
    lin_magnitude_weight: 0.5
    ang_vel_weight: 0.5
    lin_z_weight: -2.0
    joint_acc_weight: -2.5e-7
    joint_power_weight: -2.0e-5
    joint_torque_weight: -1.0e-4
    action_rate_weight: -0.01
    action_smoothness_weight: -0.01
    collision_weight: -1.0
    joint_limit_weight: -2.0

  ppo:
    rec_coef: 1.0
    value_coef: 1.0
    entropy_coef: 0.01
    max_grad_norm: 1.0
```

---

# 22. 必须通过的测试

## 22.1 Shape test

检查：

```text
o_t: [B, 27]
o_hist: [B, H, 27]
depth: [B, 1, 80, 60]
h_t: [B, 441]
e_t: [B, privileged_dim]
action: [B, 6]
V_stable: [B, 1]
V_loco: [B, 1]
e_hat: [B, privileged_dim]
h_hat: [B, 441]
```

---

## 22.2 Teacher-student switch test

构造：

```python
is_teacher = [True, False]
```

确认：

```text
teacher env 使用 teacher latent；
student env 使用 student latent；
actor 参数共享；
critic 参数共享；
```

---

## 22.3 Deployment no-privileged-input test

`act_student()` 不允许输入：

```text
e_t
h_t
teacher latent
critic input
```

只允许：

```text
o_t
o_hist
depth
gru_state
```

---

## 22.4 Reconstruction loss test

确认：

```text
loss_rec 包含 4 项：
1. student privileged latent vs teacher privileged latent
2. student height latent vs teacher height latent
3. predicted heightmap vs ground-truth heightmap
4. predicted privileged info vs ground-truth privileged info
```

并确认 teacher latent 在 supervised loss 中 detach。

---

## 22.5 Reward sanity test

当 robot 更稳定时：

```text
CoM error ↓
z error ↓
roll/pitch angular velocity ↓
=> r_stable ↑
```

当 direction 更准确时：

```text
cosine similarity ↑
=> r_dir ↑
```

当 speed magnitude 更准确时：

```text
|speed_cmd - speed_actual| ↓
=> r_mag ↑
```

当 `r_stable` 很低时：

```text
magnitude tracking reward 被压低；
direction tracking reward 仍保留；
```

---

# 23. 论文未明确、需要工程决策的细节

以下内容论文没有给出精确实现，Codex 不应假装知道：

1. MLP 层宽；
2. CNN kernel / stride；
3. GRU hidden size；
4. proprioception history length `H`；
5. latent dimension；
6. PPO learning rate、num steps、mini batch、epoch；
7. heightmap 的实际物理范围和采样间隔；
8. depth image 的 normalization / clipping / invalid pixel 处理；
9. ZMP 的精确计算方式；
10. constraint plane intercept `z` 的工程计算方式；
11. action scale；
12. terrain curriculum 参数；
13. teacher/student env mask 的具体调度策略；
14. supervised loss 和 PPO loss 的权重比例；
15. double critic 中 advantage 的精确融合方式。

建议所有这些都写进 config，不要硬编码。

---

# 24. 最小可复现版本建议

如果先做 MVP，不建议一开始完全追求论文级复杂度。推荐顺序：

## Stage 1：Blind CTS baseline

先实现：

```text
teacher privileged encoder
student privileged estimator
shared actor
shared critic
reconstruction loss
```

不加 depth，不加 heightmap，不加 double critic。

目标：确认 CTS 并发训练能跑通。

---

## Stage 2：加入 heightmap teacher 和 student depth estimator

增加：

```text
teacher heightmap encoder
student heightmap estimator
depth CNN
GRU memory
heightmap reconstruction loss
```

目标：确认 Vision-CTS 数据流能跑通。

---

## Stage 3：加入 LIPM stable reward

增加：

```text
p_hat_com
z_error
roll/pitch angular velocity penalty
r_stable
```

目标：确认稳定性 reward 对姿态有正向效果。

---

## Stage 4：加入 RFM + decoupled velocity tracking

增加：

```text
direction tracking
magnitude tracking
r_stable-gated magnitude reward
```

目标：在坡、楼梯、rough terrain 上减少摔倒。

---

## Stage 5：加入 double critic

增加：

```text
stable critic
locomotion critic
separate returns
adv_total = adv_stable + adv_loco
```

目标：提高 stable reward 学习效率，减少 single critic 中 reward conflict。

---

# 25. Codex 实现指令摘要

请 Codex 实现一个 `VisualCTSActorCritic`，满足：

1. teacher path:

   * `e_t -> teacher_priv_encoder -> l_t_e`
   * `h_t -> teacher_height_encoder -> l_t_h`

2. student path:

   * `o_hist -> student_priv_estimator -> l_s_e, e_hat`
   * `o_hist + depth -> student_height_estimator -> l_s_h, h_hat`

3. switch module:

   * teacher env 用 `concat(l_t_e, l_t_h)`
   * student env 用 `concat(l_s_e, l_s_h)`

4. actor:

   * input `concat(o_t, latent)`
   * output 6D joint position offset

5. critics:

   * input `concat(e_t, h_t, latent)`
   * output `V_stable`, `V_loco`

6. losses:

   * PPO clipped policy loss
   * stable critic value loss
   * locomotion critic value loss
   * entropy loss
   * reconstruction loss:

     * `MSE(l_s_e, detach(l_t_e))`
     * `MSE(l_s_h, detach(l_t_h))`
     * `MSE(e_hat, e_t)`
     * `MSE(h_hat, h_t)`

7. deployment:

   * export only student estimators + actor
   * no privileged information
   * no heightmap input
   * no critic
   * no decoder required unless debugging

8. reward:

   * implement LIPM stable reward
   * implement decoupled velocity tracking
   * implement stable-gated magnitude tracking
   * implement regularization terms
   * provide config switch for `table_ii` vs `eq10` reward fusion

9. tests:

   * shape tests
   * teacher/student switch test
   * deployment no-privileged-input test
   * reconstruction loss test
   * reward sanity test

---

# 26. 一句话总结

Vision-CTS 的本质是：**训练时让 teacher 用 privileged state 和 heightmap 给共享 policy 提供高质量 latent，同时让 student 从 proprioception history 和 depth image 学会生成相同 latent；PPO 和 supervised reconstruction 在同一阶段并发优化，最终部署时只保留 student estimator + shared actor。**
