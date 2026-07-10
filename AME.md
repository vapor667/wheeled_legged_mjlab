下面是一份面向你当前项目的复现技术文档：目标不是完整复现整篇论文，而是复现其中 **Attention-Based Map Encoder / AME encoder**，把它作为你项目里的 **teacher encoder / privileged encoder**。这篇论文的核心方法是 *Attention-Based Map Encoding for Learning Generalized Legged Locomotion*，其 encoder 用 proprioception 作为 query，对 robot-centric height map 的 point-wise local features 做 multi-head attention。

# AME Teacher Encoder 复现技术文档

## 1. 复现目标

你的项目是：

```text
teacher: privileged height map / terrain map
student: depth image
task: wheeled-legged / 双轮足机器人 locomotion
```

因此这里建议复现的不是完整 actor-critic，而是论文中的 **privileged terrain encoder**：

```text
robot-centric map scan  +  proprioception / command
          │
          ▼
Attention-Based Map Encoder
          │
          ▼
z_terrain_T / z_priv_T
```

它的作用是：teacher 通过真实高度图或仿真直接可得的地形点云，学习一个对当前机器人状态和运动命令有条件的地形 latent；后续 student depth encoder 再去模仿这个 latent。

论文里的 encoder 结构可以概括为两级：先用 CNN 提取每个地图点附近的局部地形特征，再用 MHA 根据 proprioception 对这些点特征做条件选择。作者明确说这个结构由 low-level CNN 和 MHA 组成，MHA 会 query point-wise local features 并与 proprioceptive observations 结合。

---

## 2. 输入定义

### 2.1 Privileged map scan

论文输入的 exteroception 是 robot-centric map scans，形状为：

```text
map_scan: [B, L, W, 3]
```

其中每个点是机器人坐标系下的 3D 坐标：

```text
(x, y, z)
```

论文写的是 `L × W × 3`，其中 L 是长度方向点数，W 是宽度方向点数，3 是每个点在机器人坐标系下的三维坐标。

对你来说，teacher encoder 不应该直接吃 depth image，而应该吃仿真中 privileged 的局部 height map。推荐：

```text
map_scan_T: [B, L, W, 3]
```

其中：

```text
x: 点在机器人 base frame 下的前后坐标
y: 点在机器人 base frame 下的左右坐标
z: 地形高度，建议相对于 base height 或 nominal ground 做归一化
```

如果你暂时不想改太多代码，可以先用：

```text
L = 26
W = 16
resolution = 0.10 m
```

这是论文中 ANYmal-D 使用的 map scan 尺寸；GR-1 用的是 `17 × 11`。论文说明两种机器人网络参数基本相同，只是 map scan 尺寸不同：ANYmal-D 是 `26 × 16`，GR-1 是 `17 × 11`。

你的双轮足机器人速度更快，而且前向预判更重要，所以我更建议采用 **前向偏置视野**：

```text
x range: [-0.4 m, 2.2 m] 或 [-0.5 m, 2.5 m]
y range: [-0.8 m, 0.8 m]
resolution: 0.10 m
L ≈ 26~31
W ≈ 16
```

如果机器人主要靠轮子连续接触，而不是离散 foothold，height map 的作用就不是“找脚点”，而是“找未来轮路、坡度、台阶边缘、可通行支撑面”。

---

### 2.2 Proprioception / query input

论文中的 policy observation 包括：

```text
base linear velocity
base angular velocity
gravity vector
joint positions
joint velocities
previous action
map scan
```

并且这些量都在 robot-centric base frame 下。

你的 teacher encoder 的 query 输入建议定义为：

```text
proprio_query = [
    base_ang_vel,             # 3
    projected_gravity,        # 3
    joint_pos - default_pos,   # n_dof
    joint_vel,                # n_dof
    previous_action,          # action_dim
    command,                  # vx, vy/yaw_rate/heading
    optional: estimated/base lin vel if teacher has it
]
```

重点：**command 应该进入 query**。否则 attention 只能根据身体状态选地形区域，不能根据“要往哪里走”改变关注区域。论文中的可视化也强调 attention 会随命令方向变化，并且可以在命令不可行时拒绝盲目跟随命令。

---

## 3. Encoder 网络结构

严格按论文复现时，结构如下：

```text
map_scan [B, L, W, 3]
    ├── z height only [B, 1, L, W]
    │       └── CNN: Conv2d(1, 16, k=5, p=2)
    │                Conv2d(16, d-3, k=5, p=2)
    │
    ├── xyz coordinates [B, L, W, 3]
    │
    └── concat → point_features [B, L*W, d]

proprio [B, d_obs]
    └── Linear(d_obs, d) → query [B, 1, d]

MHA:
    Q = proprio embedding
    K = point_features
    V = point_features

output:
    map_encoding [B, 1, d] → z_map [B, d]
```

论文明确写到：只把 z-values 送入 CNN；CNN 两层、zero padding 保持原尺寸、kernel size 为 5；第一层 16 hidden units，第二层 `d - 3` hidden units；之后把 CNN 输出和 3D 坐标拼接，得到 `LW × d` 的 point-wise local features。

MHA 部分使用 proprioception embedding 作为 query，local map features 作为 keys 和 values；query length `n=1`。

论文参数：

```text
d = 64
n = 1
h = 16 heads
```

其中 `d` 是 MHA dimension，`h` 是 attention heads 数量。

因此最小复现版本：

```text
CNN:
    Conv2d(1, 16, kernel_size=5, padding=2)
    activation
    Conv2d(16, 61, kernel_size=5, padding=2)
    activation

concat:
    [cnn_feature_61, xyz_3] -> 64-dim point feature

MHA:
    embed_dim = 64
    num_heads = 16
```

论文没有明确给出 CNN activation、MLP hidden dims、LayerNorm 等细节。因此建议你工程实现中使用：

```text
activation = ELU 或 ReLU
LayerNorm(point_features) 可选
LayerNorm(query) 可选
```

如果目标是“尽量忠实复现”，先不要加复杂 norm；如果训练不稳定，再加 `LayerNorm(d)`。

---

## 4. PyTorch 参考实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMapEncoder(nn.Module):
    """
    AME-style privileged terrain encoder.

    Inputs:
        map_scan: [B, L, W, 3], robot-centric xyz map points
        proprio: [B, obs_dim], proprioception + command

    Output:
        z_map: [B, d]
        attn_weights: [B, num_heads or averaged, 1, L*W] depending on PyTorch version/settings
    """

    def __init__(
        self,
        proprio_dim: int,
        d_model: int = 64,
        num_heads: int = 16,
        activation: str = "elu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        assert d_model > 3
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, d_model - 3, kernel_size=5, padding=2)

        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.point_ln = nn.LayerNorm(d_model)
            self.query_ln = nn.LayerNorm(d_model)
            self.out_ln = nn.LayerNorm(d_model)

        if activation == "elu":
            self.act = nn.ELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, map_scan: torch.Tensor, proprio: torch.Tensor):
        """
        map_scan: [B, L, W, 3]
        proprio: [B, proprio_dim]
        """
        assert map_scan.dim() == 4 and map_scan.shape[-1] == 3

        B, L, W, _ = map_scan.shape

        # xyz coordinates, kept as privileged geometric tokens
        xyz = map_scan.reshape(B, L * W, 3)

        # height-only CNN branch
        z = map_scan[..., 2]              # [B, L, W]
        z = z.unsqueeze(1)                # [B, 1, L, W]

        feat = self.act(self.conv1(z))    # [B, 16, L, W]
        feat = self.act(self.conv2(feat)) # [B, d-3, L, W]

        feat = feat.permute(0, 2, 3, 1).reshape(B, L * W, self.d_model - 3)

        # point-wise local features: [B, L*W, d]
        point_features = torch.cat([feat, xyz], dim=-1)

        # proprioception-conditioned query: [B, 1, d]
        query = self.proprio_proj(proprio).unsqueeze(1)

        if self.use_layer_norm:
            point_features = self.point_ln(point_features)
            query = self.query_ln(query)

        # Q = proprio, K/V = map points
        z_map, attn_weights = self.mha(
            query=query,
            key=point_features,
            value=point_features,
            need_weights=True,
            average_attn_weights=False,
        )

        z_map = z_map.squeeze(1)  # [B, d]

        if self.use_layer_norm:
            z_map = self.out_ln(z_map)

        return z_map, attn_weights
```

如果你的现有 teacher/student latent 是 32 维，可以在后面加 projection：

```python
self.terrain_proj = nn.Sequential(
    nn.Linear(64, 128),
    nn.ELU(),
    nn.Linear(128, 32),
)

z_teacher = self.terrain_proj(z_map)
```

我建议保留 AME 原始 `z_map = 64` 作为内部 terrain encoding，再投影到你现有的 `latent_dim = 32`。这样既保留论文结构，又不破坏你现有 RepTS-LinVel / teacher-student 接口。

---

## 5. Actor-Critic 集成方式

论文里 actor 和 critic 使用相同 encoder 结构，但后面的 MLP 不同：actor MLP 把 MHA 输出映射到 joint actions，critic MLP 映射到 value。

对你的项目，建议分成两种实现模式。

### 模式 A：严格复现论文结构

```text
map_scan + proprio
      │
      ▼
shared AME encoder
      │
      ├── actor MLP → action
      └── critic MLP → value
```

优点：最接近论文。

缺点：critic 的 value loss 会直接塑造 encoder，可能让 latent 更偏 value estimation，不一定最适合作为 student distillation target。

### 模式 B：更适合你的 teacher-student 项目

```text
teacher actor:
    height map + proprio → AME teacher encoder → z_T
    [proprio, z_T, command] → actor → action

critic:
    privileged critic obs → critic encoder / critic MLP → value

student:
    depth image + proprio history → depth encoder → z_S
    z_S distill z_T
```

我更推荐模式 B。原因是你的目标不是只训练一个强 teacher policy，而是得到一个可被 depth student 模仿的 privileged terrain latent。critic 可以使用更丰富 privileged obs，但不要让 critic loss 过度污染 teacher terrain latent。

---

## 6. Teacher latent 设计

建议定义两个输出：

```text
z_map_64      = AME 原始输出
z_teacher_32  = projection(z_map_64)
```

你的 actor 输入可以是：

```text
actor_input_T = concat(
    proprio_history_flat,
    command,
    z_teacher_32
)
```

如果你希望更接近论文，也可以让 actor 输入：

```text
actor_input_T = concat(
    proprio_current,
    command,
    z_map_64
)
```

论文图 8 中，在 MHA 得到 map encoding 后，会把 map encoding 和 proprioception concat，再送入 MLP 输出 action。

对你当前代码路线，我建议：

```text
teacher encoder 输出 z_terrain_T = 32
student depth encoder 输出 z_terrain_S = 32
actor 输入保持和现有 RepTS-LinVel 兼容
```

即：

```text
teacher:
    height_map → AME → z_terrain_T

student:
    depth_image/latest_depth + proprio_history → depth CNN/GRU → z_terrain_S

loss:
    PPO actor-critic loss
    + λ_latent * MSE(z_terrain_S, stopgrad(z_terrain_T))
```

---

## 7. 训练流程

论文的关键不是单独预训练 encoder，而是 **encoder 和 locomotion policy end-to-end 训练**。它使用 two-stage training pipeline：第一阶段在 base terrains 上用 perfect perception 训练，第二阶段加入更复杂地形、disturbance、uncertainty 和 perception noise。

建议你的复现流程如下。

### Stage 0：先接通 teacher encoder

先不要加 student，不要加 depth。

```text
height map privileged obs
    → AME teacher encoder
    → actor
    → PPO locomotion
```

目标是确认 AME encoder 本身能让 teacher policy 学会走多地形。

### Stage 1：perfect privileged teacher training

```text
actor input: clean height map + proprio
critic input: privileged obs
terrain: base terrains
noise: 暂时关闭 map noise / drift
```

论文第一阶段的意义是 warm up map encoding learning，并让 controller 在 ground-truth sensing 下先学到基本 locomotion skill。

对你的双轮足项目，base terrains 建议：

```text
flat
rough
slope
low stairs / low step
gap-like height discontinuity
random boxes / pallets
```

不要一开始就放非常稀疏、非常高差、非常窄的地形，否则 attention 很可能学不起来。

### Stage 2：teacher fine-tuning with noise

```text
actor input: noisy / drifted height map
critic input: clean privileged obs
terrain: base + harder terrains
randomization: friction, mass, push, sensor noise
```

论文第二阶段会给 actor 加 noise/disturbance，但 critic 继续用 privileged information。

论文还提到：非 privileged observation 会加 uniform noise；map scans 有随机 drift；另外还加 push、torso mass randomization 和 contact-foot friction randomization。 

对你的项目，Stage 2 建议加：

```text
height z noise: Uniform(-0.02, 0.02) m
height map xy drift: Normal(0, 0.03~0.08) m
height map yaw drift: Normal(0, 2~5 deg)
friction randomization: 0.4~1.5
mass randomization: ±10%~20%
external push: random base velocity perturbation
depth-student later use: depth dropout / latency / stale frame
```

---

## 8. Student distillation 接入

当 teacher policy 稳定后，再训练 student depth encoder。建议采用：

```text
teacher:
    height_map_T + proprio → AME → z_T

student:
    depth_image + proprio_history → DepthEncoder → z_S

distillation:
    L_z = MSE(normalize(z_S), stopgrad(normalize(z_T)))
```

如果 actor 已经训练好，可以再加 action distillation：

```text
a_T = actor(proprio, z_T)
a_S = actor(proprio, z_S)

L_action = MSE(a_S, stopgrad(a_T))
```

最终 loss：

```text
L = L_PPO_student
  + λ_z * L_z
  + λ_action * L_action
```

推荐初始权重：

```text
λ_z = 0.5 ~ 2.0
λ_action = 0.1 ~ 1.0
```

如果训练早期 student 影响 PPO 太大，可以先：

```text
freeze actor
train student encoder only with L_z + L_action
then unfreeze actor for PPO fine-tuning
```

---

## 9. Attention 可视化与验收指标

AME encoder 的一个好处是可以看 attention map。论文指出 MHA 会根据 proprioception 对 steppable areas 分配更高 attention，并把这些区域作为未来 foothold guidance。

你的项目不一定有“脚点”，但仍然可以检查：

```text
attention 是否集中在：
1. 未来轮子将经过的路径
2. 台阶边缘 / gap 边缘
3. 坡度突变区域
4. 当前 command 方向前方区域
5. 不可通行区域附近的边界
```

可视化方法：

```python
# attn_weights: [B, num_heads, 1, L*W]
attn = attn_weights.mean(dim=1).squeeze(1)  # [B, L*W]
attn_map = attn.reshape(B, L, W)
```

训练是否成功，不要只看 reward。建议同时看：

```text
1. terrain curriculum level 是否稳定上升
2. velocity tracking error
3. fall / termination rate
4. stuck rate
5. attention 是否随 command 改变
6. z_T 与 z_S 的 cosine similarity
7. student 换 depth 后是否明显退化
```

论文的 ablation 也说明 two-stage training 和 proposed MHA structure 都明显影响收敛和泛化；他们比较了 transformer encoder、CNN down-sample、ViT encoder，结果 proposed method 在 terrain level 和 success rate 上更好，尤其在 unseen terrains 上更明显。

---

## 10. PPO 参数参考

论文使用 4096 parallelized environments，PPO 参数包括：

```text
batch size: 24 * 4096 = 98304
mini-batch size: 8 * 4096 = 32768
epochs: 5
clip range: 0.2
entropy coefficient: 0.005 stage 1, 0.002 stage 2
discount factor: 0.99
GAE discount factor: 0.95
desired KL: 0.01
learning rate: adaptive
```

这些来自论文 supplementary 的 PPO 参数表。

你的项目不必完全照搬 entropy coefficient。你之前已经遇到过 std 过大和后期退化问题，所以建议：

```text
entropy_coef:
    stage 1: 1e-3 ~ 5e-3
    stage 2: 5e-4 ~ 2e-3

init_std:
    0.3 ~ 0.5

std_range:
    0.05 ~ 0.8
```

尤其不要因为论文用了 0.005 就直接放开 std 上界。你的 wheeled-legged 机器人对高频随机动作更敏感。

---

## 11. 推荐文件/类设计

建议新增这些文件或类：

```text
rsl_rl/rsl_rl/modules/attention_map_encoder.py
    class AttentionMapEncoder

rsl_rl/rsl_rl/models/depth_representation_actor_critic.py
    class DepthRepresentationVelocityActorCritic

env obs:
    privileged_map_scan: [B, L, W, 3]
    depth_image: [B, 1, H, W]
    proprio_history: [B, history_len, proprio_dim]
```

ActorCritic 内部接口：

```python
def encode_teacher(self, obs_dict):
    map_scan = obs_dict["privileged_map_scan"]
    proprio = obs_dict["proprio_query"]
    z_map, attn = self.teacher_map_encoder(map_scan, proprio)
    z_teacher = self.teacher_proj(z_map)
    return z_teacher, attn


def encode_student(self, obs_dict):
    depth = obs_dict["depth_image"]
    proprio_hist = obs_dict["proprio_history"]
    z_student = self.depth_encoder(depth, proprio_hist)
    return z_student
```

训练时：

```python
# teacher action
z_T, attn = model.encode_teacher(obs)
action_T = model.actor(obs_actor, z_T)

# student action
z_S = model.encode_student(obs)
action_S = model.actor(obs_actor, z_S)

latent_loss = mse(normalize(z_S), normalize(z_T.detach()))
action_loss = mse(action_S, action_T.detach())
```

---

## 12. 你应该保留和修改的部分

**必须保留：**

```text
1. height-only CNN branch
2. xyz coordinate concat
3. proprio/query-conditioned MHA
4. point-wise attention，不要先把 map 全局 flatten 成一个 vector
5. teacher 先用 privileged height map end-to-end 训练
```

**可以修改：**

```text
1. L, W, map range
2. actor MLP hidden dims
3. 是否加 projection 到 32 维
4. 是否加 LayerNorm
5. 是否让 critic 共享 encoder
```

**不建议一开始就修改：**

```text
1. 把 MHA 换成普通 transformer encoder
2. 把 map 用 CNN 直接 downsample 成全局 feature
3. 把 query 去掉，只做 height map encoder
4. teacher encoder 单独用 reconstruction loss 预训练
```

论文的 ablation 恰好说明：相比 transformer encoder、CNN down-sampling 和 ViT，作者的 point-wise MHA map encoding 在训练和泛化上更强。

---

## 13. 最小可行复现版本

你可以按这个版本先做：

```text
Input:
    privileged_map_scan: [B, 26, 16, 3]
    proprio_query: [B, proprio_dim + command_dim]

Encoder:
    Conv2d(1, 16, k=5, p=2)
    ELU
    Conv2d(16, 61, k=5, p=2)
    ELU
    concat xyz → [B, 416, 64]
    Linear(proprio_dim, 64) → query [B, 1, 64]
    MultiheadAttention(embed_dim=64, heads=16)
    output z_map [B, 64]
    Linear/MLP projection → z_teacher [B, 32]

Actor:
    concat(proprio_history_flat, command, z_teacher)
    MLP(512, 256, 128)
    action head

Critic:
    privileged critic obs
    independent critic MLP or shared AME + critic MLP

Training:
    Stage 1: clean privileged map, base terrains
    Stage 2: map noise/drift + harder terrains + randomization
    Student: depth encoder distills z_teacher
```

我的判断：对你现在的项目，AME 最值得借鉴的是 **“proprio-conditioned point-wise terrain attention”**，不是整套 humanoid/quadruped policy。teacher encoder 用 height map 学出 `z_terrain_T`，student depth encoder 模仿它，这是合理且工程上可控的改法。
