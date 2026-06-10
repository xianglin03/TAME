# TabSyn + DP-SGD 设计方案

## 目标

在 TabSyn 的 diffusion 训练阶段引入 DP-SGD，保持 VAE 阶段不变。训练完成后复用原有的 `tabsyn/sample.py` 进行采样。

## 设计原则

1. **最小改动**：只新增 2 个文件，小改 3 个已有文件
2. **配置极简**：仅暴露 4 个 DP 相关 CLI 参数
3. **训练快速**：固定 3000 steps，无 early stopping/EMA，物理 batch 拆分降低内存压力
4. **采样兼容**：复用 `tabsyn/sample.py`，通过 `--sgd` 标志切换 checkpoint 路径

## 架构

```
+---------------------------------------------------+
|  main.py --method tabsyn --mode train_sgd         |
+---------------------------------------------------+
                     |
                     v
+---------------------------------------------------+
|  tabsyn/main_train_sgd.py  (新增)                  |
|  - 加载 VAE embeddings (train_z.npy)               |
|  - 归一化: (train_z - mean) / 2                    |
|  - 创建 DataLoader (batch=4096)                    |
|  - 创建 MLPDiffusion + Model                       |
|  - 调用 SimpleDPSGDTrainer                         |
+---------------------------------------------------+
                     |
                     v
+---------------------------------------------------+
|  tabsyn/train_sgd.py  (新增)                       |
|  SimpleDPSGDTrainer                                |
|  - AdamW optimizer                                 |
|  - Opacus PrivacyEngine (hooks mode)               |
|  - BatchMemoryManager (物理 batch=1024)             |
|  - 固定 3000 steps，保存最佳 loss 的模型             |
+---------------------------------------------------+
                     |
                     v
+---------------------------------------------------+
|  Checkpoint: tabsyn/ckpt_sgd/{dataname}/model.pt   |
+---------------------------------------------------+

+---------------------------------------------------+
|  main.py --method tabsyn --mode sample --sgd       |
+---------------------------------------------------+
                     |
                     v
+---------------------------------------------------+
|  tabsyn/sample.py  (修改)                          |
|  - get_input_generate 根据 args.sgd 选择 ckpt 路径 |
|  - load tabsyn/ckpt_sgd/{dataname}/model.pt        |
|  - 复用原有采样逻辑                                 |
+---------------------------------------------------+
```

## 组件设计

### 1. `tabsyn/train_sgd.py` — SimpleDPSGDTrainer

```python
class SimpleDPSGDTrainer:
    def __init__(self, model, train_loader, model_save_path, device, args):
        # 从 args 读取 4 个 DP 参数 + steps + physical_batch_size
        # 创建 AdamW optimizer
        # Opacus make_private(module=model, optimizer=optimizer, data_loader=train_loader)

    def run_loop(self):
        # 固定步数循环 (0 .. dp_steps-1)
        # BatchMemoryManager 拆分物理 batch
        # 每步: model(inputs) -> loss.backward() -> optimizer.step()
        # 每 log_every 步: 记录 loss，保存最佳模型
        # 最终: 保存 epsilon 到 dp_metrics.json
```

**关键设计决策：**
- 仅支持 **hooks 模式**（Opacus 默认，最稳定）
- 无 EMA、无 early stopping、无 adaptive LR schedule
- 保存最佳 loss 的 checkpoint（不保存中间 checkpoint）
- `loss_reduction="mean"`，和 `Model.forward` 的 `loss.mean(-1).mean()` 一致

**Opacus 兼容性验证：**
- `Model.forward` 调用 `EDMLoss.__call__` → `Precond.forward` → `MLPDiffusion.forward`
- Opacus 递归遍历 module tree，在 `MLPDiffusion` 的 `nn.Linear` 层上注册 backward hooks
- `EDMLoss` 中随机采样 `sigma`（per-batch）不影响 per-sample 梯度计算
- `loss.mean()` 仅改变 backward 时的 `grad_output` 缩放因子，Opacus 通过 `loss_reduction="mean"` 正确处理

### 2. `tabsyn/main_train_sgd.py` — 入口

```python
def main(args):
    # 1. 加载 VAE embeddings (和 tabsyn/main.py 一致)
    train_z, curr_dir, dataset_dir, ckpt_path, info = get_input_train(args)
    in_dim = train_z.shape[1]

    # 2. 归一化 (和 tabsyn/main.py 一致)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2

    # 3. DataLoader
    train_loader = DataLoader(train_z, batch_size=4096, shuffle=True, num_workers=4)

    # 4. 创建模型 (和 tabsyn/main.py 一致)
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    # 5. DP-SGD 训练
    model_save_path = f"{curr_dir}/ckpt_sgd/{dataname}"
    os.makedirs(model_save_path, exist_ok=True)
    trainer = SimpleDPSGDTrainer(model, train_loader, model_save_path, device, args)
    trainer.run_loop()
```

**CLI 参数（仅 6 个）：**
- `--dataname` / `--gpu`（通用）
- `--dp_noise_multiplier` (default 1.0)
- `--dp_max_grad_norm` (default 1.0)
- `--dp_delta` (default 1e-5)
- `--dp_steps` (default 3000)
- `--dp_physical_batch_size` (default 1024)

### 3. `utils.py` — 路由修改

在 `execute_function` 中：

```python
elif method == 'tabsyn':
    if requested_mode == 'train_sgd':
        module_name = "tabsyn.main_train_sgd"
    else:
        module_name = f"tabsyn.{mode_name}"
```

在 `get_args()` 中添加：
- `--sgd` (action='store_true') — 采样时从 `ckpt_sgd` 加载

### 4. `tabsyn/latent_utils.py` — ckpt 路径切换

在 `get_input_generate` 中：

```python
ckpt_dir = f'{curr_dir}/ckpt/{dataname}'
if getattr(args, 'sgd', False):
    ckpt_dir = f'{curr_dir}/ckpt_sgd/{dataname}'
```

## 数据流

### 训练流程

```
train_z.npy (VAE latent embeddings)
    -> get_input_train() -> torch.Tensor [B, in_dim]
    -> 归一化: (z - mean) / 2
    -> DataLoader (batch=4096)
    -> SimpleDPSGDTrainer
        -> BatchMemoryManager (物理 batch=1024)
        -> Model.forward(x) -> scalar loss
        -> Opacus hooks 计算 per-sample 梯度 -> clip -> 加噪 -> aggregate
        -> AdamW.step()
    -> 保存最佳模型到 ckpt_sgd/{dataname}/model.pt
    -> 保存 dp_metrics.json (epsilon, noise_multiplier, etc.)
```

### 采样流程

```
main.py --method tabsyn --mode sample --sgd
    -> tabsyn.sample.main(args)
    -> get_input_generate(args) [args.sgd=True -> ckpt_sgd 路径]
    -> 加载 model.pt
    -> sample(model.denoise_fn_D, ...) [复用原有逻辑]
    -> 生成 synthetic data -> CSV
```

## 使用方式

### 训练

```bash
# 先确保 VAE 已训练
python main.py --dataname shoppers --method vae --mode train --gpu 0

# DP-SGD 训练 diffusion
python main.py --dataname shoppers --method tabsyn --mode train_sgd --gpu 0

# 自定义 DP 参数
python main.py --dataname shoppers --method tabsyn --mode train_sgd \
    --dp_noise_multiplier 0.5 --dp_max_grad_norm 2.0 --dp_steps 5000 --gpu 0
```

### 采样

```bash
python main.py --dataname shoppers --method tabsyn --mode sample \
    --sgd --save_path sample_end_csv/tabsyn_sgd.csv --gpu 0
```

## 错误处理

| 场景 | 处理 |
|------|------|
| Opacus 未安装 | `ImportError` 提示 `pip install opacus` |
| VAE embeddings 不存在 | `FileNotFoundError`（和正常训练一致） |
| NaN loss | 检测并提前终止，保存已训练的最佳模型 |
| GPU OOM | BatchMemoryManager 物理 batch 拆分缓解 |

## 与 TabDDPM DP-SGD 的对比

| 特性 | TabDDPM DP-SGD | TabSyn DP-SGD (本方案) |
|------|---------------|----------------------|
| 训练空间 | 原始数据空间 | VAE latent space |
| 配置复杂度 | 15+ 参数 | 4 个 DP 参数 |
| 训练步数 | adaptive (1000-3000) | 固定 3000 |
| Early stopping | 有 | 无 |
| EMA | 有 | 无 |
| LR schedule | warmup + cosine | 固定 |
| BatchMemoryManager | 有 | 有 |
| 采样方式 | 独立采样脚本 | 复用原有 sample.py |

## 范围

**在范围内：**
- 简化版 DP-SGD diffusion 训练
- 复用原有采样
- 仅 hooks 模式

**不在范围内：**
- VAE 阶段的 DP（VAE 保持原样）
- ghost clipping 模式
- EMA / early stopping / adaptive LR
- 多 GPU 训练
