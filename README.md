# SO-ARM101 LingBot-VLA

面向 SO-ARM101 机械臂的 LingBot-VLA-4B 训练、验证与部署工程。

本项目将 LingBot-VLA 官方模型能力接入 SO-ARM101 本体，覆盖从 LeRobot 数据准备、数据质量检查、归一化统计、模型后训练、开环验证，到本地 4090 或远端 GPU 推理部署的完整流程。项目侧重点是工程编排和本体适配：模型训练与推理核心仍复用 Robbyant/LingBot-VLA 官方实现，SO-ARM101 相关的配置、脚本、WebSocket 客户端、动作块调度和部署入口在本仓库维护。

## 项目特性

- 支持 SO-ARM101 单臂配置，预留双臂 SO-ARM101 配置模板。
- 支持 LingBot-VLA-4B 后训练流程，包括 LeRobot v3.0 数据、robot config、norm stats 和训练配置。
- 支持数据质量检查，生成 LeRobot 数据集报告，辅助筛查 episode 长度和关键字段。
- 支持开环验证，调用官方 `scripts/open_loop_eval.py` 输出动作误差和轨迹对比图。
- 支持两种推理部署方式：本地 RTX 4090 推理和远端 GPU/WebSocket 推理。
- 支持真机部署前 dry-run，用于验证配置、WebSocket、动作块和 RTC 调度逻辑。
- 对齐官方 WebSocket 协议：MessagePack + NumPy 序列化、连接 metadata、`reset(robo_name)` 初始化。

## 系统架构

```text
┌──────────────────────────────────────────────────────────────┐
│                    GPU 推理端                                │
│  本地 RTX 4090 或远端 GPU                                    │
│  Robbyant/lingbot-vla                                        │
│  python -m deploy.lingbot_vla_policy                         │
└───────────────────────────────┬──────────────────────────────┘
                                │ WebSocket
                                │ MessagePack + NumPy
┌───────────────────────────────▼──────────────────────────────┐
│                    SO-ARM101 控制端                           │
│  LeRobot Robot Adapter                                        │
│  WebsocketClientPolicy                                        │
│  RTC Action Chunk Scheduler                                   │
│  scripts/run_client.py                                        │
└──────────────────────────────────────────────────────────────┘
```

本地 4090 部署时，推理端和控制端在同一台机器上运行，客户端连接 `ws://127.0.0.1:8006`。远端部署时，只需要把 `configs/local_so_arm101.yaml` 中的 `server.host` 改成远端 WebSocket 地址。

## 目录结构

```text
configs/
  local_so_arm101.yaml              # 本地机械臂、相机、推理服务、RTC 参数
  pipeline_so_arm101.yaml           # 数据、训练、验证、部署流水线参数
  robot_configs/
    so_arm101.yaml                  # 单臂 SO-ARM101 LingBot-VLA 特征映射
    bi_so_arm101.yaml               # 双臂 SO-ARM101 特征映射模板
  vla/
    so_arm101_vla_task.yaml         # 单臂后训练配置模板
    bi_so_arm101_vla_task.yaml      # 双臂后训练配置模板

docs/
  WORKFLOW.md                       # 数据、训练、验证、部署全流程说明
  LOCAL_INFERENCE.md                # 本地 RTX 4090 推理部署说明

scripts/
  clean_lerobot_dataset.py          # LeRobot 数据质量检查
  pipeline.py                       # 按阶段生成/执行官方 LingBot-VLA 命令
  validate_config.py                # 本地配置校验
  print_server_command.py           # 远端推理服务命令生成
  print_local_inference_command.py  # 本地 4090 推理服务命令生成
  run_local_inference_server.py     # 启动本地 4090 推理服务
  run_local_inference_stack.py      # 启动本地推理服务并运行客户端
  run_client.py                     # SO-ARM101 真机客户端入口

src/lingbot_so_arm101/
  config.py                         # YAML 配置加载与校验
  local_inference.py                # 本地推理服务进程管理
  pipeline.py                       # 流水线命令构造
  robot_adapter.py                  # LeRobot 适配层
  rtc.py                            # 实时动作块调度
  serialization.py                  # 与官方兼容的 MessagePack + NumPy 序列化
  server_command.py                 # 推理服务命令构造
  websocket_policy.py               # WebSocket 推理客户端
```

## 前置条件
可更换不同的末端设备
<img width="1280" height="1706" alt="96d48961beac52eb41b9d4e5249743a5" src="https://github.com/user-attachments/assets/0c75892a-3190-4a1e-82df-deee8e89e9bb" />
<img width="1280" height="1706" alt="760b5363e7107169ca9a622690a3a22d" src="https://github.com/user-attachments/assets/3e6991f6-8cc4-490e-a26b-a48f2903525f" />

本仓库不负责安装环境和下载模型。运行前需要准备：

- Python 3.10+，用于运行本项目脚本。
- LeRobot 环境，用于连接 SO-ARM101 本体。
- LingBot-VLA 官方仓库环境，用于模型后训练和推理。
- LingBot-VLA-4B 或后训练后的 `hf_ckpt`。
- `Qwen2.5-VL-3B-Instruct` 模型路径。
- 本地推理推荐 RTX 4090 24G，真机部署推荐 `use_length=25`。

官方仓库：

```bash
git clone https://github.com/Robbyant/lingbot-vla.git
```

将 `configs/pipeline_so_arm101.yaml` 中的 `official_repo` 指向该仓库路径。

## 快速开始

### 1. 修改本地配置

编辑：

```text
configs/local_so_arm101.yaml
```

关键字段：

- `mode`：`single` 或 `dual`。
- `robot.port`：SO-ARM101 串口，例如 `COM3`。
- `robot.id`：LeRobot 中使用的机器人 ID。
- `cameras.*.index_or_path`：顶置相机和腕部相机 ID。
- `server.host` / `server.port`：推理服务 WebSocket 地址。
- `task`：自然语言任务指令。
- `lingbot.robot_config_name`：默认 `so_arm101`。

校验配置：

```bash
python scripts/validate_config.py --config configs/local_so_arm101.yaml
```

### 2. dry-run 验证客户端

dry-run 不连接真实机械臂，用零状态和空图像检查客户端链路：

```bash
python scripts/run_client.py --config configs/local_so_arm101.yaml --dry-run
```

### 3. 运行真机客户端

确认推理服务已启动后运行：

```bash
python scripts/run_client.py --config configs/local_so_arm101.yaml
```

## 本地 4090 推理部署

本地 4090 推理适合单机闭环调试：同一台电脑运行 LingBot-VLA 推理服务和 SO-ARM101 控制客户端。

编辑 `configs/local_so_arm101.yaml`：

```yaml
server:
  host: "ws://127.0.0.1"
  port: 8006

local_inference:
  official_repo: ".codex/vendor/lingbot-vla"
  qwen25_path: "E:/models/Qwen2.5-VL-3B-Instruct"
  model_path: "E:/models/so_arm101_hf_ckpt"
  cuda_visible_devices: "0"
  port: 8006
  use_length: 25
  num_denoising_step: 5
  use_compile: true
```

打印本地推理命令：

```bash
python scripts/print_local_inference_command.py --config configs/local_so_arm101.yaml
```

启动本地推理服务：

```bash
python scripts/run_local_inference_server.py --config configs/local_so_arm101.yaml
```

另开终端启动客户端：

```bash
python scripts/run_client.py --config configs/local_so_arm101.yaml
```

一键启动服务端和客户端：

```bash
python scripts/run_local_inference_stack.py --config configs/local_so_arm101.yaml
```

更多说明见 [docs/LOCAL_INFERENCE.md](docs/LOCAL_INFERENCE.md)。

## 远端 GPU 推理部署

远端部署适合云端 GPU 推理、本地电脑控制 SO-ARM101 的场景。

生成远端推理服务命令：

```bash
python scripts/print_server_command.py --config configs/local_so_arm101.yaml
```

典型服务端命令：

```bash
export TOKENIZERS_PARALLELISM=false
export QWEN25_PATH=/data/models/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
  --model_path /data/output/checkpoints/global_step_xxxx/hf_ckpt \
  --use_length 25 \
  --num_denoising_step 5 \
  --port 8006 \
  --use_compile
```

本地客户端只需要将 `server.host` 改为远端地址，例如：

```yaml
server:
  host: "wss://your-server.example.com"
  port: 30499
```

## 数据与后训练流程

完整流程由 `scripts/pipeline.py` 统一生成命令，默认只打印，不执行。确认路径正确后可加 `--execute`。

### 数据合并与转换

```bash
python scripts/pipeline.py --stage merge_v21
python scripts/pipeline.py --stage convert_v30
```

如果已经是 LeRobot v3.0 数据，可以跳过转换。

### 数据质量检查

```bash
python scripts/pipeline.py --stage quality
```

输出示例：

```text
reports/data_quality_so_arm101.json
```

### 归一化统计

```bash
python scripts/pipeline.py --stage norm
```

对应官方：

```bash
bash train.sh scripts/compute_norm.py ...
```

### 模型后训练

```bash
python scripts/pipeline.py --stage train
```

训练使用：

- `configs/robot_configs/so_arm101.yaml`
- `configs/vla/so_arm101_vla_task.yaml`
- `assets/norm_stats/so_arm101.json`

### 开环验证

```bash
python scripts/pipeline.py --stage eval
```

该阶段调用官方 `scripts/open_loop_eval.py`，输出 MSE、MAE 和轨迹可视化图。

更多流程细节见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## Robot Config 说明

单臂 SO-ARM101 默认状态/动作维度：

```text
observation.state: [joint_0 ... joint_5, gripper]
action:            [joint_0 ... joint_5, gripper]
```

映射文件：

```text
configs/robot_configs/so_arm101.yaml
```

其中：

- `observation.state.arm.position` 使用前 6 维。
- `observation.state.effector.position` 使用第 7 维。
- `action.arm.position` 使用 `subtract_state: true`，符合官方真机数据推荐。
- `action.effector.position` 使用绝对夹爪动作。

双臂模板：

```text
configs/robot_configs/bi_so_arm101.yaml
```

默认布局为左臂 7 维 + 右臂 7 维。

## 与 LingBot-VLA 官方仓库的关系

本仓库不复制、不修改 LingBot-VLA 模型训练和推理核心。以下能力直接调用官方仓库：

- `scripts/compute_norm.py`
- `tasks/vla/train_lingbotvla.py`
- `scripts/open_loop_eval.py`
- `deploy/lingbot_vla_policy.py`
- `deploy/websocket_policy_server.py`

本仓库维护 SO-ARM101 专属的工程层：

- 本体与相机配置。
- LeRobot 数据检查和流水线命令编排。
- SO-ARM101 robot config 和 VLA config。
- 本地/远端推理服务启动脚本。
- WebSocket 客户端、动作块调度和真机入口。

## 安全建议

- 真机运行前先执行 `--dry-run`。
- 确认 SO-ARM101 已完成零点校准。
- 首次闭环测试降低速度、限制运行时长，并保持急停可用。
- 本地 4090 首次启用 `use_compile` 会有编译等待时间，服务稳定后再启动机械臂。
- 若动作抖动明显，可提高 `num_denoising_step` 或调整 `rtc.action_ema_alpha`、`rtc.blend_steps`。

## License

本仓库作为 SO-ARM101 工程适配层维护。LingBot-VLA 模型和官方源码遵循其原仓库许可证。
