# SO-ARM101 + LingBot-VLA-4B 工作流

本项目采用“本仓库负责编排，官方仓库负责模型训练/推理”的方式。

## 0. 官方仓库

先在训练或推理机器上准备官方仓库：

```bash
git clone https://github.com/Robbyant/lingbot-vla.git
```

然后在 `configs/pipeline_so_arm101.yaml` 里把 `official_repo` 指向它。

## 1. 数据采集

采集本身建议继续使用 LeRobot 的 `record` 能力或你已有的 SO-ARM101 采集脚本。LingBot-VLA 后训练需要 LeRobot v3.0 格式数据，核心字段是：

- `observation.state`
- `action`
- `observation.images.camera_top`
- `observation.images.camera_wrist` 或左右腕相机

单个任务建议先采 80-100 条演示，任务指令和相机命名要和训练配置保持一致。

## 2. 数据合并、转换、清洗

打印命令：

```bash
python scripts/pipeline.py --stage merge_v21
python scripts/pipeline.py --stage convert_v30
python scripts/pipeline.py --stage quality
```

`quality` 会生成 JSON 报告，检查 `meta/info.json`、episode 长度、图像字段和常用状态/动作字段。它不会删除数据，只给出建议排除的 episode，避免误伤原始采集数据。

## 3. 归一化统计

```bash
python scripts/pipeline.py --stage norm
```

对应官方命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm.py ...
```

输出写到 `training.norm_stats_file`。

## 4. 模型后训练

```bash
python scripts/pipeline.py --stage train
```

对应官方 real-robot 推荐配置 `configs/vla/real_load20000h.yaml` 的参数风格。本项目里的 `configs/vla/so_arm101_vla_task.yaml` 是 SO-ARM101 单臂模板。

## 5. 仿真/开环验证

```bash
python scripts/pipeline.py --stage eval
```

它调用官方：

```bash
python scripts/open_loop_eval.py --model_path <hf_ckpt> --data_path <validation_data>
```

结果图保存到 `open_loop_results/so_arm101`。

## 6. 实际部署

服务器端：

```bash
python scripts/pipeline.py --stage serve
```

本地机械臂端：

```bash
python scripts/pipeline.py --stage deploy_client
```

本地 4090 本体推理：

```bash
python scripts/pipeline.py --stage local_serve
python scripts/pipeline.py --stage local_stack
```

真机前先运行：

```bash
python scripts/run_client.py --config configs/local_so_arm101.yaml --dry-run
python scripts/validate_config.py --config configs/local_so_arm101.yaml
```
