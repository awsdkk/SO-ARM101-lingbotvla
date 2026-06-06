# 本地 4090 推理部署

适用场景：本机有 24G RTX 4090，可以直接运行 LingBot-VLA-4B 推理服务，不需要云端 GPU。

## 1. 修改配置

编辑 `configs/local_so_arm101.yaml` 的 `local_inference`：

```yaml
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

同时确认 `server` 指向本机：

```yaml
server:
  host: "ws://127.0.0.1"
  port: 8006
```

## 2. 打印本地服务端命令

```powershell
python scripts\print_local_inference_command.py --config configs\local_so_arm101.yaml
```

## 3. 启动本地推理服务

```powershell
python scripts\run_local_inference_server.py --config configs\local_so_arm101.yaml
```

服务启动后会等待 `http://127.0.0.1:8006/healthz` 返回 OK。

## 4. 启动本体客户端

另开一个终端：

```powershell
python scripts\run_client.py --config configs\local_so_arm101.yaml
```

或者一条命令同时启动服务端和客户端：

```powershell
python scripts\run_local_inference_stack.py --config configs\local_so_arm101.yaml
```

先做无硬件验证：

```powershell
python scripts\run_local_inference_stack.py --config configs\local_so_arm101.yaml --dry-run-client
```

## 5. 4090 推荐参数

- `use_length: 25`：真机闭环部署推荐。
- `num_denoising_step: 5`：更快，精度略降；如果动作质量不够，再提高到 10。
- `use_compile: true`：适合长时间运行，首次推理会有编译开销。
- `CUDA_VISIBLE_DEVICES=0`：只用第一张 GPU。

