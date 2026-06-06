from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project_root(self) -> Path:
        return self.path.parent.parent

    @property
    def official_repo(self) -> Path:
        repo = Path(str(self.raw.get("official_repo", "../lingbot-vla")))
        if not repo.is_absolute():
            repo = (self.project_root / repo).resolve()
        return repo

    def project_path(self, value: str | Path) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping: {config_path}")
    return PipelineConfig(raw=raw, path=config_path)


def build_stage_commands(config: PipelineConfig, stage: str) -> list[str]:
    builders = {
        "merge_v21": _merge_v21_commands,
        "convert_v30": _convert_v30_commands,
        "quality": _quality_commands,
        "norm": _norm_commands,
        "train": _train_commands,
        "eval": _eval_commands,
        "serve": _serve_commands,
        "local_serve": _local_serve_commands,
        "local_stack": _local_stack_commands,
        "deploy_client": _deploy_client_commands,
    }
    if stage == "all":
        commands: list[str] = []
        for name in ("merge_v21", "convert_v30", "quality", "norm", "train", "eval", "serve", "deploy_client"):
            commands.extend(build_stage_commands(config, name))
        return commands
    if stage not in builders:
        raise ValueError(f"Unknown stage: {stage}")
    return builders[stage](config)


def run_commands(commands: Iterable[str], cwd: Path | None = None) -> None:
    for command in commands:
        print(f"$ {command}")
        subprocess.run(command, cwd=cwd, shell=True, check=True)


def _merge_v21_commands(config: PipelineConfig) -> list[str]:
    data = config.raw.get("data", {})
    sources = data.get("raw_v21_sources", [])
    if not sources:
        return ["# merge_v21 skipped: data.raw_v21_sources is empty"]
    return [
        _in_repo(
            config,
            _join(
                [
                    "python",
                    "scripts/merge_lerobot_v21.py",
                    "--sources",
                    ",".join(str(source) for source in sources),
                    "--output",
                    str(data["merged_v21_path"]),
                ]
            ),
        )
    ]


def _convert_v30_commands(config: PipelineConfig) -> list[str]:
    data = config.raw.get("data", {})
    source = data.get("merged_v21_path") or "<path_to_lerobot_v21_dataset>"
    return [
        _join(
            [
                "python",
                "-m",
                "lerobot.datasets.v30.convert_dataset_v21_to_v30",
                f"--repo-id={data.get('v30_repo_id', 'so_arm101_task')}",
                f"--root={data.get('v30_root', source)}",
                "--push-to-hub=false",
            ]
        )
    ]


def _quality_commands(config: PipelineConfig) -> list[str]:
    data = config.raw.get("data", {})
    return [
        _join(
            [
                "python",
                "scripts/clean_lerobot_dataset.py",
                "--dataset",
                str(data.get("v30_root", "data/so_arm101_task_v30")),
                "--report",
                str(data.get("quality_report", "reports/data_quality_so_arm101.json")),
            ]
        )
    ]


def _norm_commands(config: PipelineConfig) -> list[str]:
    train = config.raw.get("training", {})
    robot_config_root = config.project_path("configs/robot_configs")
    vla_config = config.project_path(train.get("vla_config", "configs/vla/so_arm101_vla_task.yaml"))
    command = _join(
        [
            "bash",
            "train.sh",
            "scripts/compute_norm.py",
            str(vla_config),
            "--model.model_path",
            str(train.get("model_path", "/data/models/lingbot-vla-4b")),
            "--model.tokenizer_path",
            str(train.get("tokenizer_path", "/data/models/Qwen2.5-VL-3B-Instruct")),
            "--data.data_name",
            str(train.get("robot_config_name", "so_arm101")),
            "--data.robot_config_root",
            str(robot_config_root),
            "--data.train_path",
            str(config.raw.get("data", {}).get("v30_root", "data/so_arm101_task_v30")),
            "--data.norm_stats_file",
            str(train.get("norm_stats_file", "assets/norm_stats/so_arm101.json")),
        ]
    )
    return [_in_repo(config, _with_cuda(train, command))]


def _train_commands(config: PipelineConfig) -> list[str]:
    train = config.raw.get("training", {})
    robot_config_root = config.project_path("configs/robot_configs")
    vla_config = config.project_path(train.get("vla_config", "configs/vla/so_arm101_vla_task.yaml"))
    command = _join(
        [
            "bash",
            "train.sh",
            "tasks/vla/train_lingbotvla.py",
            str(vla_config),
            "--model.model_path",
            str(train.get("model_path", "/data/models/lingbot-vla-4b")),
            "--model.tokenizer_path",
            str(train.get("tokenizer_path", "/data/models/Qwen2.5-VL-3B-Instruct")),
            "--data.data_name",
            str(train.get("robot_config_name", "so_arm101")),
            "--data.robot_config_root",
            str(robot_config_root),
            "--data.train_path",
            str(config.raw.get("data", {}).get("v30_root", "data/so_arm101_task_v30")),
            "--data.norm_stats_file",
            str(train.get("norm_stats_file", "assets/norm_stats/so_arm101.json")),
            "--train.output_dir",
            str(train.get("output_dir", "output/so_arm101")),
        ]
    )
    return [_in_repo(config, _with_cuda(train, command))]


def _eval_commands(config: PipelineConfig) -> list[str]:
    eval_config = config.raw.get("evaluation", {})
    train = config.raw.get("training", {})
    traj_ids = [str(item) for item in eval_config.get("traj_ids", [0])]
    commands = [
        f"export QWEN25_PATH={shlex.quote(str(train.get('tokenizer_path', '/data/models/Qwen2.5-VL-3B-Instruct')))}",
        _in_repo(
            config,
            _join(
                [
                    "python",
                    "scripts/open_loop_eval.py",
                    "--model_path",
                    str(eval_config.get("hf_ckpt", "<path_to_hf_ckpt>")),
                    "--robo_name",
                    str(train.get("robot_config_name", "so_arm101")),
                    "--norm_path",
                    str(train.get("norm_stats_file", "assets/norm_stats/so_arm101.json")),
                    "--data_path",
                    str(eval_config.get("validation_data", config.raw.get("data", {}).get("v30_root", ""))),
                    "--traj_ids",
                    *traj_ids,
                    "--use_length",
                    str(eval_config.get("use_length", 50)),
                    "--save_plot_path",
                    str(eval_config.get("save_plot_path", "open_loop_results/so_arm101")),
                ]
            ),
        ),
    ]
    return commands


def _serve_commands(config: PipelineConfig) -> list[str]:
    deploy = config.raw.get("deployment", {})
    args = [
        "python",
        "-m",
        "deploy.lingbot_vla_policy",
        "--model_path",
        str(deploy.get("hf_ckpt", "<path_to_hf_ckpt>")),
        "--use_length",
        str(deploy.get("use_length", 25)),
        "--num_denoising_step",
        str(deploy.get("num_denoising_step", 5)),
        "--port",
        str(deploy.get("port", 8006)),
    ]
    if deploy.get("use_compile", True):
        args.append("--use_compile")
    return [
        "export TOKENIZERS_PARALLELISM=false",
        f"export QWEN25_PATH={shlex.quote(str(deploy.get('qwen25_path', '/data/models/Qwen2.5-VL-3B-Instruct')))}",
        _in_repo(config, _join(args)),
    ]


def _deploy_client_commands(config: PipelineConfig) -> list[str]:
    return [
        _join(["python", "scripts/run_client.py", "--config", str(config.raw.get("local_config", "configs/local_so_arm101.yaml"))])
    ]


def _local_serve_commands(config: PipelineConfig) -> list[str]:
    return [
        _join(
            [
                "python",
                "scripts/run_local_inference_server.py",
                "--config",
                str(config.raw.get("local_config", "configs/local_so_arm101.yaml")),
            ]
        )
    ]


def _local_stack_commands(config: PipelineConfig) -> list[str]:
    return [
        _join(
            [
                "python",
                "scripts/run_local_inference_stack.py",
                "--config",
                str(config.raw.get("local_config", "configs/local_so_arm101.yaml")),
            ]
        )
    ]


def _with_cuda(train: dict[str, Any], command: str) -> str:
    devices = train.get("cuda_visible_devices")
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(str(devices))} {command}" if devices else command


def _join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _in_repo(config: PipelineConfig, command: str) -> str:
    return f"cd {shlex.quote(str(config.official_repo))} && {command}"
