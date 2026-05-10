"""学科配置文件加载与验证模块。"""

from __future__ import annotations

import json
from pathlib import Path

from thesischeck.core.config.models import DisciplineConfig

# 学科配置文件所在的目录
_DISCIPLINES_DIR = Path(__file__).parent / "disciplines"

# 内置学科配置缓存
_builtin_cache: dict[str, DisciplineConfig] | None = None


def _load_discipline_file(name: str) -> dict:
    """从 JSON 文件加载学科原始配置数据。

    Args:
        name: 学科名称（同时也是文件名，不含 .json 后缀）。

    Returns:
        解析后的字典数据。

    Raises:
        FileNotFoundError: 指定学科配置文件不存在。
        json.JSONDecodeError: 配置文件格式错误。
    """
    file_path = _DISCIPLINES_DIR / f"{name}.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"学科配置文件 '{name}.json' 不存在于 {_DISCIPLINES_DIR}"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_default_config() -> dict:
    """加载默认学科配置。

    Returns:
        默认配置字典。
    """
    return _load_discipline_file("default")


def load_discipline_config(discipline_name: str) -> DisciplineConfig:
    """加载指定学科的配置，与默认配置合并。

    使用深合并策略：学科配置中的字段会覆盖默认配置的同名字段，
    而未指定的字段则沿用默认值。

    Args:
        discipline_name: 学科名称标识。

    Returns:
        验证通过后的 DisciplineConfig 实例。
    """
    defaults = _load_default_config()

    if discipline_name == "default":
        return DisciplineConfig(**defaults)

    try:
        specific = _load_discipline_file(discipline_name)
    except FileNotFoundError:
        return DisciplineConfig(**defaults)

    # 深合并：学科配置覆盖默认配置
    merged = _deep_merge(defaults, specific)
    return DisciplineConfig(**merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典。

    Args:
        base: 基础字典。
        override: 要覆盖的字典。

    Returns:
        合并后的新字典。
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def list_available_disciplines() -> list[dict[str, str]]:
    """列出所有可用的学科配置。

    Returns:
        list，每个元素为包含 name 和 display_name 的 dict。
    """
    disciplines: list[dict[str, str]] = []
    for file_path in _DISCIPLINES_DIR.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            disciplines.append({
                "name": data.get("name", file_path.stem),
                "display_name": data.get("display_name", file_path.stem),
                "description": data.get("description", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(disciplines, key=lambda d: d["name"])


def validate_config(data: dict) -> DisciplineConfig:
    """验证配置数据的完整性。

    Args:
        data: 待验证的配置字典。

    Returns:
        验证通过后的 DisciplineConfig 实例。

    Raises:
        pydantic.ValidationError: 配置数据无效。
    """
    return DisciplineConfig(**data)


__all__ = [
    "load_discipline_config",
    "list_available_disciplines",
    "validate_config",
]
