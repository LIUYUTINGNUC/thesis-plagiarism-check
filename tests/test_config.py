"""Tests for configuration models and loader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from thesischeck.core.config.loader import (
    list_available_disciplines,
    load_discipline_config,
    validate_config,
)
from thesischeck.core.config.models import (
    AIConfig,
    CitationConfig,
    DisciplineConfig,
    SimilarityConfig,
)


class TestConfigModels:
    def test_citation_config_defaults(self):
        cfg = CitationConfig()
        assert cfg.max_quotation_ratio == 0.3
        assert cfg.min_novel_finding_ratio == 0.3
        assert cfg.self_plagiarism_threshold == 0.3

    def test_citation_config_invalid(self):
        with pytest.raises(ValidationError):
            CitationConfig(max_quotation_ratio=1.5)

    def test_similarity_config_defaults(self):
        cfg = SimilarityConfig()
        assert cfg.semantic_threshold == 0.75
        assert cfg.kgraph_weight == 0.4

    def test_ai_config_defaults(self):
        cfg = AIConfig()
        assert "entropy" in cfg.feature_weights
        assert 0.0 <= cfg.ensemble_threshold <= 1.0

    def test_discipline_config(self):
        cfg = DisciplineConfig(name="test")
        assert cfg.name == "test"
        assert isinstance(cfg.citation, CitationConfig)

    def test_discipline_config_no_name_fails(self):
        with pytest.raises(ValidationError):
            DisciplineConfig()

    def test_discipline_config_round_trip(self):
        cfg = DisciplineConfig(name="medicine", display_name="Medicine")
        data = cfg.model_dump()
        restored = DisciplineConfig(**data)
        assert restored.name == cfg.name
        assert restored.display_name == cfg.display_name


class TestConfigLoader:
    def test_load_default_config(self):
        cfg = load_discipline_config("default")
        assert cfg.name == "default"

    def test_load_medicine_config(self):
        cfg = load_discipline_config("medicine")
        assert cfg.name == "medicine"

    def test_load_cs_config(self):
        cfg = load_discipline_config("cs")
        assert cfg.name == "cs"

    def test_load_humanities_config(self):
        cfg = load_discipline_config("humanities")
        assert cfg.name == "humanities"

    def test_unknown_discipline_falls_back_to_default(self):
        cfg = load_discipline_config("nonexistent_discipline_xyz")
        assert cfg.name == "default"

    def test_list_disciplines_returns_all(self):
        disciplines = list_available_disciplines()
        names = [d["name"] for d in disciplines]
        assert "default" in names
        assert "medicine" in names
        assert "cs" in names
        assert "humanities" in names

    def test_validate_config(self):
        data = {
            "name": "custom",
            "citation": {"max_quotation_ratio": 0.5},
        }
        cfg = validate_config(data)
        assert cfg.name == "custom"
        assert cfg.citation.max_quotation_ratio == 0.5


class TestConfigMerging:
    def test_discipline_overrides_default(self):
        cfg = load_discipline_config("medicine")
        assert cfg.citation.max_quotation_ratio > 0.3


class TestConfigPersistence:
    def test_config_files_exist(self):
        config_dir = Path(__file__).parent.parent / "src" / "thesischeck" / "core" / "config" / "disciplines"
        assert (config_dir / "default.json").exists()
        assert (config_dir / "medicine.json").exists()
        assert (config_dir / "cs.json").exists()
        assert (config_dir / "humanities.json").exists()

    def test_config_files_valid_json(self):
        config_dir = Path(__file__).parent.parent / "src" / "thesischeck" / "core" / "config" / "disciplines"
        for f in config_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            assert "name" in data
            assert "citation" in data
            assert "similarity" in data
            assert "ai_detection" in data