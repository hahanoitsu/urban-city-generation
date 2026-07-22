from urban_model.config import load_training_config


def test_training_config_resolves_paths(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    path = config_dir / "test.yaml"
    path.write_text(
        "data:\n"
        "  train_manifest: data/manifests/train.jsonl\n"
        "  validation_manifest: data/manifests/validation.jsonl\n"
        "run:\n"
        "  output_dir: runs/test\n"
    )
    config = load_training_config(path)
    assert config.data.train_manifest == (tmp_path / "data/manifests/train.jsonl").resolve()
    assert config.run.output_dir == (tmp_path / "runs/test").resolve()
