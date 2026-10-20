"""Publishing to Hugging Face, tested with a fake API: nothing is ever uploaded."""

import json
import sys
import types

import pytest

from sindhi_nlp.cli.publish_model import (
    DEFAULT_REPO_ID,
    PublishError,
    build_readme,
    main,
    plan_upload,
    publish,
)

CARD = "# Sindhi sentiment classifier\n\nA model card.\n"


class Rejected(Exception):
    """Looks like huggingface_hub's HTTP error for a rejected token."""

    response = types.SimpleNamespace(status_code=401)


class LocalTokenNotFoundError(Exception):
    """Same name as huggingface_hub's error for "no token saved on this computer"."""


class FakeApi:
    def __init__(self, name="alinawazmahar", orgs=(), logged_in=True, error=None):
        self.name, self.orgs, self.logged_in, self.error = name, orgs, logged_in, error
        self.calls = []
        self.uploaded = None

    def whoami(self):
        if self.error is not None:
            raise self.error
        if not self.logged_in:
            raise Rejected()
        return {"name": self.name, "orgs": [{"name": org} for org in self.orgs]}

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def update_repo_settings(self, **kwargs):
        self.calls.append(("update_repo_settings", kwargs))

    def upload_folder(self, **kwargs):
        from pathlib import Path

        folder = Path(kwargs["folder_path"])
        self.uploaded = {
            "files": sorted(p.name for p in folder.iterdir()),
            "readme": (folder / "README.md").read_text("utf-8"),
            "kwargs": kwargs,
        }
        self.calls.append(("upload_folder", kwargs))

    def names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "models" / "xlmr_mixed"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"weights" * 100)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "label_mapping.json"):
        (folder / name).write_text("{}", encoding="utf-8")
    (folder / "preprocessing.json").write_text(json.dumps({"cleaning": {"unicode_form": "NFKC"}}), encoding="utf-8")
    (folder / "training_config.json").write_text(
        json.dumps({"seed": 42, "config": {"run_name": "x"}}), encoding="utf-8"
    )
    (folder / "metrics.json").write_text("{}", encoding="utf-8")
    (folder / "training_args.bin").write_bytes(b"pickle")
    (folder / "training_log.jsonl").write_text("{}\n", encoding="utf-8")
    (folder / "checkpoints").mkdir()
    return folder


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "model_card.md"
    path.write_text(CARD, encoding="utf-8")
    return path


# ------------------------------------------------------------------ planning


def test_default_repo_name():
    assert DEFAULT_REPO_ID == "alinawazmahar/sindhi-sentiment-xlmr"


def test_plan_includes_only_safe_files(model_dir, card):
    plan = plan_upload(model_dir, card, DEFAULT_REPO_ID)
    assert {p.name for p in plan.files} == {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "label_mapping.json",
        "preprocessing.json",
        "training_config.json",
        "metrics.json",
    }
    assert set(plan.left_out) == {"training_args.bin", "training_log.jsonl", "checkpoints"}


def test_readme_gets_hub_metadata():
    readme = build_readme(CARD)
    assert readme.startswith("---\nlanguage: sd\nlicense: mit\n")
    assert "pipeline_tag: text-classification" in readme and readme.endswith(CARD)
    already = "---\nlicense: apache-2.0\n---\n# Card\n"
    assert build_readme(already) == already


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.json", "preprocessing.json"])
def test_missing_required_files_are_reported(model_dir, card, missing):
    (model_dir / missing).unlink()
    with pytest.raises(PublishError, match=missing):
        plan_upload(model_dir, card, DEFAULT_REPO_ID)


def test_missing_preprocessing_suggests_the_fix(model_dir, card):
    (model_dir / "preprocessing.json").unlink()
    with pytest.raises(PublishError, match="finalize-model"):
        plan_upload(model_dir, card, DEFAULT_REPO_ID)


@pytest.mark.parametrize("bad", ["noslash", "a/b/c", "/a/b", "owner/"])
def test_bad_repo_names_are_rejected(model_dir, card, bad):
    with pytest.raises(PublishError, match="owner/name"):
        plan_upload(model_dir, card, bad)


def test_missing_folder_or_card(model_dir, card, tmp_path):
    with pytest.raises(PublishError, match="Model folder not found"):
        plan_upload(tmp_path / "nope", card, DEFAULT_REPO_ID)
    with pytest.raises(PublishError, match="Model card not found"):
        plan_upload(model_dir, tmp_path / "nope.md", DEFAULT_REPO_ID)


# ---------------------------------------------------------------- privacy guard


def test_machine_paths_block_the_upload(model_dir, card):
    (model_dir / "training_config.json").write_text(
        json.dumps({"path": "C:\\Users\\Someone\\Documents\\data"}), encoding="utf-8"
    )
    api = FakeApi()
    with pytest.raises(PublishError, match="machine-specific paths.*training_config.json"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=api)
    assert api.calls == []  # nothing happened, not even repo creation


@pytest.mark.parametrize("leak", ["/home/ali/data", "/kaggle/working/x", "/Users/ali/x", "/tmp/run"])
def test_other_path_styles_are_detected(model_dir, card, leak):
    card.write_text(CARD + f"\nTrained in {leak}\n", encoding="utf-8")
    with pytest.raises(PublishError, match="README.md"):
        publish(model_dir, DEFAULT_REPO_ID, card)


def test_allow_local_paths_overrides(model_dir, card):
    (model_dir / "metrics.json").write_text('{"p": "/kaggle/working"}', encoding="utf-8")
    assert publish(model_dir, DEFAULT_REPO_ID, card, allow_local_paths=True)["uploaded"] is False


# --------------------------------------------------------------------- dry run


def test_dry_run_touches_nothing(model_dir, card):
    api = FakeApi()
    result = publish(model_dir, DEFAULT_REPO_ID, card, upload=False, api=api)
    assert result["uploaded"] is False and api.calls == []
    assert result["url"] == f"https://huggingface.co/{DEFAULT_REPO_ID}"


# ---------------------------------------------------------------------- upload


def test_upload_is_private_by_default(model_dir, card):
    api = FakeApi()
    result = publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=api)
    assert result["uploaded"] is True and result["public"] is False
    assert api.names() == ["create_repo", "upload_folder"]
    created = api.calls[0][1]
    assert created["private"] is True and created["repo_type"] == "model" and created["exist_ok"] is True
    assert api.uploaded["files"] == sorted(
        [
            "README.md",
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "label_mapping.json",
            "preprocessing.json",
            "training_config.json",
            "metrics.json",
        ]
    )
    assert "training_args.bin" not in api.uploaded["files"]
    assert api.uploaded["readme"].startswith("---\nlanguage: sd")


def test_public_needs_typed_confirmation(model_dir, card):
    api = FakeApi()
    with pytest.raises(PublishError, match="Not confirmed"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, public=True, api=api, confirm=lambda prompt: "yes")
    assert api.calls == []

    result = publish(
        model_dir, DEFAULT_REPO_ID, card, upload=True, public=True, api=api, confirm=lambda prompt: DEFAULT_REPO_ID
    )
    assert result["public"] is True
    assert api.names() == ["create_repo", "update_repo_settings", "upload_folder"]
    assert api.calls[0][1]["private"] is False and api.calls[1][1]["private"] is False


def test_yes_flag_skips_the_prompt(model_dir, card):
    def never(prompt):
        raise AssertionError("should not prompt")

    publish(model_dir, DEFAULT_REPO_ID, card, upload=True, public=True, yes=True, api=FakeApi(), confirm=never)


def test_must_be_logged_in(model_dir, card):
    with pytest.raises(PublishError, match="Not logged in"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=FakeApi(logged_in=False))


def test_cannot_publish_under_someone_elses_name(model_dir, card):
    with pytest.raises(PublishError, match="cannot publish to 'alinawazmahar'"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=FakeApi(name="someone-else"))
    api = FakeApi(name="someone-else", orgs=["alinawazmahar"])  # an organization membership is fine
    assert publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=api)["uploaded"] is True


def test_missing_huggingface_hub_package(model_dir, card, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(PublishError, match="pip install huggingface_hub"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True)


# ------------------------------------------------------------------------- CLI


def test_cli_dry_run(model_dir, card, tmp_path, capsys):
    code = main(["--model-dir", str(model_dir), "--model-card", str(card), "--project-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "DRY RUN" in out and "model.safetensors" in out
    assert "training_args.bin" in out and "Left out on purpose" in out


def test_cli_reports_problems(model_dir, card, tmp_path, capsys):
    (model_dir / "model.safetensors").unlink()
    assert main(["--model-dir", str(model_dir), "--model-card", str(card), "--project-root", str(tmp_path)]) == 1
    assert "model.safetensors" in capsys.readouterr().err


@pytest.mark.parametrize("error", [Rejected(), LocalTokenNotFoundError("no token")])
def test_login_problems_say_so(model_dir, card, error):
    with pytest.raises(PublishError, match="Not logged in.*hf auth login"):
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=FakeApi(error=error))


@pytest.mark.parametrize("error", [ConnectionError("connection reset"), TimeoutError("timed out")])
def test_network_problems_are_not_blamed_on_the_login(model_dir, card, error):
    api = FakeApi(error=error)
    with pytest.raises(PublishError) as caught:
        publish(model_dir, DEFAULT_REPO_ID, card, upload=True, api=api)
    message = str(caught.value)
    assert "Could not reach Hugging Face" in message and "try again" in message
    assert "Not logged in" not in message and type(error).__name__ in message
    assert api.calls == []  # nothing was created or uploaded
