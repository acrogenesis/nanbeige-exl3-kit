"""Resolve pinned public models into the standard Hugging Face cache."""
import json
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

def resolve_models(require_draft=True, target_dir=None, draft_dir=None):
    lock = json.loads((ROOT / "models.lock.json").read_text())
    def resolve(name, local):
        item = lock[name]
        return str(Path(local).expanduser().resolve()) if local else snapshot_download(
            repo_id=item["repo_id"], revision=item["revision"])
    return {"target": {"path": resolve("target", target_dir)},
            "draft": {"path": resolve("draft", draft_dir) if require_draft else None}}

def seed_sparse_profile(model_dir, vocab_size):
    from exllamav3.modules.sparse_head import token_stats_path
    destination = Path(token_stats_path(model_dir, vocab_size))
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "profiles/sparse-seed.pt", destination)
