from it_eval_framework.utils.env import huggingface_dataset_revisions


def test_huggingface_dataset_revisions_reads_cached_refs(tmp_path):
    ref = tmp_path / "hub" / "datasets--owner--corpus" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n", encoding="utf-8")

    assert huggingface_dataset_revisions({"HF_HOME": str(tmp_path)}) == [
        {"dataset_repo": "owner/corpus", "ref": "main", "revision": "abc123"}
    ]
