from llamascopium.utils.evo2_hooks import normalize_evo2_model_name


def test_normalize_evo2_model_name_accepts_official_hf_name() -> None:
    assert normalize_evo2_model_name("arcinstitute/evo2_7b") == "evo2_7b"
