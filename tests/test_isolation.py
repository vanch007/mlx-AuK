import sys

def test_no_mlx_audio_dependency():
    sys.path.insert(0, "/Users/vanch/mlx-AuK/src")
    import mlx_auk
    for mod in list(sys.modules.keys()):
        assert not mod.startswith("mlx_audio"), f"Forbidden dependency detected: {mod}"
    print("Isolation check passed: mlx-auk has zero dependency on mlx_audio!")

if __name__ == "__main__":
    test_no_mlx_audio_dependency()
