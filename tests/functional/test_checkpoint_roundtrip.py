import pytest
import torch
import tempfile
from pathlib import Path


class TestCheckpointRoundtrip:

    def test_save_load_identical_output(self, model, synthetic_sample):
        eeg, _ = synthetic_sample
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            out_before = model(eeg)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model_state": model.state_dict()}, checkpoint_path)

        from src.phase2_network.physdeepsif import build_physdeepsif
        new_model = build_physdeepsif(
            leadfield_path=str(Path(__file__).resolve().parent.parent.parent / "data" / "leadfield_19x76.npy"),
            connectivity_path=str(Path(__file__).resolve().parent.parent.parent / "data" / "connectivity_76.npy"),
            lstm_hidden_size=76,
        )
        new_model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model.load_state_dict(checkpoint["model_state"])
        new_model.eval()

        with torch.no_grad():
            out_after = new_model(eeg)

        diff = (out_before - out_after).abs().max().item()
        assert diff < 1e-5, f"Output difference: {diff:.2e}"

    def test_state_dict_keys_match(self, model):
        keys_before = set(model.state_dict().keys())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state": model.state_dict()}, f.name)
            checkpoint = torch.load(f.name, map_location="cpu")
        keys_loaded = set(checkpoint["model_state"].keys())
        assert keys_before == keys_loaded, (
            f"Missing keys: {keys_before - keys_loaded}"
        )

    def test_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            torch.load("/nonexistent/checkpoint.pt")

    def test_corrupted_checkpoint_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"not a valid checkpoint")
            f.flush()
            with pytest.raises(Exception):
                torch.load(f.name, map_location="cpu")
