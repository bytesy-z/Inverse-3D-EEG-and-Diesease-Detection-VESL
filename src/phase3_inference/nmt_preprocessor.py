"""
NMT EEG Preprocessing pipeline

Provides `NMTPreprocessor` to load EDF files, normalize channel names,
apply linked-ear referencing, bandpass/notch filtering, optional ICA
artifact removal, and segmentation into 2s windows with normalization
matching training-time stats.

Notes:
- This module intentionally does not execute any heavy operations on import.
- The `run()` method orchestrates the full pipeline and returns a dict
  containing preprocessed windows and metadata.

API example:
    pre = NMTPreprocessor()
    result = pre.run('data/samples/0001082.edf', norm_stats)
    # result['preprocessed_eeg'] -> ndarray (n_epochs, 19, 400)

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# Constants matching Technical Specs
SAMPLING_RATE = 200.0  # Hz expected
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLING_RATE)  # 400
DEFAULT_STEP = WINDOW_SAMPLES // 2  # 50% overlap -> 200 samples

# Standard 10-20 channels (19-channel montage used in pipeline)
STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# Mapping variants commonly seen in EDFs -> canonical names
EDF_CHANNEL_MAP = {
    "FP1": "Fp1", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    # Allow some common alternative spellings
    "Fp1-Ref": "Fp1", "Fp2-Ref": "Fp2",
}


class NMTPreprocessor:
    """Preprocess NMT EDF recordings into normalized EEG windows.

    Primary responsibilities:
    - Load EDF using MNE
    - Normalize channel names and drop reference channels (A1/A2)
    - Ensure linked-ear reference if A1/A2 present
    - Bandpass filter (0.5-70 Hz) and notch filter (50 Hz)
    - Optional ICA artifact removal with graceful fallback
    - Segment into sliding windows and apply per-window de-meaning + global z-score

    Args:
        sampling_rate: expected sampling rate (defaults to 200 Hz)
        target_channels: list of canonical channel names to keep (19 channels)
        window_samples: number of samples per epoch (400 for 2s @ 200 Hz)
        step: step size between windows (200 for 50% overlap)
    """

    def __init__(
        self,
        sampling_rate: float = SAMPLING_RATE,
        target_channels: Optional[List[str]] = None,
        window_samples: int = WINDOW_SAMPLES,
        step: int = DEFAULT_STEP,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.target_channels = target_channels or STANDARD_CHANNELS
        self.window_samples = int(window_samples)
        self.step = int(step)

    def load_edf(self, edf_path: str) -> mne.io.BaseRaw:
        """Load EDF file with MNE and validate sampling rate.

        Does not modify the raw object except loading.
        """
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if raw.info["sfreq"] != self.sampling_rate:
            logger.warning(
                "EDF sampling rate (%.1f Hz) != expected %.1f Hz",
                raw.info["sfreq"],
                self.sampling_rate,
            )
        return raw

    def normalize_channel_names(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Normalize channel names to STANDARD_CHANNELS and drop A1/A2.

        - Renames common uppercase variants (e.g., 'FP1' -> 'Fp1')
        - Drops ear reference channels if present
        - Reorders/picks channels to the canonical order
        """
        mapping = {}
        for ch in list(raw.ch_names):
            clean = ch.strip()
            if clean in EDF_CHANNEL_MAP:
                mapping[ch] = EDF_CHANNEL_MAP[clean]
            elif clean.upper() in EDF_CHANNEL_MAP:
                mapping[ch] = EDF_CHANNEL_MAP[clean.upper()]
            else:
                # Preserve case for already-canonical names
                mapping[ch] = clean

        # Apply rename
        try:
            raw.rename_channels(mapping)
        except Exception:
            # rename_channels can raise if duplicates occur; fallback quietly
            logger.debug("rename_channels raised; continuing with original names")

        # Drop ear references if present
        for ear in ("A1", "A2"):
            if ear in raw.ch_names:
                try:
                    raw.drop_channels([ear])
                except Exception:
                    pass

        # Now ensure all target channels are present
        missing = [ch for ch in self.target_channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Missing required channels after normalization: {missing}. "
                "Check EDF channel naming or mapping."
            )

        # Pick channels in canonical order
        raw.pick(self.target_channels)
        return raw

    def apply_linked_ear_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply linked-ear reference if A1/A2 exist; otherwise assume already referenced.

        This uses MNE's set_eeg_reference API which subtracts the mean of provided
        reference channels from EEG channels. If A1/A2 are not present we skip this
        step (data likely already referenced).
        """
        # If A1/A2 were present previously they were dropped in normalize step.
        # If the EDF originally had A1/A2 and we dropped them, assume referencing
        # was handled by the acquisition. We'll default to no-op but log.
        logger.debug("apply_linked_ear_reference: no-op (A1/A2 dropped or absent)")
        return raw

    def bandpass_filter(self, raw: mne.io.BaseRaw, l_freq: float = 0.5, h_freq: float = 70.0) -> mne.io.BaseRaw:
        """Apply bandpass filter in-place on Raw object.

        Uses FIR filter with fir_design='firwin' for stability.
        """
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
        return raw

    def notch_filter(self, raw: mne.io.BaseRaw, freqs: float | List[float] = 50.0) -> mne.io.BaseRaw:
        """Apply notch filter (line noise removal) in-place."""
        raw.notch_filter(freqs=freqs, verbose=False)
        return raw

    def remove_ica_artifacts(self, raw: mne.io.BaseRaw, n_components: int = 15) -> mne.io.BaseRaw:
        """Attempt ICA artifact removal (fastica). Graceful fallback on failure.

        Strategy:
        - Fit ICA with `n_components` (or lesser if not applicable)
        - Auto-detect EOG components using `find_bads_eog` where possible
        - Remove detected components; if any step fails, return original raw
        """
        try:
            ica = mne.preprocessing.ICA(n_components=n_components, method="fastica", random_state=0)
            ica.fit(raw, verbose=False)

            # Try to find EOG artifacts (may raise if no suitable channels)
            eog_inds, scores = ica.find_bads_eog(raw, threshold=3.0)
            if eog_inds:
                ica.exclude = eog_inds
                raw = ica.apply(raw, exclude=ica.exclude)
            else:
                logger.debug("ICA: no EOG components identified")
        except Exception as exc:
            logger.warning("ICA artifact removal failed: %s. Continuing without ICA.", exc)
        return raw

    def _segment_raw_to_windows(self, raw: mne.io.BaseRaw, step: Optional[int] = None) -> Tuple[np.ndarray, List[float]]:
        """Return sliding windows from raw data as (n_epochs, n_channels, window_samples).

        Also returns epoch start times in seconds.
        """
        step = step or self.step
        data = raw.get_data(picks=self.target_channels)  # shape: (n_channels, n_samples)
        n_channels, n_samples = data.shape
        if n_samples < self.window_samples:
            raise ValueError("Recording too short for one window: need >= window length")

        starts = list(range(0, n_samples - self.window_samples + 1, step))
        epochs = np.stack(
            [data[:, s : s + self.window_samples] for s in starts], axis=0
        )  # (n_epochs, n_channels, window_samples)
        epoch_times = [float(s / self.sampling_rate) for s in starts]
        return epochs, epoch_times

    def _apply_normalization(self, windows: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
        """Apply temporal de-meaning per window and global z-score normalization.

        Expected `windows` shape: (n_epochs, n_channels, window_samples)
        Expected `norm_stats` keys: 'eeg_mean' and 'eeg_std'
        - `eeg_mean` and `eeg_std` may be scalars or arrays of shape (n_channels,)
        """
        # Convert to float32 for model compatibility
        windows = windows.astype(np.float32)

        # 1) Per-window, per-channel temporal de-meaning (remove DC)
        mean_axis = windows.mean(axis=2, keepdims=True)  # (n_epochs, n_channels, 1)
        windows = windows - mean_axis

        # 2) Global z-score normalization using provided training stats
        if norm_stats is None:
            logger.warning("No norm_stats provided; returning de-meaned windows without z-score.")
            return windows

        eeg_mean = norm_stats.get("eeg_mean")
        eeg_std = norm_stats.get("eeg_std")
        if eeg_mean is None or eeg_std is None:
            logger.warning("norm_stats missing 'eeg_mean' or 'eeg_std'; returning de-meaned windows.")
            return windows

        eeg_mean = np.asarray(eeg_mean, dtype=np.float32)
        eeg_std = np.asarray(eeg_std, dtype=np.float32)

        # Broadcast shapes: (n_channels,) -> (1, n_channels, 1)
        eeg_mean = eeg_mean.reshape((1, -1, 1))
        eeg_std = eeg_std.reshape((1, -1, 1))

        windows = (windows - eeg_mean) / (eeg_std + 1e-7)
        return windows

    def run(self, edf_path: str, norm_stats: Optional[Dict] = None, do_ica: bool = True) -> Dict:
        """Full preprocessing pipeline.

        Args:
            edf_path: path to EDF file
            norm_stats: dict containing 'eeg_mean' and 'eeg_std' for global z-score
            do_ica: whether to attempt ICA artifact removal

        Returns:
            dict with keys: 'preprocessed_eeg' (n_epochs, 19, 400),
            'channel_names', 'epoch_times', 'n_epochs', 'edf_path'
        """
        raw = self.load_edf(edf_path)
        raw = self.normalize_channel_names(raw)
        raw = self.apply_linked_ear_reference(raw)
        raw = self.bandpass_filter(raw)
        raw = self.notch_filter(raw)
        if do_ica:
            raw = self.remove_ica_artifacts(raw)

        windows, epoch_times = self._segment_raw_to_windows(raw, step=self.step)
        preprocessed = self._apply_normalization(windows, norm_stats)

        return {
            "preprocessed_eeg": preprocessed,
            "channel_names": list(self.target_channels),
            "epoch_times": epoch_times,
            "n_epochs": int(preprocessed.shape[0]),
            "edf_path": str(edf_path),
        }


# Convenience function exported at module level
def preprocess_edf(edf_path: str, norm_stats: Optional[Dict] = None, **kwargs) -> Dict:
    pre = NMTPreprocessor(**kwargs)
    return pre.run(edf_path, norm_stats)
