#!/usr/bin/env python3
"""
Lightweight test to exercise /api/analyze with a synthetic EEG data file.
This script relies on the 'requests' library being available. If not installed,
it will exit with a friendly message prompting installation.
"""
import os
import sys
import json
import subprocess

EEG_PATH = "/tmp/synthetic_eeg.npy"

def ensure_synthetic_eeg():
    if os.path.exists(EEG_PATH):
        return EEG_PATH
    try:
        import numpy as np
    except Exception:
        print("NumPy is required to generate a synthetic EEG file. Install NumPy and re-run.")
        sys.exit(3)
    arr = (np.random.randn(400)).astype('float32')
    np.save(EEG_PATH, arr)
    return EEG_PATH

def main():
    path = ensure_synthetic_eeg()
    try:
        import requests
        resp = requests.post(
            'http://127.0.0.1:8000/api/analyze',
            files={'file': open(path, 'rb')},
            data={'mode': 'source_localization'}
        )
        status = resp.status_code
        content = resp.text
        if status != 200:
            print(f"Error: HTTP {status} from /api/analyze. Response: {content}")
            sys.exit(2)
        data = resp.json()
    except Exception as e:
        print(f"Failed to run /api/analyze with requests: {e}")
        print("Falling back to raw curl via subprocess for robustness...")
        try:
            curl = [
                'curl', '-sS', '-X', 'POST', 'http://127.0.0.1:8000/api/analyze',
                '-F', f'file=@{path}', '-F', 'mode=source_localization'
            ]
            p = subprocess.run(curl, capture_output=True, text=True, check=True)
            data = json.loads(p.stdout)
        except Exception as e2:
            print(f"Fallback analyze request failed: {e2}")
            sys.exit(4)

    # Validate response content
    if 'eegData' not in data:
        print("Test failed: 'eegData' not found in response.")
        sys.exit(5)
    waveform_keys = ['waveform','waveformImage','image','plot','waveform_data']
    if not any(k in data for k in waveform_keys):
        print("Test warning: No waveform key found in response, but eegData is present.")
    else:
        print("Test passed: eegData present and waveform key found in response.")
    print("Response keys:", list(data.keys()))

if __name__ == '__main__':
    main()
