# EEG Brain Activity Visualization Frontend

## Overview

This Next.js application provides a web interface for analyzing EEG recordings and visualizing estimated brain source activity in real-time. The system processes EEG data through a trained 1D-CNN model to estimate cortical source activity, then presents the results as an interactive, animated 3D brain visualization.

## System Architecture

### High-Level Flow

1. **File Upload**: User selects an EDF (European Data Format) file containing EEG recordings through the web interface

2. **Server Processing**: The Next.js API route receives the file, saves it to disk, and spawns a Python subprocess to perform the analysis

3. **Python Inference Pipeline**: 
   - Loads the pre-trained 1D-CNN model and head model (leadfield matrix, source space, electrode montage)
   - Preprocesses the EEG data (channel alignment, resampling, average referencing, interpolation)
   - Segments the entire recording into overlapping windows (default: 1-second windows with 50% overlap)
   - Runs inference on each window to estimate source activity at 994 cortical locations
   - Interpolates predictions across overlapping windows using weighted averaging to create smooth temporal transitions
   - Generates animation data containing source activity across all timeframes

4. **Visualization Generation**: 
   - Creates a 3D brain mesh using fsaverage source space geometry
   - Maps estimated activity to source positions using a color scale
   - Produces an interactive, playable animation showing brain activity evolution over time

5. **Results Display**: The frontend renders the 3D brain visualization with playback controls allowing users to play, pause, and scrub through the temporal evolution of brain activity

### Key Components

**Frontend (Next.js/React/TypeScript)**:
- `app/page.tsx` - Main application layout coordinating all components
- `components/file-upload-section.tsx` - EDF file selection and upload interface
- `components/processing-window.tsx` - Real-time processing status display
- `components/animated-brain-visualization.tsx` - Three.js-based animated 3D brain renderer with playback controls
- `app/api/analyze-eeg/route.ts` - API endpoint handling file upload and Python script orchestration

**Backend (Python)**:
- `inverse_problem/run_edf_inference.py` - Main inference pipeline coordinating all processing steps
- `inverse_problem/models/cnn_1d.py` - 1D-CNN model architecture for source estimation
- `inverse_problem/load_data/HeadModel.py` - Head model loading (leadfield, electrodes, sources)
- `inverse_problem/utils/` - Utility functions for preprocessing, metrics, and visualization

### Window Processing Details

**Sliding Window Approach**:
- **Window Duration**: 1.0 second (500 samples at 500 Hz sampling rate)
- **Overlap**: 50% (0.5 seconds overlap between consecutive windows)
- **Step Size**: 0.5 seconds (moves window forward by half its duration)
- **Padding**: Last incomplete window is edge-padded to full length

**Example Timeline** (for a 10-second recording):
```
Window 1: [0.0s - 1.0s]
Window 2: [0.5s - 1.5s]  ← 50% overlap with Window 1
Window 3: [1.0s - 2.0s]  ← 50% overlap with Window 2
...
Window 19: [9.0s - 10.0s]
```

**Interpolation Strategy**:
- Overlapping predictions are combined using Gaussian-weighted averaging
- Weights are highest at window centers, decreasing toward edges
- This produces smooth temporal transitions without artifacts at window boundaries

### Output Format

The system generates:
- **Animation Data** (`animation_data.npz`): Compressed NumPy archive containing:
  - Source activity matrix: `(n_sources=994, n_frames)` 
  - Timestamps for each frame
  - 3D source positions on cortical surface
  - Triangle mesh indices for brain geometry
  - Framerate specification (default: 30 FPS)

- **Metadata** (`best_window_summary.json`): Processing parameters and statistics

- **Intermediate Files** (`segments/`): Individual window predictions saved as `.mat` files for debugging

## Running the Application

### Prerequisites

**Node.js Environment**:
- Node.js 18+ and pnpm package manager
- All frontend dependencies (installed via `pnpm install`)

**Python Environment**:
- Python 3.9+
- Required packages: `numpy`, `scipy`, `torch`, `mne`, `plotly`, `pyyaml`
- Pre-trained 1D-CNN model checkpoint in `results/` directory
- fsaverage head model assets in `simulation/fsaverage/` directory

### Development Server

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies** (first time only):
   ```bash
   pnpm install
   ```

3. **Start the development server**:
   ```bash
   pnpm dev
   ```

4. **Open the application**:
   Navigate to `http://localhost:3000` in your web browser

### Production Build

1. **Build the application**:
   ```bash
   pnpm build
   ```

2. **Start the production server**:
   ```bash
   pnpm start
   ```

## Usage

1. **Upload EEG File**: Click "Choose File" and select an EDF file from your system (sample file available at `sample/0001082.edf`)

2. **Configure Analysis** (optional): Default settings use 1-second windows with 50% overlap, trained on the standard 10-20 electrode montage with ico3 source space

3. **Start Analysis**: Click "Analyze EEG" to begin processing

4. **Monitor Progress**: Watch the processing window for real-time status updates

5. **View Results**: Once complete, the animated 3D brain visualization appears with:
   - Play/Pause button for animation control
   - Timeline slider for manual scrubbing
   - Current timestamp display
   - Interactive 3D rotation/zoom (click and drag to rotate, scroll to zoom)

6. **Interpret Activity**: 
   - Warm colors (yellow/orange/red) indicate higher estimated source activity
   - Cool colors (dark purple/black) indicate lower activity
   - Activity patterns evolve over time showing dynamic brain processes

## Technical Details

### Model Architecture
- **Type**: 1D Convolutional Neural Network
- **Input**: Normalized EEG sensor data (75 channels × 500 timepoints)
- **Output**: Estimated source activity (994 cortical sources × 500 timepoints)
- **Architecture**: Conv1D [75 → 4096 → 994], kernel size 5, no bias
- **Training Loss**: Cosine similarity between predicted and true source activity

### Head Model
- **Subject**: fsaverage (standard brain template)
- **Source Space**: ico3 (642 vertices per hemisphere, 1284 total → 994 after orientation constraint)
- **Electrode Montage**: Standard 10-20 system (75 electrodes)
- **Forward Model**: Boundary Element Method (BEM) leadfield matrix
- **Orientation**: Constrained to cortical surface normals

### Performance Considerations
- **Processing Time**: ~30-60 seconds for a 60-second EEG recording (depends on CPU)
- **Memory Usage**: ~2GB for model loading and inference
- **Animation File Size**: ~5-10MB for a 60-second recording at 30 FPS
- **Browser Requirements**: WebGL support required for 3D visualization

## Troubleshooting

**"Model checkpoint not found"**: Ensure trained model exists in `results/` directory with matching simulation parameters

**"Channel names don't match"**: EDF file must contain standard EEG channel names (e.g., Fp1, Fp2, F3, F4, etc.)

**"Python script failed"**: Check that all Python dependencies are installed and the Python environment is activated

**"Visualization not loading"**: Verify browser supports WebGL and animation data file was generated successfully

**"Slow performance"**: Try reducing overlap fraction or limiting max windows for faster processing

## Future Enhancements

- Real-time streaming EEG support
- Multiple simultaneous visualizations for comparison
- Downloadable animations as video files
- Region-of-interest selection and activity quantification
- Statistical significance testing and confidence intervals
- Support for custom electrode montages and source spaces
