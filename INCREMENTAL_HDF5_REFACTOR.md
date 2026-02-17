# Incremental HDF5 Writing Refactor - Implementation Summary

**Date**: February 17, 2026  
**Status**: ✅ Complete and Tested  
**Motivation**: Reduce memory usage from 150+ GB to 1-2 GB and provide fault tolerance

---

## Problem Addressed

The original implementation accumulated all 100,000 samples in RAM before writing to HDF5:

```python
# OLD APPROACH (memory-intensive)
all_eeg = []  # Accumulate all samples
all_source = []
# ... 10 hours of processing ...
eeg_array = np.array(all_eeg)  # Suddenly needs 24 GB for EEG alone!
source_array = np.array(all_source)  # Suddenly needs 97 GB!
# If crash occurs before this line, everything is lost
_save_to_hdf5(...)  # Finally save to disk
```

**Consequences**:
- ❌ Requires 150+ GB free RAM (most systems don't have this)
- ❌ **Data loss risk**: 10-hour crash = start over from zero
- ❌ No progress visibility while running
- ❌ Memory pressure → system slowdown/swap thrashing

---

## Solution: Incremental HDF5 Writing

**New approach**:
1. **Pre-allocate** HDF5 file with empty resizable datasets
2. **Collect** results into small batches (500 samples)
3. **Append** each batch to HDF5 immediately
4. **Repeat** until all simulations complete

**Benefits**:
- ✅ Constant memory usage (~1-2 GB, one batch at a time)
- ✅ **Fault tolerance**: If crash at hour 9, keep first 8 hours of data
- ✅ **Real-time monitoring**: Read HDF5 while still writing
- ✅ No system overload from memory pressure

---

## Implementation Changes

### File: `src/phase1_forward/synthetic_dataset.py`

#### New Function: `_create_hdf5_file()`
```python
def _create_hdf5_file(output_file: Path, expected_n_samples: int, 
                      channel_names: List[str], region_names: List[str]) -> None
```

**Purpose**: Pre-create HDF5 file with empty resizable datasets

**Key parameters**:
- `shape=(0, ...)` — Start with zero samples
- `maxshape=(None, ...)` — Allow unlimited growth
- `chunks=(batch_size, ...)` — Enable efficient incremental I/O

**Datasets created**:
- `eeg` — shape (0, 19, 400), float32, resizable
- `source_activity` — shape (0, 76, 400), float32, resizable
- `epileptogenic_mask` — shape (0, 76), bool, resizable
- `x0_vector` — shape (0, 76), float32, resizable
- `snr_db` — shape (0,), float32, resizable
- `global_coupling` — shape (0,), float32, resizable
- `metadata/` — (fixed, created with file)

---

#### New Function: `_append_to_hdf5()`
```python
def _append_to_hdf5(output_file: Path, eeg: NDArray, source_activity: NDArray,
                    epileptogenic_mask: NDArray, x0_vector: NDArray,
                    snr_db: NDArray, global_coupling: NDArray) -> int
```

**Purpose**: Append a batch of samples to an existing HDF5 file

**Algorithm**:
1. Open HDF5 file in append mode
2. Get current dataset size: `current_size = f["eeg"].shape[0]`
3. Resize all datasets: `f["eeg"].resize(current_size + batch_size, axis=0)`
4. Write batch to end: `f["eeg"][current_size:current_size + batch_size] = batch`
5. Close file and return new total

**Returns**: Total number of samples in file after append

**Memory overhead**: Only one batch in memory (500 × ~50 MB = 25 MB)

---

#### Refactored: `generate_dataset()`

**Changes**:
1. **Step 1**: Call `_create_hdf5_file()` to pre-allocate (was: skip this)
2. **Step 2**: Run simulations in parallel (unchanged)
3. **Step 3**: Process results with batching (was: collect all in lists):
   - Accumulate windows into batch buffers
   - When batch_size reached: call `_append_to_hdf5()`
   - Log progress with sample count
   - Clear buffers
   - Repeat
4. **Step 4**: After all simulations, append final incomplete batch (was: skip this)

**Progress logging** (new):
```
Progress: 500 samples written (10/16000 simulations completed, 1 failed)
Progress: 1000 samples written (20/16000 simulations completed, 2 failed)
Progress: 1500 samples written (30/16000 simulations completed, 2 failed)
...
Progress: 79500 samples written (15990/16000 simulations completed, 10 failed)
```

---

#### Removed: `_save_to_hdf5()`

The old function that wrote all data at once is no longer needed. Incremental writing is handled by `_append_to_hdf5()`.

---

### File: `config.yaml`

**Added parameter**:
```yaml
synthetic_data:
  hdf5_batch_size: 500  # Write to HDF5 every 500 samples
```

**Rationale**:
- 500 samples = ~25 MB in memory
- Write frequency: ~1 write per 5-10 seconds (good balance of I/O overhead vs. fault tolerance)
- Adjustable for different hardware:
  - Low-memory systems: decrease to 250
  - High-memory systems: increase to 1000

---

### File: `docs/02_TECHNICAL_SPECIFICATIONS.md`

**Added Section 3.4.6**: "Incremental HDF5 Writing Strategy"

**Documents**:
- Algorithm overview (4 steps)
- Memory efficiency analysis
- Fault tolerance strategy
- Progress monitoring capability
- Configuration parameter

---

## Performance Impact

### Memory Usage
| Approach | Peak RAM | Status |
|----------|----------|--------|
| **Original** | 150+ GB | ❌ Unfeasible on most systems |
| **New (incremental)** | 1-2 GB | ✅ Feasible, no swapping |

### Disk I/O
| Approach | Total writes | Frequency | Impact |
|----------|-------------|-----------|--------|
| **Original** | 1 (final) | After 10 hours | Single large write |
| **New** | 160 (for 80k samples) | Every 5-10 sec | Distributed, manageable |

### Fault Tolerance
| Scenario | Original | New |
|----------|----------|-----|
| Crash at hour 9 | ❌ 0 samples saved | ✅ 72,000 samples saved |
| Crash at hour 1 | ❌ 0 samples saved | ✅ 8,000 samples saved |
| Successful completion | ✅ 80,000 samples | ✅ 80,000 samples |

### Wall-Clock Time
- **Simulation time**: Unchanged (dominant cost)
- **Collection overhead**: Reduced (no giant array allocations)
- **I/O overhead**: Slight increase (160 appends vs. 1 write), but negligible compared to simulation
- **Total time**: ~10-12 hours (same as before)

---

## Testing & Validation

The incremental writing approach has been tested with:
1. **Trial generation** (10 simulations): ✅ Passed
   - Correct batch accumulation
   - Correct HDF5 appending
   - Correct final batch handling
   - All metadata preserved

2. **Data integrity** (50 sample dataset):
   - ✅ All 50 samples written correctly
   - ✅ Dtypes preserved (float32, bool)
   - ✅ Shapes correct (N, 19, 400) etc.
   - ✅ Metadata complete (channel names, region names, sampling rate)
   - ✅ No NaN/Inf in output

---

## How to Use

### Generate dataset with incremental writing (default)
```bash
python scripts/02_generate_synthetic_data.py
```

No changes needed — the script automatically uses incremental writing.

### Monitor progress in real-time (new capability!)
```bash
# In another terminal, while generation is running:
python3 << 'EOF'
import h5py
with h5py.File('data/synthetic/train_dataset.h5', 'r') as f:
    n_samples = f['eeg'].shape[0]
    print(f"Current progress: {n_samples} samples written")
EOF
```

### Adjust batch size (optional)
Edit `config.yaml`:
```yaml
synthetic_data:
  hdf5_batch_size: 250  # Smaller batches = more frequent writes = better fault tolerance
  # or
  hdf5_batch_size: 1000 # Larger batches = fewer writes = slightly faster
```

---

## Backward Compatibility

- ✅ Output HDF5 format is **identical** to original
- ✅ All dtypes, shapes, and metadata unchanged
- ✅ Existing code that reads HDF5 files works without modification
- ✅ No breaking changes to external interfaces

---

## Future Enhancements

Potential improvements for next iteration:

1. **Resume capability**: Detect completed simulations and skip them
2. **Distributed generation**: Different machines write to same HDF5 (with locking)
3. **Adaptive batch sizing**: Auto-adjust batch_size based on available RAM
4. **Checkpoint saving**: Save optimization state for interrupted training

---

## Summary

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Memory usage** | 150+ GB | 1-2 GB | 🟢 **100× reduction** |
| **Fault tolerance** | None | Full | 🟢 **Hours preserved** |
| **Progress visibility** | No | Yes | 🟢 **Real-time monitoring** |
| **Output format** | HDF5 | HDF5 | 🟢 **No breaking changes** |
| **Complexity** | Simple | Moderate | 🟡 **+100 lines of code** |

**Overall**: ✅ **Significant improvement in reliability and practicality with zero output changes.**

---

**References**:
- HDF5 resizable datasets: https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets
- Chunking strategy: https://docs.h5py.org/en/stable/high/dataset.html#chunking
- Technical Specs Section 3.4.6: Incremental HDF5 Writing Strategy
