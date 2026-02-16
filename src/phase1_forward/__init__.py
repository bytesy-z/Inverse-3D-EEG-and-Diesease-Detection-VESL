"""
Package: src.phase1_forward
Phase: 1 — Forward Modeling and Synthetic Data Generation
Purpose: Contains all modules for building the forward model (source space,
         neural mass simulation, leadfield construction) and generating the
         synthetic EEG training dataset.

This package implements the physics-based forward pipeline:
  1. source_space.py — Load/validate TVB connectivity and parcellation
  2. parameter_sampler.py — Random parameter generation for dataset diversity
  3. epileptor_simulator.py — TVB Epileptor wrapper for batch simulation
  4. leadfield_builder.py — MNE-Python BEM forward model construction
  5. synthetic_dataset.py — Orchestrate simulation + projection + noise + HDF5

See docs/02_TECHNICAL_SPECIFICATIONS.md Sections 3.1–3.4 for full specs.
"""
