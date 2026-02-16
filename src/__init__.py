"""
Package: src
Purpose: Top-level package for the PhysDeepSIF pipeline.

This package contains all source code for the PhysDeepSIF epileptogenicity
mapping pipeline, organized into five phases:
  - phase1_forward: Forward modeling and synthetic data generation
  - phase2_network: PhysDeepSIF neural network architecture and training
  - phase3_inference: Real EEG preprocessing and inference
  - phase4_inversion: Patient-specific parameter inversion (CMA-ES)
  - phase5_validation: Validation framework and baseline comparisons
"""
