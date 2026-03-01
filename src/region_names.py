"""
Region Name Mapping for Desikan-Killiany 76-Region Atlas

Maps brief region codes (like "rCCA", "lPFCM") to full anatomical names for clinical display.
Useful for converting model output brain regions into human-readable labels for neurologists.

Each region is mapped to its full anatomical description and hemisphere designation.
"""

# Comprehensive mapping from DK76 region codes to full anatomical names
REGION_CODE_TO_NAME = {
    # Right Hemisphere - Auditory/Speech
    "rA1": "Right Auditory Cortex Area 1",
    "rA2": "Right Auditory Cortex Area 2",
    
    # Right Hemisphere - Limbic/Memory
    "rAMYG": "Right Amygdala",
    "rHC": "Right Hippocampus",
    "rPHC": "Right Parahippocampal Cortex",
    
    # Right Hemisphere - Cingulate
    "rCCA": "Right Cingulate Cortex Anterior",
    "rCCP": "Right Cingulate Cortex Posterior",
    "rCCR": "Right Cingulate Cortex Rostral",
    "rCCS": "Right Cingulate Cortex Subgenual",
    
    # Right Hemisphere - Posterior Cingulate
    "rPCI": "Right Posterior Cingulate Interior",
    "rPCIP": "Right Posterior Cingulate Inferior Parietal",
    "rPCM": "Right Posterior Cingulate Middle",
    "rPCS": "Right Posterior Cingulate Superior",
    
    # Right Hemisphere - Frontal
    "rFEF": "Right Frontal Eye Field",
    "rM1": "Right Primary Motor Cortex",
    "rPMCDL": "Right Premotor Cortex Dorso-Lateral",
    "rPMCM": "Right Premotor Cortex Medial",
    "rPMCVL": "Right Premotor Cortex Ventro-Lateral",
    
    # Right Hemisphere - Prefrontal Cortex
    "rPFCCL": "Right Prefrontal Cortex Central-Lateral",
    "rPFCDL": "Right Prefrontal Cortex Dorso-Lateral",
    "rPFCDM": "Right Prefrontal Cortex Dorso-Medial",
    "rPFCM": "Right Prefrontal Cortex Medial",
    "rPFCORB": "Right Prefrontal Cortex Orbital",
    "rPFCPOL": "Right Prefrontal Cortex Polar",
    "rPFCVL": "Right Prefrontal Cortex Ventro-Lateral",
    
    # Right Hemisphere - Somatosensory
    "rS1": "Right Primary Somatosensory Cortex",
    "rS2": "Right Secondary Somatosensory Cortex",
    
    # Right Hemisphere - Insula
    "rIA": "Right Insula Anterior",
    "rIP": "Right Insula Posterior",
    
    # Right Hemisphere - Parietal
    "rG": "Right Cuneus",
    
    # Right Hemisphere - Temporal
    "rTCC": "Right Temporal Cortex Central",
    "rTCI": "Right Temporal Cortex Inferior",
    "rTCPOL": "Right Temporal Cortex Polar",
    "rTCS": "Right Temporal Cortex Superior",
    "rTCV": "Right Temporal Cortex Ventral",
    
    # Right Hemisphere - Visual
    "rV1": "Right Visual Cortex Area 1 (V1)",
    "rV2": "Right Visual Cortex Area 2 (V2)",
    
    # Right Hemisphere - Midline
    "rCC": "Right Corpus Callosum",
    
    # ============================================================================
    
    # Left Hemisphere - Auditory/Speech
    "lA1": "Left Auditory Cortex Area 1",
    "lA2": "Left Auditory Cortex Area 2",
    
    # Left Hemisphere - Limbic/Memory
    "lAMYG": "Left Amygdala",
    "lHC": "Left Hippocampus",
    "lPHC": "Left Parahippocampal Cortex",
    
    # Left Hemisphere - Cingulate
    "lCCA": "Left Cingulate Cortex Anterior",
    "lCCP": "Left Cingulate Cortex Posterior",
    "lCCR": "Left Cingulate Cortex Rostral",
    "lCCS": "Left Cingulate Cortex Subgenual",
    
    # Left Hemisphere - Posterior Cingulate
    "lPCI": "Left Posterior Cingulate Interior",
    "lPCIP": "Left Posterior Cingulate Inferior Parietal",
    "lPCM": "Left Posterior Cingulate Middle",
    "lPCS": "Left Posterior Cingulate Superior",
    
    # Left Hemisphere - Frontal
    "lFEF": "Left Frontal Eye Field",
    "lM1": "Left Primary Motor Cortex",
    "lPMCDL": "Left Premotor Cortex Dorso-Lateral",
    "lPMCM": "Left Premotor Cortex Medial",
    "lPMCVL": "Left Premotor Cortex Ventro-Lateral",
    
    # Left Hemisphere - Prefrontal Cortex
    "lPFCCL": "Left Prefrontal Cortex Central-Lateral",
    "lPFCDL": "Left Prefrontal Cortex Dorso-Lateral",
    "lPFCDM": "Left Prefrontal Cortex Dorso-Medial",
    "lPFCM": "Left Prefrontal Cortex Medial",
    "lPFCORB": "Left Prefrontal Cortex Orbital",
    "lPFCPOL": "Left Prefrontal Cortex Polar",
    "lPFCVL": "Left Prefrontal Cortex Ventro-Lateral",
    
    # Left Hemisphere - Somatosensory
    "lS1": "Left Primary Somatosensory Cortex",
    "lS2": "Left Secondary Somatosensory Cortex",
    
    # Left Hemisphere - Insula
    "lIA": "Left Insula Anterior",
    "lIP": "Left Insula Posterior",
    
    # Left Hemisphere - Parietal
    "lG": "Left Cuneus",
    
    # Left Hemisphere - Temporal
    "lTCC": "Left Temporal Cortex Central",
    "lTCI": "Left Temporal Cortex Inferior",
    "lTCPOL": "Left Temporal Cortex Polar",
    "lTCS": "Left Temporal Cortex Superior",
    "lTCV": "Left Temporal Cortex Ventral",
    
    # Left Hemisphere - Visual
    "lV1": "Left Visual Cortex Area 1 (V1)",
    "lV2": "Left Visual Cortex Area 2 (V2)",
    
    # Left Hemisphere - Midline
    "lCC": "Left Corpus Callosum",
}


def get_region_name(region_code: str) -> str:
    """
    Get the full anatomical name for a region code.
    
    Args:
        region_code: Brief region code (e.g., "rCCA", "lPFCM")
    
    Returns:
        Full anatomical name (e.g., "Right Cingulate Cortex Anterior")
        If code not found, returns the code itself as fallback.
    """
    name = REGION_CODE_TO_NAME.get(region_code, region_code)
    return name


def format_region_for_display(region_code: str) -> str:
    """
    Format a region code and name for UI display.
    
    Args:
        region_code: Brief region code
    
    Returns:
        Formatted string like "rCCA (Right Cingulate Cortex Anterior)"
    """
    full_name = get_region_name(region_code)
    if full_name == region_code:
        # Unknown region, just return the code
        return region_code
    return f"{region_code} ({full_name})"


def get_all_region_codes() -> list:
    """Get list of all region codes."""
    return list(REGION_CODE_TO_NAME.keys())
