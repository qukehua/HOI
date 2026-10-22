"""Independent masked-task autoregressive HOI model; never imports parametric models."""

REPRESENTATION = "parametric_v1"  # Shared, read-only native data format.
ARCHITECTURE = "token_ar_v1"       # Incompatible with diffusion and contact-AE checkpoints.
MODALITIES = ("human", "object", "contact")
MODES = ("100", "010", "001", "110", "101", "011", "111")
