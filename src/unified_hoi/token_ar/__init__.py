"""Independent masked-task autoregressive HOI model; never imports parametric models."""

REPRESENTATION = "parametric_v1"  # Shared, read-only native data format.
ARCHITECTURE = "token_ar_v1"       # Incompatible with diffusion and contact-AE checkpoints.
MODALITIES = ("human", "object", "contact")
# H/O/C with 1=generate. Order is the mode-token order: text-to-HOI, object-to-human, human-to-object.
MODES = ("111", "101", "011")
