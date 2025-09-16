import numpy as np

# Default configuration
RVE_DEFAULTS = {
    "core_type": "zpr",
    "mirror_symmetry": [True, True],
    "num_tiles": [1, 1],
    "chevron_angle": np.deg2rad(60.0),
    "chevron_wall_length": 0.0125,
    "chevron_thickness": 0.001,
    "chevron_pitch": 0.006,
    "rib_thickness": 0.001,
    "core_thickness": 0.011,
    "facesheet_thickness": 0.0008,
    "core_material": (396e6, 0.48),
    "facesheet_material": (22.9e6, 0.48),
    "element_type": "C3D8R",
    "element_size": 0.001,
}

FAIRING_DEFAULTS = {
    "aerofoil_name": "naca0015",
    "fairing_span": 0.8, # span of the morphing fairing
    "fairing_chord": 1.6,
    "hinge_chord_eta": 0.5,
    "num_floating_ribs": 0, # number of floating ribs in each rib bay
    "pre_strain": 0.1,
    "rotation_angle": np.deg2rad(40.0),
    "bool_isotropic": False, # [True, False], if true core properites use, else equivalent panel properties
    "model_type": "fairing", # either ["fairing", "slice"]
    "element_size": 0.020,
    "element_type": "S4R", # either of ["S4R", "B31", "C3D8R"]
    "solver": "newton" # either of ['linear', "newton", "riks", "dynamic"]
}