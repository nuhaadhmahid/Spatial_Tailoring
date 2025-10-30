import Utils

# Default configuration
RVE_DEFAULTS = {
    "core_type": "zpr",
    "mirror_symmetry": [True, False],
    "num_tiles": [1, 1],
    "chevron_angle": Utils.Units.deg2rad(60.0),
    "chevron_wall_length": Utils.Units.mm2m(40.0),
    "chevron_thickness": Utils.Units.mm2m(1.5),
    "chevron_pitch": Utils.Units.mm2m(10.0),
    "rib_thickness": Utils.Units.mm2m(3.0),
    "core_thickness": Utils.Units.mm2m(15.0),
    "facesheet_thickness": Utils.Units.mm2m(0.5),
    "core_material": (396e6, 0.48),
    "facesheet_material": (22.9e6, 0.48),
    "element_type": "C3D8R", 
    "element_size": 0.001
}

FAIRING_DEFAULTS = {
    "aerofoil_name": "naca0015",
    "fairing_span": 0.8, # span of the morphing fairing
    "fairing_chord": 1.6,
    "hinge_chord_eta": 0.5,
    "num_floating_ribs": 0, # number of floating ribs in each rib bay
    "pre_strain": 0.1,
    "rotation_angle": Utils.Units.deg2rad(15.0),
    "model_type": "fairing", # either ["fairing", "slice"]
    "element_size": 0.020,
    "model_fidelity": "equivalent", # either of ["equivalent", "explicit", "fullscale"]
    "model_fidelity_settings":{
        "equivalent":{
            "bool_isotropic": False, # [True, False], if true core properites use, else equivalent panel properties
        },
        "explicit":{
            "reference_case": 0, # int, case number of the reference case from which the explicit model is generated
            "reference_field": 15, # int, rotation angle for the folding wingtip from whose deformation the field is extracted
        },
    },
    "solver": "newton" # either of ['linear', "newton", "riks", "dynamic"]
}