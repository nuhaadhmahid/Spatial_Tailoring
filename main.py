import sys
import os
import shutil
import traceback

sys.path.append(os.path.abspath(os.getcwd()))
import Utils
from DEFAUTLS import FAIRING_DEFAULTS
from RVE import RVE
from Fairing import FairingAnalysis, FairingData
from Tailoring import Lattice

if __name__ == "__main__":
    # Analysis case directory
    directory = Utils.Directory(case_name="test_case_4")

    # RVE definition
    panel = RVE(
        variables={
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
            "element_size": 0.001,
        },
        directory=directory,
        case_number=0
    )
    panel.analysis()

    # Fairing definition
    fairing = FairingAnalysis(
        {
            "aerofoil_name": "naca0015",
            "fairing_span": 0.8, # span of the morphing fairing
            "fairing_chord": 1.6,
            "hinge_chord_eta": 0.5,
            "num_floating_ribs": 0, # number of floating ribs in each rib bay
            "pre_strain": 0.1,
            "rotation_angle": Utils.Units.deg2rad(15.0),
            "bool_isotropic": False, # [True, False], if true core properites use, else equivalent panel properties
            "model_type": "fairing", # either ["fairing", "slice"]
            "element_size": 0.020,
            "element_type": "S4R", # either of ["S4R", "B31", "C3D8R"]
            "solver": "newton" # either of ['linear', "newton", "riks", "dynamic"]
        }, 
        RVE_identifier=panel.case_number, 
        directory=directory, 
        case_number=0
    )
    fairing.analysis()

    # # Lattice definition
    # traced_lines = Lattice(directory, 0)
    # traced_lines.analysis()

    #