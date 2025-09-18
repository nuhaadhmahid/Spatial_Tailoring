#%%
import sys
import os
import shutil
import traceback

sys.path.append(os.path.abspath(os.getcwd()))
import Utils
from DEFAUTLS import FAIRING_DEFAULTS
from RVE import RVE
from Fairing import FairingAnalysis, FairingData
from Tailoring import Lattice, Tailored

if __name__ == "__main__":
    # Analysis case directory
    directory = Utils.Directory(case_name="test_case_4")
    case_number = 0

    #%%
    # RVE definition
    panel = RVE(
        directory=directory,
        case_number=case_number
    )
    panel.analysis()

    #%%
    # Fairing definition
    fairing = FairingAnalysis(
        RVE_identifier=panel.case_number, 
        directory=directory, 
        case_number=case_number 
    )
    fairing.analysis()

    #%%
    # Lattice definition
    lattice_data = Lattice(directory, case_number)
    lattice_data.analysis()

    #%%
    # Generate field-aligned lattice 
    tailored = Tailored(directory, case_number, lattice_data=lattice_data)
    tailored.create_fairing_2D(bool_plot=True)

    # 2D Mesh Plot
    increment_key = "15"
    tailored.mesh2D[increment_key].plot_2D(
        {"group_sets": [], "triangle_sets": []},
        False,
        os.path.join(directory.case_folder, "fig", f"{case_number}_initial_2D_{increment_key}.svg"),
        True
    )

    # Mapping lattice to 3D wing
    tailored.mapping_2D_to_3D(bool_plot=True)
    tailored.validate_midplane_mesh()

    #%%
    # 3D Mesh Plot
    increment_key = "15"
    tailored.mesh3D[increment_key].plot_3D(
        {"beam_sets": [], "triangle_sets": []},
        False,
        os.path.join(directory.case_folder, "fig", f"{case_number}_tailored_3D_{increment_key}.svg"),
        True
    )