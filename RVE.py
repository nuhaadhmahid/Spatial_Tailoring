import Utils
from DEFAUTLS import RVE_DEFAULTS

import os
import shutil
import traceback
from itertools import product
import gmsh
import numpy as np

class RVE:
    def __init__(self, variables=None, directory=None, case_number: int = 0):
        # directory setup
        if directory is None:
            directory = Utils.Directory()

        # loading default variables
        self.var = RVE_DEFAULTS.copy()
        if variables:
            self.var.update(variables)

        # saving independent variables
        Utils.ReadWriteOps.save_object(
            self.var,
            os.path.join(directory.case_folder, "input", f"{case_number}_UC"),
            method="json",
        )

        # directory and case number
        self.directory = directory
        self.case_number = case_number

    def ppr_core_geometry(self):
        """
        Defines the core geometry for a PPR core by calculating node coordinates,
        line connectivity, surface connectivity, and core end surface box based on input parameters.
        Functionality:
            - Computes node coordinates in `self.derived_var["coordinates"]` as numpy arrays.
            - Defines line connectivity between nodes in `self.derived_var["line_connectivity"]`.
            - Specifies surface connectivity for solid and facesheet surfaces in
                `self.derived_var["solid_surface_connectivity"]` and
                `self.derived_var["facesheet_surfaces_connectivity"]`.
            - Sets the core end surface box for extrusion in `self.derived_var["core_end_surface_box"]`.
        Modifies:
            self.derived_var: Updates with calculated coordinates, connectivity, and surface box information.
        Dependencies:
            - Assumes `self.var` and `self.derived_var` are dictionaries with required geometric parameters.
        """

        # Node coordinates
        self.derived_var["coordinates"] = {
            "1": np.array([-self.derived_var["qlx"], 0.0, -self.derived_var["qlz"]]),
            "2": np.array(
                [
                    -self.derived_var["qlx"],
                    -0.5 * self.var["chevron_pitch"]
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "3": np.array(
                [
                    -self.derived_var["qlx"],
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "4": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "5": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    0.0,
                    -self.derived_var["qlz"],
                ]
            ),
            "6": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "7": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "8": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "9": np.array(
                [-self.var["rib_thickness"] / 2.0, 0.0, -self.derived_var["qlz"]]
            ),
            "10": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "11": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "12": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "13": np.array([0.0, 0.0, -self.derived_var["qlz"]]),
            "14": np.array(
                [
                    0.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "15": np.array(
                [
                    0.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "16": np.array([0.0, -self.derived_var["qly"], -self.derived_var["qlz"]]),
        }

        # Line connectivity
        self.derived_var["line_connectivity"] = [
            [[1, 5], [2, 6], [3, 7], [4, 8]],  # horizontal, left to right, column 1
            [[5, 9], [6, 10], [7, 11], [8, 12]],  # horizontal, left to right, column 2
            [
                [9, 13],
                [10, 14],
                [11, 15],
                [12, 16],
            ],  # horizontal, left to right, column 3
            [[1, 2], [5, 6], [9, 10], [13, 14]],  # vertical, top to bottom, row 1
            [[2, 3], [6, 7], [10, 11], [14, 15]],  # vertical, top to bottom, row 2
            [[3, 4], [7, 8], [11, 12], [15, 16]],  # vertical, top to bottom, row 3
        ]

        # Solid surface connectivity
        self.derived_var["solid_surface_connectivity"] = [
            [[1, "1_5"], [1, "5_6"], [-1, "2_6"], [-1, "1_2"]],
            [[1, "2_6"], [1, "6_7"], [-1, "3_7"], [-1, "2_3"]],
            [[1, "6_10"], [1, "10_11"], [-1, "7_11"], [-1, "6_7"]],
            [[1, "10_14"], [1, "14_15"], [-1, "11_15"], [-1, "10_11"]],
            [[1, "11_15"], [1, "15_16"], [-1, "12_16"], [-1, "11_12"]],
        ]

        # Facesheet surface connectivity
        self.derived_var["facesheet_surfaces_connectivity"] = [
            [[1, "5_9"], [1, "9_10"], [-1, "6_10"], [-1, "5_6"]],
            [[1, "9_13"], [1, "13_14"], [-1, "10_14"], [-1, "9_10"]],
            [[1, "3_7"], [1, "7_8"], [-1, "4_8"], [-1, "3_4"]],
            [[1, "7_11"], [1, "11_12"], [-1, "8_12"], [-1, "7_8"]],
        ]

        # Core end surface box for extrusion
        self.derived_var["core_end_surface_box"] = [
            [[3, 7], [5, 5]],
            [[7, 11], [10, 6]],
            [[12, 12], [14, 10]],
        ]

    def npr_core_geometry(self):
        """
        Defines the core geometry for an NPR core by calculating node coordinates,
        line connectivity, surface connectivity, and core end surface box based on input parameters.
        Functionality:
            - Computes node coordinates in `self.derived_var["coordinates"]` as numpy arrays.
            - Defines line connectivity between nodes in `self.derived_var["line_connectivity"]`.
            - Specifies surface connectivity for solid and facesheet surfaces in
                `self.derived_var["solid_surface_connectivity"]` and
                `self.derived_var["facesheet_surfaces_connectivity"]`.
            - Sets the core end surface box for extrusion in `self.derived_var["core_end_surface_box"]`.
        Modifies:
            self.derived_var: Updates with calculated coordinates, connectivity, and surface box information.
        Dependencies:
            - Assumes `self.var` and `self.derived_var` are dictionaries with required geometric parameters.
        """

        # Node coordinates
        self.derived_var["coordinates"] = {
            "1": np.array([-self.derived_var["qlx"], 0.0, -self.derived_var["qlz"]]),
            "2": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "3": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "4": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "5": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    0.0,
                    -self.derived_var["qlz"],
                ]
            ),
            "6": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "7": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "8": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "9": np.array(
                [-self.var["rib_thickness"] / 2.0, 0.0, -self.derived_var["qlz"]]
            ),
            "10": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "11": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["rib_thickness"]
                    / 2.0
                    * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "12": np.array(
                [
                    -self.var["rib_thickness"] / 2.0,
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "13": np.array([0.0, 0.0, -self.derived_var["qlz"]]),
            "14": np.array(
                [
                    0.0,
                    -0.5 * self.var["chevron_pitch"]
                    + self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "15": np.array(
                [
                    0.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["chevron_thickness"]
                    / 2.0
                    * 1.0
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "16": np.array([0.0, -self.derived_var["qly"], -self.derived_var["qlz"]]),
        }

        # Line connectivity
        self.derived_var["line_connectivity"] = [
            [[1, 5], [2, 6], [3, 7], [4, 8]],  # horizontal, left to right, column 1
            [[5, 9], [6, 10], [7, 11], [8, 12]],  # horizontal, left to right, column 2
            [
                [9, 13],
                [10, 14],
                [11, 15],
                [12, 16],
            ],  # horizontal, left to right, column 3
            [[1, 2], [5, 6], [9, 10], [13, 14]],  # vertical, top to bottom, row 1
            [[2, 3], [6, 7], [10, 11], [14, 15]],  # vertical, top to bottom, row 2
            [[3, 4], [7, 8], [11, 12], [15, 16]],  # vertical, top to bottom, row 3
        ]

        # Solid surface connectivity
        self.derived_var["solid_surface_connectivity"] = [
            [[1, "1_5"], [1, "5_6"], [-1, "2_6"], [-1, "1_2"]],
            [[1, "2_6"], [1, "6_7"], [-1, "3_7"], [-1, "2_3"]],
            [[1, "6_10"], [1, "10_11"], [-1, "7_11"], [-1, "6_7"]],
            [[1, "10_14"], [1, "14_15"], [-1, "11_15"], [-1, "10_11"]],
            [[1, "11_15"], [1, "15_16"], [-1, "12_16"], [-1, "11_12"]],
        ]

        # Facesheet surface connectivity
        self.derived_var["facesheet_surfaces_connectivity"] = [
            [[1, "5_9"], [1, "9_10"], [-1, "6_10"], [-1, "5_6"]],
            [[1, "9_13"], [1, "13_14"], [-1, "10_14"], [-1, "9_10"]],
            [[1, "3_7"], [1, "7_8"], [-1, "4_8"], [-1, "3_4"]],
            [[1, "7_11"], [1, "11_12"], [-1, "8_12"], [-1, "7_8"]],
        ]

        # Core end surface box for extrusion
        self.derived_var["core_end_surface_box"] = [
            [[3, 3], [5, 5]],
            [[7, 7], [10, 10]],
            [[12, 12], [14, 14]],
        ]

    def zpr_core_geometry(self):
        """
        Defines the core geometry for a ZPR core by calculating node coordinates,
        line connectivity, surface connectivity, and core end surface box based on input parameters.
        Functionality:
            - Checks for chevron overlap and raises an error if adjacent chevrons intersect.
            - Computes node coordinates in `self.derived_var["coordinates"]` as numpy arrays.
            - Defines line connectivity between nodes in `self.derived_var["line_connectivity"]`.
            - Specifies surface connectivity for solid and facesheet surfaces in
              `self.derived_var["solid_surface_connectivity"]` and
              `self.derived_var["facesheet_surfaces_connectivity"]`.
            - Sets the core end surface box for extrusion in `self.derived_var["core_end_surface_box"]`.
        Raises:
            Exception: If the chevron pitch is too small and adjacent chevrons intersect.
        Modifies:
            self.derived_var: Updates with calculated coordinates, connectivity, and surface box information.
        Dependencies:
            - Assumes `self.var` and `self.derived_var` are dictionaries with required geometric parameters.
        """

        # Check for chevron overlap
        if (
            self.var["chevron_pitch"]
            < self.var["chevron_thickness"] / (np.cos(self.var["chevron_angle"]))
            and self.var["mirror_symmetry"][1]
        ):
            raise (
                "Error: Tips of adjecent chevrons intersect. \nAdvice: Increase chevron_pitch on decrease "
                "chevron_thickness"
            )

        # Define node coordinates using lists
        self.derived_var["coordinates"] = {
            "1": np.array([-self.derived_var["qlx"], 0.0, -self.derived_var["qlz"]]),
            "2": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"] / 2 * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "3": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"] / 2 * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "4": np.array(
                [
                    -self.derived_var["qlx"],
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "5": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2,
                    0.0,
                    -self.derived_var["qlz"],
                ]
            ),
            "6": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"] / 2 * np.tan(self.var["chevron_angle"])
                    + self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "7": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2,
                    -self.derived_var["qly"]
                    + 0.5 * self.var["chevron_pitch"]
                    + self.var["rib_thickness"] / 2 * np.tan(self.var["chevron_angle"])
                    - self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "8": np.array(
                [
                    -self.derived_var["qlx"] + self.var["rib_thickness"] / 2,
                    -self.derived_var["qly"],
                    -self.derived_var["qlz"],
                ]
            ),
            "9": np.array([0.0, 0.0, -self.derived_var["qlz"]]),
            "10": np.array(
                [
                    0.0,
                    -0.5 * self.var["chevron_pitch"]
                    + self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "11": np.array(
                [
                    0.0,
                    -0.5 * self.var["chevron_pitch"]
                    - self.var["chevron_thickness"]
                    / 2
                    / np.cos(self.var["chevron_angle"]),
                    -self.derived_var["qlz"],
                ]
            ),
            "12": np.array([0.0, -self.derived_var["qly"], -self.derived_var["qlz"]]),
        }

        # Define line connectivity
        self.derived_var["line_connectivity"] = [
            [[1, 5], [2, 6], [3, 7], [4, 8]],  # horizontal, left to right, column 1
            [[5, 9], [6, 10], [7, 11], [8, 12]],  # horizontal, left to right, column 2
            [[1, 2], [5, 6], [9, 10]],  # vertical, top to bottom, row 1
            [[2, 3], [6, 7], [10, 11]],  # vertical, top to bottom, row 2
            [[3, 4], [7, 8], [11, 12]],  # vertical, top to bottom, row 3
        ]

        # Surface connectivity: 1 is positive & -1 is negative orientation, "1_5" is line label
        self.derived_var["solid_surface_connectivity"] = [
            [[1, "1_5"], [1, "5_6"], [-1, "2_6"], [-1, "1_2"]],
            [[1, "2_6"], [1, "6_7"], [-1, "3_7"], [-1, "2_3"]],
            [[1, "3_7"], [1, "7_8"], [-1, "4_8"], [-1, "3_4"]],
            [[1, "6_10"], [1, "10_11"], [-1, "7_11"], [-1, "6_7"]],
        ]
        self.derived_var["facesheet_surfaces_connectivity"] = [
            [[1, "5_9"], [1, "9_10"], [-1, "6_10"], [-1, "5_6"]],
            [[1, "7_11"], [1, "11_12"], [-1, "8_12"], [-1, "7_8"]],
        ]

        # Define core end surface box for extruding where [part:[min:[x_node, y_node], max:[x_node, y_node]] ]
        self.derived_var["core_end_surface_box"] = [
            [[4, 4], [5, 5]],
            [[7, 7], [10, 10]],
        ]

    def eval_derived_variables(self):
        # Compute derived geometric variables common for all cores and store in self.derived
        self.derived_var = {}
        self.derived_var["qlx"] = self.var["chevron_wall_length"] * np.cos(
            self.var["chevron_angle"]
        )
        self.derived_var["qly"] = self.var["chevron_pitch"] + self.var[
            "chevron_wall_length"
        ] * np.sin(self.var["chevron_angle"])
        self.derived_var["qlz"] = (
            self.var["core_thickness"] / 2.0 + self.var["facesheet_thickness"]
        )
        self.derived_var["core_area_ratio"] = (
            self.derived_var["qly"] * self.var["rib_thickness"] / 2.0
            + (
                self.var["chevron_wall_length"]
                - (self.var["rib_thickness"] / 2.0) / np.cos(self.var["chevron_angle"])
            )
            * self.var["chevron_thickness"]
        ) / (self.derived_var["qlx"] * self.derived_var["qly"])
        self.derived_var["lx"] = (
            self.derived_var["qlx"]
            * (int(self.var["mirror_symmetry"][0]) + 1)
            * int(self.var["num_tiles"][0])
        )
        self.derived_var["ly"] = (
            self.derived_var["qly"]
            * (int(self.var["mirror_symmetry"][1]) + 1)
            * int(self.var["num_tiles"][1])
        )
        self.derived_var["lz"] = self.derived_var["qlz"] * 2

        # ZPR core specific variables
        match self.var["core_type"]:
            case "zpr":
                self.zpr_core_geometry()
            case "npr":
                self.npr_core_geometry()
            case "ppr":
                self.ppr_core_geometry()
            case _:
                raise ValueError(f"Unknown core type: {self.var['core_type']}")

        # save derived variables
        keys_to_save = ["qlx", "qly", "qlz", "core_area_ratio", "lx", "ly", "lz"]
        Utils.ReadWriteOps.save_object(
            {key: self.derived_var[key] for key in keys_to_save},
            os.path.join(
                self.directory.case_folder, "input", f"{self.case_number}_UC_derived"
            ),
            method="json",
        )

    def generate_mesh(self):
        """
        Generates the geometry and mesh for the RVE using GMSH.
        Steps:
            1. Builds RVE geometry from variables, derived variables and mesh parameters.
            2. Generates mesh, supporting linear/quadratic block elements.
            4. Applies mirroring and patterning for tessellation.
            5. Defines physical groups for volumes and surfaces.
            6. Removes duplicates, renumbers, and saves mesh as .inp.
        """

        def create_surface(connectivity: list, lines: dict) -> int:
            """
            Creates a surface by connecting curves based on the provided connectivity order.
            Args:
                connectivity (list of tuple): A list where each tuple contains orientation (int:+1/-1) and identifier
                (str) for the curve.
                lines (dict): A dictionary mapping curve identifiers to their GMSH tags.
            Returns:
                int: The tag or identifier of the created surface.
            """
            curve_ids: list[int] = [info[0] * lines[info[1]] for info in connectivity]
            loop: int = CAD.addCurveLoop(curve_ids)
            surf_tag: int = CAD.addPlaneSurface([loop])
            CAD.mesh.setTransfiniteSurface(surf_tag)
            return surf_tag

        def create_solid(
            surfaces: list, extrude_height: float, num_layers: int
        ) -> list[int]:
            """
            Builds a solid by extruding given surfaces to a specified height and number of layers.
            Args:
                surfaces (array-like): The GMSH tags of surfaces to be extruded.
                extrude_height (float): The height to extrude the surfaces.
                num_layers (int): The number of layers in the extrusion.
            Returns:
                list: The GMSH tags of the generated solid surfaces.
            """
            solid = np.asarray(
                CAD.extrude(
                    surfaces,
                    0.0,
                    0.0,
                    extrude_height,
                    numElements=[num_layers],
                    recombine=True,
                )
            )
            solid_tags: list[int] = solid[solid[:, 0] == 3, 1].tolist()
            np.vectorize(CAD.mesh.setTransfiniteSurface)(solid_tags)
            return solid_tags

        def create_cad():
            """
            Constructs the CAD geometry and mesh for representative volume element (RVE) based on the RVE's derived
            variables and mesh parameters.
            """
            # create nodes
            NODES: dict[str, int] = {}
            for label, value in self.derived_var["coordinates"].items():
                NODES[label] = CAD.addPoint(*value)

            # create lines
            LINES: dict[str, int] = {}
            for line_set in self.derived_var["line_connectivity"]:
                line_lengths: list = [
                    np.linalg.norm(
                        self.derived_var["coordinates"][str(a)]
                        - self.derived_var["coordinates"][str(b)]
                    )
                    for a, b in line_set
                ]
                num_nodes: int = (
                    int(np.ceil(max(line_lengths) / self.var["element_size"])) + 1
                )
                for a, b in line_set:
                    line_name = f"{a}_{b}"
                    LINES[line_name] = CAD.addLine(NODES[str(a)], NODES[str(b)])
                    CAD.synchronize()
                    CAD.mesh.setTransfiniteCurve(LINES[line_name], num_nodes)

            # create core surfaces
            CORE_SURFACES: list[int] = [
                create_surface(surface, lines=LINES)
                for surface in self.derived_var["solid_surface_connectivity"]
            ]

            # extrude facesheet surfaces and create solid facesheet if it exists
            FACESHEET_SURFACES: list[int] = []
            if self.var["facesheet_thickness"] > 0.0:
                # create facesheet surfaces
                FACESHEET_SURFACES = [
                    create_surface(surface, LINES)
                    for surface in self.derived_var["facesheet_surfaces_connectivity"]
                ]
                # extrude facesheet surfaces to create solid
                num_layers: int = int(
                    np.ceil(self.var["facesheet_thickness"] / self.var["element_size"])
                )
                FACESHEET_SOLIDS: list[int] = create_solid(
                    list(
                        (2, surface) for surface in FACESHEET_SURFACES + CORE_SURFACES
                    ),
                    self.var["facesheet_thickness"],
                    num_layers,
                )
            else:
                # if facesheet does not exist, just translate core surfaces
                CAD.translate(
                    [(2, s) for s in CORE_SURFACES],
                    dx=0.0,
                    dy=0.0,
                    dz=self.var["facesheet_thickness"],
                )

            # synchronize the CAD model
            CAD.synchronize()

            # identify core surfaces
            CORE_EXTRUDED_FACES: list = []
            for nodeLabel in self.derived_var["core_end_surface_box"]:
                min_x: float = self.derived_var["coordinates"][str(nodeLabel[0][0])][0]
                min_y: float = self.derived_var["coordinates"][str(nodeLabel[0][1])][1]
                max_x: float = self.derived_var["coordinates"][str(nodeLabel[1][0])][0]
                max_y: float = self.derived_var["coordinates"][str(nodeLabel[1][1])][1]
                CORE_EXTRUDED_FACES.extend(
                    MODEL.getEntitiesInBoundingBox(
                        min_x - eps,
                        min_y - eps,
                        -self.var["core_thickness"] / 2.0 - eps,
                        max_x + eps,
                        max_y + eps,
                        -self.var["core_thickness"] / 2.0 + eps,
                        2,
                    )
                )

            # extrude core surfaces to create solid
            num_layers: int = int(
                np.ceil((self.var["core_thickness"] / 2.0) / self.var["element_size"])
            )
            CORE_SOLIDS: list[int] = create_solid(
                CORE_EXTRUDED_FACES, self.var["core_thickness"] / 2.0, num_layers
            )

            # synchronize the CAD model
            CAD.synchronize()

        def tesselate(mirror_steps: list | None, pattern_steps: list | None) -> None:
            """
            Generates a tessellated mesh by mirroring and patterning a repeating unit of the model.

            Args:
                mirror_steps (list | None): List of mirroring steps, each as [tx, ty, tz], or None to skip mirroring.
                pattern_steps (list | None): List of pattern repetitions along each axis, e.g., [nx, ny], or None to
                skip patterning.
            """

            def get_entities(dim: int) -> dict:
                """
                Retrieves mesh data for all entities of the MODEL in a given dimension.

                Args:
                    dim (int): The dimension of the entities to retrieve (-1 for all).

                Returns:
                    dict: A dictionary mapping (dim, tag) to a tuple of:
                        - boundary entities,
                        - node data (node tags, coordinates),
                        - element data (element types, element tags, node tags).
                """
                m: dict = {}
                entities = MODEL.getEntities(dim)
                for e in entities:
                    bnd = MODEL.getBoundary([e])
                    nod = MESH.getNodes(e[0], e[1])
                    ele = MESH.getElements(e[0], e[1])
                    m[e] = (bnd, nod, ele)
                return m

            def reorder_nodes(e, m, tx: int, ty: int, tz: int) -> list[np.ndarray]:
                """
                Reorders the nodes of mesh elements according to specified axis transformations to create consistent
                element orientation when the mesh is mirrored.
                """
                num_elements: int = np.size(m[e][2][1])
                num_nodes: int = np.size(m[e][2][2])
                num_nodes_per_element: int = int(num_nodes / num_elements)

                if num_nodes_per_element == 8:
                    node_index = np.array(
                        [
                            [0, 1, 2, 3],  # -Z corner nodes
                            [4, 5, 6, 7],  # +Z corner nodes
                        ]
                    )
                elif num_nodes_per_element == 20:
                    node_index = np.array(
                        [
                            [0, 1, 2, 3],  # -Z corner nodes
                            [4, 5, 6, 7],  # +Z corner nodes
                            [8, 11, 13, 9],  # -Z edge nodes
                            [16, 18, 19, 17],  # +Z edge nodes
                            [10, 12, 14, 15],  # 0Z edge nodes
                        ]
                    )
                else:
                    raise (
                        f"Error: Unsupported number of nodes per element: {num_nodes_per_element}"
                    )

                if tx == -1:
                    for count, plane_index in enumerate(node_index):
                        if count in [0, 1, 4]:
                            node_index[count] = np.flip(plane_index)
                        elif count in [2, 3]:
                            node_index[count] = plane_index[[2, 1, 0, 3]]
                        else:
                            raise (
                                "Error: Unsupported number of node planes per element"
                            )
                if ty == -1:
                    for count, plane_index in enumerate(node_index):
                        if count in [0, 1, 4]:
                            node_index[count] = plane_index[[1, 0, 3, 2]]
                        elif count in [2, 3]:
                            node_index[count] = plane_index[[0, 3, 2, 1]]
                        else:
                            raise (
                                "Error: Unsupported number of node planes per element"
                            )
                if tz == -1:  #
                    if len(node_index) == 2:
                        node_index = node_index[[1, 0]]
                    elif len(node_index) == 5:
                        node_index = node_index[[1, 0, 3, 2, 4]]
                    else:
                        raise ("Error: Unsupported number of node planes per element")

                node_index = np.concatenate(node_index, axis=None)
                if num_nodes_per_element != 8:
                    abaqus_to_gmsh_node_order = [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        11,
                        16,
                        9,
                        17,
                        10,
                        18,
                        19,
                        12,
                        15,
                        13,
                        14,
                    ]
                    node_index = node_index[abaqus_to_gmsh_node_order]

                ordered_nodes: list[np.ndarray] = [
                    m[e][2][2][0][
                        list(
                            element_start_index + ordered_index
                            for ordered_index in node_index
                        )
                    ]
                    for element_start_index in range(
                        0, num_nodes, num_nodes_per_element
                    )
                ]
                ordered_nodes = [np.concatenate(ordered_nodes, axis=None)]

                return ordered_nodes

            def transform(
                m: dict,
                offset_entity: int,
                offset_node: int,
                offset_element: int,
                mirror_step: list | None,
                pattern_step: list | None,
            ) -> None:
                """
                Transforms and adds mesh entities, nodes, and elements to the MODEL and MESH objects with optional
                mirroring and pattern translation.

                Args:
                    m (dict): Dictionary of mesh entities and their data.
                    offset_entity (int): Offset to apply to entity tags.
                    offset_node (int): Offset to apply to node tags.
                    offset_element (int): Offset to apply to element tags.
                    mirror_step (list | None): List of [tx, ty, tz] for mirroring along axes, or None.
                    pattern_step (list | None): List of [ox, oy] for translation along axes, or None.
                """
                if mirror_step is not None:
                    tx, ty, tz = mirror_step[0], mirror_step[1], mirror_step[2]
                else:
                    tx, ty, tz = 1, 1, 1
                if pattern_step is not None:
                    ox, oy = pattern_step[0], pattern_step[1]
                else:
                    ox, oy = 0, 0

                for e in sorted(m):
                    MODEL.addDiscreteEntity(
                        e[0],
                        e[1] + offset_entity,
                        [
                            (abs(b[1]) + offset_entity) * np.copysign(1, b[1])
                            for b in m[e][0]
                        ],
                    )
                    coords = m[e][1][1]
                    transformed_coords = coords.copy()
                    transformed_coords[0::3] = transformed_coords[0::3] * tx + ox
                    transformed_coords[1::3] = transformed_coords[1::3] * ty + oy
                    transformed_coords[2::3] = transformed_coords[2::3] * tz

                    MESH.addNodes(
                        e[0],
                        e[1] + offset_entity,
                        m[e][1][0] + offset_node,
                        transformed_coords,
                    )

                    nodes = (
                        reorder_nodes(e, m, tx, ty, tz)
                        if all([mirror_step != [0, 0, 0], e[0] == 3])
                        else m[e][2][2]
                    )

                    MESH.addElements(
                        e[0],
                        e[1] + offset_entity,
                        m[e][2][0],
                        [t + offset_element for t in m[e][2][1]],
                        [n + offset_node for n in nodes],
                    )

                MESH.removeDuplicateNodes()

            def mirror(mirror_steps: list) -> None:
                """
                Mirrors mesh entities according to a sequence of mirror steps.

                Args:
                    mirror_steps (list): A list of lists, each specifying the mirror transformation for [x, y, z] axes.
                """
                m = get_entities(-1)
                for count, mirror_step in enumerate(mirror_steps, start=1):
                    transform(
                        m,
                        int(count * 10**4),
                        int(count * 10**7),
                        int(count * 10**6),
                        mirror_step,
                        None,
                    )
                MESH.renumberNodes()
                MESH.renumberElements()

            def pattern(pattern_steps: list) -> None:
                """
                Applies a pattern to unit cell model along specified axes to create a RVE.
                Args:
                    pattern_steps (list): A list of integers specifying the number of pattern repetitions along each
                    axis (e.g., [nx, ny]).
                """
                xmin, ymin, _, xmax, ymax, _ = MODEL.getBoundingBox(-1, -1)
                axis_lengths: list[float] = [xmax - xmin, ymax - ymin]

                for axis_index, axis_units in enumerate(pattern_steps):
                    axis_length = axis_lengths[axis_index]
                    m = get_entities(-1)

                    for count in range(1, axis_units):
                        quo, rem = divmod(count, 2)
                        offset_length = (quo + rem) * axis_length * (-1) ** (count - 1)
                        transform(
                            m,
                            int(count * 10 ** (5 + axis_index)),
                            int(count * 10 ** (7 + axis_index)),
                            int(count * 10 ** (6 + axis_index)),
                            None,
                            [
                                (offset_length if axis_index == 0 else 0),
                                offset_length if axis_index == 1 else 0,
                            ],
                        )

                    MESH.renumberNodes()
                    MESH.renumberElements()

            if mirror_steps is not None:
                mirror(mirror_steps)

            if pattern_steps is not None:
                pattern(pattern_steps)

        def create_physical_groups():
            """
            Creates named physical groups for further processing (e.g., boundary conditions).
            Physical groups are created for:
                - FACESHEET_TOP and FACESHEET_BOTTOM (volume, named 'FACESHEET')
                - CORE (volume, named 'CORE')
                - BOTTOM_FACE (surface, named 'ZNEG')
                - TOP_FACE (surface, named 'ZPOS')
                - RIGHT_FACE (surface, named 'XPOS')
                - REAR_FACE (surface, named 'YPOS')
                - LEFT_FACE (surface, named 'XNEG')
                - FRONT_FACE (surface, named 'YNEG')
            """

            # creating required phycal groups
            xmin, ymin, zmin, xmax, ymax, zmax = MODEL.getBoundingBox(-1, -1)
            FACESHEET_TOP = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmax - self.var["facesheet_thickness"] - eps),
                (xmax + eps),
                (ymax + eps),
                (zmax + eps),
                3,
            )
            FACESHEET_BOTTOM = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmin - eps),
                (xmax + eps),
                (ymax + eps),
                (zmin + self.var["facesheet_thickness"] + eps),
                3,
            )
            CORE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (-self.var["core_thickness"] / 2.0 - eps),
                (xmax + eps),
                (ymax + eps),
                (self.var["core_thickness"] / 2.0 + eps),
                3,
            )
            TOP_FACE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmax - eps),
                (xmax + eps),
                (ymax + eps),
                (zmax + eps),
                2,
            )
            BOTTOM_FACE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmin - eps),
                (xmax + eps),
                (ymax + eps),
                (zmin + eps),
                2,
            )
            LEFT_FACE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmin - eps),
                (xmin + eps),
                (ymax + eps),
                (zmax + eps),
                2,
            )
            RIGHT_FACE = MODEL.getEntitiesInBoundingBox(
                (xmax - eps),
                (ymin - eps),
                (zmin - eps),
                (xmax + eps),
                (ymax + eps),
                (zmax + eps),
                2,
            )
            FRONT_FACE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymin - eps),
                (zmin - eps),
                (xmax + eps),
                (ymin + eps),
                (zmax + eps),
                2,
            )
            REAR_FACE = MODEL.getEntitiesInBoundingBox(
                (xmin - eps),
                (ymax - eps),
                (zmin - eps),
                (xmax + eps),
                (ymax + eps),
                (zmax + eps),
                2,
            )

            MODEL.addPhysicalGroup(
                3,
                list(item[1] for item in FACESHEET_TOP + FACESHEET_BOTTOM),
                name="FACESHEET",
            )  # FACESHEET: MAT-2
            MODEL.addPhysicalGroup(
                3, list(item[1] for item in CORE), name="CORE"
            )  # CORE: MAT-1
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in BOTTOM_FACE), name="ZNEG"
            )  # FACE_1 # numbering based on element node ordering
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in TOP_FACE), name="ZPOS"
            )  # FACE_2
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in RIGHT_FACE), name="XPOS"
            )  # FACE_3
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in REAR_FACE), name="YPOS"
            )  # FACE_4
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in LEFT_FACE), name="XNEG"
            )  # FACE_5
            MODEL.addPhysicalGroup(
                2, list(item[1] for item in FRONT_FACE), name="YNEG"
            )  # FACE_6
            CAD.synchronize()

        def create_mesh() -> None:
            """
            Generate the mesh along with the mirror and pattern steps required for the RVE model, and saves it in
            ABAQUS format.
            """

            # mesh attributes
            meshparams = {
                "C3D20": {
                    "ELEMENT_SHAPE": "BLOCK",
                    "ELEMENT_ORDER": 2,
                    "REDUCED_INTEGRATION": False,
                },
                "C3D20R": {
                    "ELEMENT_SHAPE": "BLOCK",
                    "ELEMENT_ORDER": 2,
                    "REDUCED_INTEGRATION": True,
                },
                "C3D8": {
                    "ELEMENT_SHAPE": "BLOCK",
                    "ELEMENT_ORDER": 1,
                    "REDUCED_INTEGRATION": False,
                },
                "C3D8R": {
                    "ELEMENT_SHAPE": "BLOCK",
                    "ELEMENT_ORDER": 1,
                    "REDUCED_INTEGRATION": True,
                },
            }

            # Mesh generation
            if meshparams[self.var["element_type"]]["ELEMENT_SHAPE"] == "BLOCK":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                if meshparams[self.var["element_type"]]["ELEMENT_ORDER"] == 2:
                    gmsh.option.setNumber("Mesh.ElementOrder", 2)
                    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
                MESH.generate(3)

            # Mirroring repeating unit to create unit cell
            mirror_steps = []
            # Generate all mirror combinations for enabled axes, except the identity
            axes = self.var["mirror_symmetry"] + [True]  # Z axis always mirrored
            # For each axis, use [-1, 1] if enabled, else [1]
            options = [[-1, 1] if en else [1] for en in axes]
            for mirror in product(*options):
                if mirror != (1, 1, 1):
                    mirror_steps.append(list(mirror))
            # Create mirrored mesh
            tesselate(mirror_steps, None)
            CAD.synchronize()

            # Tessellating unit cell to create RVE
            if self.var["num_tiles"][0] > 1 or self.var["num_tiles"][1] > 1:
                tesselate(None, [self.var["num_tiles"][0], self.var["num_tiles"][1]])
            CAD.synchronize()

            # Define physical groups
            create_physical_groups()

            # Printing options
            gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -1000)
            gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -100)

            # Saving geometry mesh file
            gmsh.write(
                os.path.join(
                    self.directory.case_folder,
                    "mesh",
                    f"{self.case_number}_RVE_geometry.inp",
                )
            )

        # Initialize GMSH
        gmsh.initialize()

        # Set GMSH options
        eps = np.finfo(np.float32).eps  # machine epsilon
        gmsh.option.setNumber(
            "General.Verbosity", 2
        )  # Reduce verbosity, print error and warnings

        # Set GMSH model name
        MODEL = gmsh.model
        CAD = MODEL.geo
        MESH = MODEL.mesh

        # GMSH processes
        create_cad()
        create_mesh()

        # Show the model, if needed
        # gmsh.fltk.run()

        gmsh.finalize()

    def generate_inp(self):
        def read_mesh_data():
            """Reads node, element, and set data from a Gmsh-generated .inp file."""

            # Open the mesh file and read its contents
            with open(
                os.path.join(
                    self.directory.case_folder,
                    "mesh",
                    f"{self.case_number}_RVE_geometry.inp",
                ),
                "r",
            ) as file:
                lines = file.read().splitlines()

            # reach contents of the mesh file
            num_nodes_per_element = (
                int(self.var["element_type"].strip("C3D").strip("R")) + 1
            )
            temp_list = []
            section = None
            current_set = None
            for line in lines:
                if line.startswith("*NODE"):
                    section = "nodes"
                elif line.startswith("*ELEMENT"):
                    section = "elements"
                elif line.startswith("*ELSET,ELSET="):
                    section = "elsets"
                    current_set = line.strip().split("=")[1]
                    RVE_mesh["elsets"][current_set] = []
                elif line.startswith("*NSET,NSET="):
                    section = "nsets"
                    current_set = line.strip().split("=")[1]
                    RVE_mesh["nsets"][current_set] = []
                elif line.startswith("*"):
                    section = None
                elif section:
                    items = [item.strip() for item in line.split(",") if item.strip()]
                    if section == "nodes":
                        RVE_mesh["nodes"].append(list(map(float, items)))
                    elif section == "elements":
                        temp_list.extend(list(map(int, items)))
                        if (
                            len(temp_list) == num_nodes_per_element
                        ):  # number of nodes in each element
                            RVE_mesh["elements"].append(
                                temp_list
                            )  # element number and node numbers
                            temp_list = []
                    elif section == "elsets":
                        RVE_mesh["elsets"][current_set].extend(map(int, items))
                    elif section == "nsets":
                        RVE_mesh["nsets"][current_set].extend(map(int, items))

            # Convert lists to numpy arrays
            RVE_mesh["nodes"] = np.array(RVE_mesh["nodes"], dtype=object)
            RVE_mesh["nodes"][:, 0] = RVE_mesh["nodes"][:, 0].astype(np.int32)
            RVE_mesh["nodes"][:, 1:] = RVE_mesh["nodes"][:, 1:].astype(np.float64)

            RVE_mesh["elements"] = np.array(RVE_mesh["elements"], dtype=np.int32)

            for key in RVE_mesh["elsets"]:
                RVE_mesh["elsets"][key] = np.array(
                    RVE_mesh["elsets"][key], dtype=np.int32
                )
            for key in RVE_mesh["nsets"]:
                RVE_mesh["nsets"][key] = np.array(
                    RVE_mesh["nsets"][key], dtype=np.int32
                )

        def center_and_renumber_mesh():
            """Renumbers elements, and centers the model."""

            original_element_number = RVE_mesh["elements"][:, 0].copy()
            RVE_mesh["elements"][:, 0] = np.arange(
                1, RVE_mesh["elements"].shape[0] + 1, dtype=np.int32
            )
            for key in RVE_mesh["elsets"]:
                elset_indices = np.searchsorted(
                    original_element_number, RVE_mesh["elsets"][key]
                )
                RVE_mesh["elsets"][key] = elset_indices + 1

            min_coords = np.min(RVE_mesh["nodes"][:, 1:], axis=0)
            max_coords = np.max(RVE_mesh["nodes"][:, 1:], axis=0)
            center = min_coords + (max_coords - min_coords) / 2.0
            RVE_mesh["nodes"][:, 1:] -= center

        def add_pin_and_control_nodes():
            """Adds nodes for pinning and loading the RVE."""

            node_central_idx = np.argmin(
                np.sum(np.square(RVE_mesh["nodes"][:, 1:]), axis=1)
            )
            node_central = RVE_mesh["nodes"][node_central_idx, 0]

            all_set_nodes = np.unique(np.concatenate(list(RVE_mesh["nsets"].values())))
            if node_central in all_set_nodes:
                raise ("Error: central node for pinning is already in another set.")
            RVE_mesh["nsets"]["PIN-NODE"] = np.array([node_central], dtype=np.int32)

            node_number = RVE_mesh["nodes"][-1, 0] + 1
            for nset_name in ["STRAIN", "CURVATURE", "SHEAR"]:
                # Add the new node with correct types: [node_number, x, y, z]
                new_node = np.array([node_number, 0.0, 0.0, 0.0], dtype=object)
                new_node[0] = np.int32(node_number)
                new_node[1:] = np.float64(0.0)
                RVE_mesh["nodes"] = np.vstack((RVE_mesh["nodes"], new_node))
                RVE_mesh["nsets"][nset_name] = np.array([node_number], dtype=np.int32)
                node_number += 1

        def find_node_pairs() -> np.ndarray:
            """Finds corresponding node pairs on opposite faces for periodic BCs."""

            def pair_nodes(face1_nodes, face2_nodes, sort_coord1, sort_coord2):
                def sort_by_coords(nodes):
                    return nodes[
                        np.lexsort((nodes[:, sort_coord2], nodes[:, sort_coord1]))
                    ]

                sorted1 = sort_by_coords(face1_nodes)
                sorted2 = sort_by_coords(face2_nodes)
                return np.column_stack((sorted1[:, 0], sorted2[:, 0])).astype(np.int32)

            node_map = {node[0]: node for node in RVE_mesh["nodes"]}

            xpos_nodes = np.array([node_map[n] for n in RVE_mesh["nsets"]["XPOS"]])
            xneg_nodes = np.array([node_map[n] for n in RVE_mesh["nsets"]["XNEG"]])
            if len(xpos_nodes) != len(xneg_nodes):
                raise ("X+ and X- faces have different node counts.")
            xnode_pairs = pair_nodes(xpos_nodes, xneg_nodes, 2, 3)

            ypos_nodes = np.array([node_map[n] for n in RVE_mesh["nsets"]["YPOS"]])
            yneg_nodes = np.array([node_map[n] for n in RVE_mesh["nsets"]["YNEG"]])
            if len(ypos_nodes) != len(yneg_nodes):
                raise ("Y+ and Y- faces have different node counts.")
            ynode_pairs = pair_nodes(ypos_nodes, yneg_nodes, 1, 3)

            if np.unique(xnode_pairs).size != xnode_pairs.size:
                raise ("Duplicate node found in X-face pairs.")
            if np.unique(ynode_pairs).size != ynode_pairs.size:
                raise ("Duplicate node found in Y-face pairs.")

            # Filter out edge nodes from Y pairs to remove redundancy
            edge_nodes = np.intersect1d(
                RVE_mesh["nsets"]["XPOS"], RVE_mesh["nsets"]["YPOS"]
            )
            ynode_pairs_filtered = ynode_pairs[~np.isin(ynode_pairs[:, 0], edge_nodes)]

            return np.vstack((xnode_pairs, ynode_pairs_filtered))

        def generate_constraint_equations(
            node_pairs,
        ):  # TODO add transverse shear loading
            """Generates *EQUATION cards for periodic boundary conditions (vectorized for speed)."""

            eps = np.finfo(np.float32).eps
            node_map = {node[0]: node[1:] for node in RVE_mesh["nodes"]}

            equations = []
            for n1_id, n2_id in node_pairs:
                p1 = node_map[n1_id]
                p2 = node_map[n2_id]

                # Terms for displacement difference equations
                A, C = p1[0] - p2[0], p1[1] - p2[1]
                B, D = A * p1[2], C * p1[2]
                E = -(p1[0] ** 2 - p2[0] ** 2) / 2.0
                F = -(p1[1] ** 2 - p2[1] ** 2) / 2.0
                G = -(p1[0] * p1[1] - p2[0] * p2[1])

                # Equation for u_x difference
                eq_x = [f"{n1_id}, 1, -1.0", f"{n2_id}, 1, 1.0"]
                if abs(A) > eps:
                    eq_x.append(f"STRAIN, 1, {A}")
                if abs(B) > eps:
                    eq_x.append(f"CURVATURE, 1, {B}")
                if abs(C) > eps:
                    eq_x.append(f"STRAIN, 3, {0.5 * C}")
                if abs(D) > eps:
                    eq_x.append(f"CURVATURE, 3, {0.5 * D}")
                equations.append(eq_x)

                # Equation for u_y difference
                eq_y = [f"{n1_id}, 2, -1.0", f"{n2_id}, 2, 1.0"]
                if abs(A) > eps:
                    eq_y.append(f"STRAIN, 3, {0.5 * A}")
                if abs(B) > eps:
                    eq_y.append(f"CURVATURE, 3, {0.5 * B}")
                if abs(C) > eps:
                    eq_y.append(f"STRAIN, 2, {C}")
                if abs(D) > eps:
                    eq_y.append(f"CURVATURE, 2, {D}")
                equations.append(eq_y)

                # Equation for u_z difference
                eq_z = [f"{n1_id}, 3, -1.0", f"{n2_id}, 3, 1.0"]
                if abs(E) > eps:
                    eq_z.append(f"CURVATURE, 1, {E}")
                if abs(F) > eps:
                    eq_z.append(f"CURVATURE, 2, {F}")
                if abs(G) > eps:
                    eq_z.append(f"CURVATURE, 3, {0.5 * G}")
                equations.append(eq_z)

            return equations

        def write_input_file(equations=None):
            """Writes the complete Abaqus .inp file."""

            def format_lines(data_list, items_per_line=16):
                lines: list[str] = []
                for i in range(0, len(data_list), items_per_line):
                    chunk = data_list[i : i + items_per_line]
                    lines.append(
                        ", ".join(map(str, chunk))
                        + ("," if i + items_per_line < len(data_list) else "")
                    )
                return lines

            lines = ["*NODE"]
            lines.extend(
                [line for node in RVE_mesh["nodes"] for line in format_lines(node)]
            )

            lines.append(f"*ELEMENT, TYPE={self.var["element_type"]}")
            lines.extend(
                [
                    line
                    for element in RVE_mesh["elements"]
                    for line in format_lines(element)
                ]
            )

            for set_name, set_items in RVE_mesh["elsets"].items():
                lines.append(f"*ELSET, ELSET={set_name}")
                lines.extend(format_lines(set_items))
            for set_name, set_items in RVE_mesh["nsets"].items():
                lines.append(f"*NSET, NSET={set_name}")
                lines.extend(format_lines(set_items))

            lines.extend(
                [
                    "*ORIENTATION, NAME=GLOBAL, DEFINITION=COORDINATES",
                    "1., 0., 0., 0., 1., 0.",
                    "3, 0.",
                ]
            )

            # Add section controls for reduced integration elements
            bool_reduced_integration = (
                True if self.var["element_type"][-1] == "R" else False
            )
            if bool_reduced_integration:
                lines.append("*SECTION CONTROLS, NAME=EC-1, HOURGLASS=ENHANCED")

            # Write solid section definitions for each element set
            for set_name in RVE_mesh["elsets"]:
                lines.append(
                    f"*SOLID SECTION, ELSET={set_name}, MATERIAL=MAT-{set_name}, ORIENTATION=GLOBAL"
                    + (", CONTROLS=EC-1" if bool_reduced_integration else "")
                )

            # Define materials
            if "CORE" in RVE_mesh["elsets"].keys():
                lines.extend(
                    [
                        "*MATERIAL, NAME=MAT-CORE",
                        "*ELASTIC",
                        f"{self.var["core_material"][0]}, {self.var["core_material"][1]}",
                    ]
                )
            if "FACESHEET" in RVE_mesh["elsets"].keys():
                lines.extend(
                    [
                        "*MATERIAL, NAME=MAT-FACESHEET",
                        "*ELASTIC",
                        f"{self.var["facesheet_material"][0]}, {self.var["facesheet_material"][1]}",
                    ]
                )

            # Add equations for periodic boundary conditions
            if equations:
                lines.append("*EQUATION")
                for eq in equations:
                    lines.append(str(len(eq)))
                    lines.extend(format_lines(eq, items_per_line=4))

            # Add boundary conditions for homogenisation
            lines.append("*BOUNDARY")
            lines.append("PIN-NODE, PINNED")  # Pin all DOFs at the central node

            # Define homogenisation load cases and associated nodes/DOFs
            load_cases = [
                ("E11", "STRAIN", 1),
                ("E22", "STRAIN", 2),
                ("E12", "STRAIN", 3),
                ("K11", "CURVATURE", 1),
                ("K22", "CURVATURE", 2),
                ("K12", "CURVATURE", 3),
            ]  # TODO add transverse shear loading

            output_requested = False
            # Start perturbation step
            lines.append(
                "********************************************** PERTURBATION STEP : 0 START"
            )
            lines.append("*STEP, NAME=HOMOGENISATION-0, PERTURBATION")
            lines.append("*STATIC")
            if not output_requested:
                lines.extend(
                    [
                        "*OUTPUT, FIELD",
                        "NODE OUTPUT",
                        "CF, RF, U",
                        "*ELEMENT OUTPUT, DIRECTIONS=YES",
                        "E, LE, S, TRSHR",
                        "*OUTPUT, FIELD, VARIABLE=PRESELECT",
                        "*OUTPUT, HISTORY, FREQUENCY=0",
                    ]
                )
                output_requested = True

            # For each load case, apply unit displacement to the corresponding control node/DOF
            for case_name, nset, dof in load_cases:
                lines.append(f"*LOAD CASE, NAME=STATE-0-{case_name}")
                lines.append(
                    "*BOUNDARY, OP=NEW"
                )  # FIXME this is not correct, but works for now
                lines.append(f"{nset}, {dof}, {dof}, 1.0")
                # All other control nodes fixed (homogenisation: only one DOF active at a time)
                for other_nset, other_dof in [
                    (n, d) for (_, n, d) in load_cases if (n, d) != (nset, dof)
                ]:
                    lines.append(f"{other_nset}, {other_dof}, {other_dof}")
                # lines.append("PIN-NODE, PINNED")  # Ensure pin node remains fixed
                lines.append("*END LOAD CASE")

            lines.append("*END STEP")
            lines.append(
                "********************************************** PERTURBATION STEP : 0 END"
            )

            with open(
                os.path.join(
                    self.directory.case_folder, "inp", f"{self.case_number}_RVE.inp"
                ),
                "w",
            ) as f:
                f.write("\n".join(lines))

        # output container
        RVE_mesh = {
            "nodes": [],
            "elements": [],
            "element_type": self.var["element_type"],
            "elsets": {},
            "nsets": {},
        }

        # Read mesh data from the generated .inp file
        read_mesh_data()

        # renumber elements and center the mesh
        center_and_renumber_mesh()

        # Add pin and control nodes for loading the RVE
        add_pin_and_control_nodes()

        # Find node pairs for periodic boundary conditions
        consistent_node_pairs = find_node_pairs()

        # Generate *EQUATION cards for periodic boundary conditions
        eqns = generate_constraint_equations(consistent_node_pairs)

        # Write the complete Abaqus .inp file
        write_input_file(eqns)

    def run_abaqus(
        self, abaqus_path: str = "C:\\SIMULIA\\Commands\\abaqus.bat", num_core: int = 1
    ):
        """Runs the Abaqus analysis for the generated RVE model."""
        try:
            # Run the Abaqus analysis
            command = (
                f"{abaqus_path} analysis double=both job={self.case_number}_RVE input={self.job_file} cpus="
                f"{num_core} mp_mode=thread interactive"
            )
            Utils.run_subprocess(command, self.run_folder, self.log_file)
        except:
            traceback.print_exc()
            with open(self.log_file, "a") as f:
                f.write(
                    f"ERROR: Cannot run Abaqus for case number {self.case_number}\n"
                )

        # Copy result files
        for ext in ["odb", "msg", "dat", "sta"]:
            src = os.path.join(self.run_folder, f"{self.case_number}_RVE.{ext}")
            dst = os.path.join(
                self.directory.case_folder, ext, f"{self.case_number}_RVE.{ext}"
            )
            try:
                if os.path.exists(src):
                    shutil.copy(src, dst)
            except:
                traceback.print_exc()
                with open(
                    os.path.join(
                        self.directory.case_folder,
                        "log",
                        f"{self.case_number}_RVE_Abaqus.log",
                    ),
                    "a",
                ) as f:
                    f.write(
                        f"ERROR: Cannot copy result file {ext} for case number {self.case_number}\n"
                    )

    def extract_ABD_data(
        self, abaqus_path: str = "C:\\SIMULIA\\Commands\\abaqus.bat"
    ):
        # Extract results using a Python script
        script_file = os.path.abspath("abaqus_ABD_data.py")
        command = f"{abaqus_path} python {script_file} -- {self.directory.case_folder} {self.case_number}"
        Utils.run_subprocess(command, self.run_folder, self.log_file)

    @Utils.logger
    def analysis(self):
        print(f"\tUPDATE: starting {self.__class__.__name__} analysis {self.directory.case_name} - {self.case_number}")
        # Initialise geometry
        self.eval_derived_variables()
        # Generate mesh
        self.generate_mesh()
        # Generate input file
        self.generate_inp()
        # Job definition
        self.job_file = os.path.join(
            self.directory.case_folder, "inp", f"{self.case_number}_RVE.inp"
        )
        self.log_file = os.path.join(
            self.directory.case_folder, "log", f"{self.case_number}_RVE_Abaqus.log"
        )
        self.run_folder = os.path.join(
            self.directory.abaqus_folder, f"{self.case_number}"
        )
        os.makedirs(self.run_folder, exist_ok=True)
        self.run_abaqus()
        # Extract ABD matrix
        self.extract_ABD_data()


if __name__ == "__main__":
    # default analysis
    panel = RVE()
    panel.analysis()