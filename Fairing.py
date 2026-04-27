# Fairing.py
import sys
import os
import shutil
import traceback

import gmsh
import numpy as np
import scipy
import pandas as pd

import Utils
import Tailoring
from RVE import RVE
from DEFAUTLS import FAIRING_DEFAULTS

class FairingData:

    def __init__(self, case_folder, case_number, hinge_node={}, surface_nodes={}, shell_equivalent={}):
        self.case_folder = case_folder
        self.case_number = case_number
        self.hinge_node = hinge_node
        self.surface_nodes = surface_nodes
        self.shell_equivalent = shell_equivalent

class FairingGeometry:

    def __init__(
        self, variables=None, RVE_identifier=None, directory=Utils.Directory(), case_number: int = 0
    ):

        # set directory and case number
        self.directory = directory
        self.case_number = case_number

        # loading default variables
        self.var = FAIRING_DEFAULTS.copy()  
        if variables:
            self.var.update(variables)

        # add required RVE variables
        if self.var["model_fidelity"]=="equivalent":
            if RVE_identifier is None:
                panel = RVE(directory=directory)
                panel.analysis()
                RVE_identifier = panel.case_number
                uc_inputs = panel.var
                uc_derived_inputs = panel.derived_var
            else:
                uc_inputs =Utils.ReadWriteOps.load_object(
                    os.path.join(
                        self.directory.case_folder, "input", f"{RVE_identifier}_UC"
                    ),
                    "json",
                )
                uc_derived_inputs =Utils.ReadWriteOps.load_object(
                    os.path.join(
                        self.directory.case_folder, "input", f"{RVE_identifier}_UC_derived"
                    ),
                    "json",
                )
        elif self.var["model_fidelity"]=="explicit":
            reference_case = self.var["model_fidelity_settings"]["explicit"]["reference_case"]
            uc_inputs =Utils.ReadWriteOps.load_object(
                os.path.join(
                    self.directory.case_folder, "input", f"{reference_case}_UC"
                ),
                "json",
            )
            uc_derived_inputs =Utils.ReadWriteOps.load_object(
                os.path.join(
                    self.directory.case_folder, "input", f"{reference_case}_UC_derived"
                ),
                "json",
            )
        else:
            raise ValueError(f"ERROR: Unacceptable model_fidelity={self.var['model_fidelity']}")

        # extracting RVE variables
        self.RVE_variables = {
            key: uc_inputs[key]
            for key in [
                "chevron_wall_length",
                "chevron_thickness",
                "chevron_pitch",
                "rib_thickness",
                "core_thickness",
                "facesheet_thickness",
                "core_material",
                "facesheet_material",
            ]
        } | {key: uc_derived_inputs[key] for key in ["lx", "ly", "lz"]}

        # extracting stiffness
        if self.var["model_fidelity"]=="equivalent" and not self.var["model_fidelity_settings"]["equivalent"]["bool_isotropic"]:
            self.RVE_variables["stiffness"] =Utils.ReadWriteOps.load_object(
                os.path.join(
                    self.directory.case_folder, "data", f"{RVE_identifier}_stiffness"
                ),
                "json",
            )

        # saving independent variables
        Utils.ReadWriteOps.save_object(
            self.var,
            os.path.join(self.directory.case_folder, "input", f"{self.case_number}_FR"),
            method="json",
        )

    def load_aerofoil(self,aerofoil_database=None, num_points=None):
        if aerofoil_database is None:
            aerofoil_database = os.path.join(
                self.directory.run_folder, "aerofoil_database"
            )
        aerofoil_path = os.path.join(
            aerofoil_database, f"{self.var['aerofoil_name']}.dat"
        )
        if aerofoil_path is not None:
            try:
                with open(aerofoil_path, "r") as f:
                    lines = f.readlines()

                # Try to parse the first line
                first_line = lines[0].strip().split(",")
                try:
                    # Check if first line contains floats
                    [float(val) for val in first_line]
                    start_idx = 0  # First line contains data
                except ValueError:
                    start_idx = 1  # First line is text, ignore it

                # Parse the remaining lines into an array of floats
                coordinates = []
                for i in range(start_idx, len(lines)):
                    line = lines[i].strip().split(",")
                    if line:  # Skip empty lines
                        try:
                            coords = [float(val) for val in line]
                            coordinates.append(coords)
                        except ValueError:
                            continue
            except Exception:
                traceback.print_exc()
                return None
        else:
            raise ValueError(
                f"ERROR: Unacceptable value for aerofoil_path={aerofoil_path}"
            )

        # Resample
        coordinates = np.array(coordinates)

        if num_points is not None:
            # interpolate to required number of points
            le_idx = Utils.GeoOps.index_LE(coordinates) 
            # Generate x-coordinates (cosine spacing for better point distribution)
            beta = np.linspace(0, np.pi, num_points)
            x = (1.0 - np.cos(beta)) / 2.0
            # Interpolate y-coordinates for upper and lower surfaces
            yu = scipy.interpolate.interp1d(
                coordinates[: le_idx + 1, 0],
                coordinates[: le_idx + 1, 1]
                # fill_value="extrapolate"
            )(x)
            yl = scipy.interpolate.interp1d(
                coordinates[le_idx:, 0],
                coordinates[le_idx:, 1]
                # fill_value="extrapolate"
            )(x)
            # Combine upper and lower surfaces
            x_coords = np.concatenate([np.flip(x), x[1:]])
            y_coords = np.concatenate([np.flip(yu), yl[1:]])
            coordinates = np.column_stack((x_coords, y_coords))

        return coordinates

    def load_mesh_data(self, folder="inp"):
        """
        Load the mesh data for the fairing.
        """
        # load mesh data
        self.mesh_data = Utils.ReadWriteOps.read_mesh_data(
            os.path.join(
                self.directory.case_folder,
                folder,
                f"{self.case_number}_fairing_mesh.inp",
            )
        )

    @Utils.logger
    def generate_mesh(self, bool_show=False):
        """
        Generate the mesh for the fairing using GMSH.
        """

        def create_reference_nodes():
            """
            Add reference nodes for the fairing ribs used for boundary conditions and loading.
            """
            num_ribs = 2 + self.var["num_floating_ribs"]
            rib_spacing = (self.var["fairing_span"]/(1+self.var["pre_strain"])) / (2 * (num_ribs - 1))
            chordwise_position = self.var["hinge_chord_eta"] * self.var['fairing_chord']

            # rib nodes
            rib_ref_tags = []
            for i in range(1, num_ribs):
                rib_position = i * rib_spacing
                rib_ref_tags.append(CAD.addPoint(chordwise_position, rib_position, 0.0))

            # Synchronise
            CAD.synchronize()

            # create physical groups for floating ribs
            if num_ribs > 2:
                for i in range(1, num_ribs-1):
                    MODEL.addPhysicalGroup(0, [rib_ref_tags[i-1]], name=f"FLOAT_{i}")
            # pivoting rib
            MODEL.addPhysicalGroup(0, [rib_ref_tags[-1]], name="PIVOT")

            # Synchronise
            CAD.synchronize()

        def get_aerofoil_coords(num_points=None):
            # Check if it"s a NACA 4-digit specification (e.g., "0012")
            if (
                self.var["aerofoil_name"].startswith("naca")
                and len(self.var["aerofoil_name"].split("naca")[-1]) == 4
            ):
                # Use NACA 4-digit generator
                naca_code = self.var["aerofoil_name"].split("naca")[-1]
                reference_aerofoil_coords = Utils.GeoOps.generate_naca_4digit(
                    naca_code, num_points
                )
            else:
                # Load airfoil from file
                reference_aerofoil_coords = self.load_aerofoil(
                    num_points=num_points
                )

            # Scaling cross-section
            reference_aerofoil_coords *= self.var["fairing_chord"]

            self.aerofoil_coords = {"reference":reference_aerofoil_coords}

        def offset_aerofoil_coords(num_points=None, num_averaging=0, window=3, bool_plot=False):

            # get aerofoil coords if not already loaded
            if not hasattr(self, "aerofoil_coords"):
                get_aerofoil_coords(num_points=num_points)

            # Offset airfoil coordinates
            panel_thickness = self.RVE_variables["lz"]
            self.aerofoil_coords |= Utils.GeoOps.offset_airfoil(
                self.aerofoil_coords["reference"], panel_thickness, num_averaging, window
            )

            # Saving aerofoil coordinates
            Utils.ReadWriteOps.save_object(
                self.aerofoil_coords,
                os.path.join(
                    self.directory.case_folder,
                    "data",
                    f"{self.case_number}_aerofoil_coords",
                ),
                "pickle",
            )

            # Aerofoil plotting
            if bool_plot:
                Utils.Plots.aerofoils(
                    aerofoils=self.aerofoil_coords,
                    save_path=os.path.join(
                        self.directory.case_folder,
                        "fig",
                        f"{self.case_number}_aerofoils.png",
                    ),
                )

        def create_surface_elsets(aerofoil_coords):
            """
            Create element sets for the top and bottom surfaces of the fairing based on the chordwise position of the surface element centroids.
            """
            # checks
            assert np.all(
                self.mesh_data["nodes"][:, 0]
                == np.arange(self.mesh_data["nodes"].shape[0]) + 1
            ), (
                "Node numbers in mesh data are expected to start from 1 and be in ascending order without any missing numbers. Please check the mesh data."
            )
            assert len(self.mesh_data["elements"]) == 1, (
                "Expected only one element type in the mesh data. Please check the mesh data."
            )
            elements = np.vstack([
                    list(
                        elements
                        for elements in self.mesh_data["elements"].values()
                    )
                ]).squeeze()  
            assert np.all(elements[:, 0] == np.arange(elements.shape[0]) + 1),"Element numbers in mesh data are expected to start from 1 and be in ascending order without any missing numbers. Please check the mesh data."

            # surface elements
            surface_elements_set = self.mesh_data["elsets"]["OUTER_SURFACE"]
            surface_elements_nodes = elements[
                Utils.indices(elements[:, 0], surface_elements_set),
                1:,
            ]

            # surface element centroids
            surface_elements_centroids = self.mesh_data["nodes"][
                surface_elements_nodes - 1
            ][:, :, 1:].mean(axis=1)

            # chordline classifier
            chordline_interp_f = Utils.GeoOps.chordline_interpolator(
                aerofoil_coords,
            )
            classifier = chordline_interp_f(
                surface_elements_centroids[:, [0, 2]]
            )

            # create element sets for top and bottom surfaces
            top_surface_elements_set = surface_elements_set[classifier > 0]
            bottom_surface_elements_set = surface_elements_set[classifier < 0]
            self.mesh_data["elsets"]["OUTER_SURFACE_TOP"] = (
                top_surface_elements_set
            )
            self.mesh_data["elsets"]["OUTER_SURFACE_BOTTOM"] = (
                bottom_surface_elements_set
            )
            assert (
                self.mesh_data["elsets"]["OUTER_SURFACE"].shape[0]
                == self.mesh_data["elsets"]["OUTER_SURFACE_TOP"].shape[0]
                + self.mesh_data["elsets"]["OUTER_SURFACE_BOTTOM"].shape[0]
            )

            ###
            # This is required for calculation of distortion and for tailoring process

            # surface nodes, masks and coords
            surface_nodes_set = self.mesh_data["nsets"]["OUTER_SURFACE"]
            surface_nodes_coords = self.mesh_data["nodes"][
                Utils.indices(self.mesh_data["nodes"][:, 0], surface_nodes_set), 1:
            ]

            # collect mesh data
            mesh_data = {
                "surface_nodes": surface_nodes_set,
                "surface_nodes_coords": surface_nodes_coords,
                "surface_elements": surface_elements_set,
                "surface_elements_nodes": surface_elements_nodes,
                "surface_element_centroids": surface_elements_centroids,
                "top_surface_elements": top_surface_elements_set,
                "bottom_surface_elements": bottom_surface_elements_set,
            }

            # params for tailoring
            if "S4R" in self.mesh_data["elements"].keys():
                try:
                    TE = self.mesh_data["nsets"]["TE_TOP"]
                except:
                    TE = self.mesh_data["nsets"]["TE"]

                mesh_data["corner_node"] = self.mesh_data["nsets"]["RIB_0"][
                    np.argwhere(
                        np.isin(self.mesh_data["nsets"]["RIB_0"], TE)
                    ).squeeze()
                ]
                mesh_data["corner_element"] = surface_elements_set[
                    np.argwhere(
                        surface_elements_nodes[:, 0] == mesh_data["corner_node"]
                    ).squeeze()
                ]
                mesh_data["element_grid_shape"] = np.array(
                    [TE.shape[0] - 1, self.mesh_data["nsets"]["RIB_0"].shape[0] - 1]
                )

            # Saving mesh data to be used in the tailoring process
            Utils.ReadWriteOps.save_object(
                mesh_data,
                os.path.join(
                    self.directory.case_folder,
                    "data",
                    f"{self.case_number}_fairing_mesh_data",
                ),
                "pickle",
            )

        def create_equivalent_model():
            """
            Generate a shell model for the fairing either using isotropic material properties of the core or equivalent shell stiffness of the panel.
            """

            def create_shell_geometry():
                """
                Create the shell surface for the fairing.
                """
                # Chordwise Discretisation
                wing_cross_section_coords = self.aerofoil_coords["mid"]
                wing_cross_section_arc_length = np.sum(
                    np.linalg.norm(np.diff(wing_cross_section_coords, axis=0), axis=1)
                )
                num_arc_units, rem = divmod(
                    wing_cross_section_arc_length, self.RVE_variables["ly"]
                )
                
                if rem > 0.0:
                    print(
                        f"\t\tWARNING: rib arc length not an integer multiple of cell size. Num cells {num_arc_units}, Remainder {rem:.4f}"
                    )
                if self.var["element_size"] is None:
                    num_nodes_in_wing_cross_section = int(num_arc_units)
                else:
                    num_nodes_in_wing_cross_section = int(
                        wing_cross_section_arc_length / self.var["element_size"]
                    )
                if num_nodes_in_wing_cross_section % 2 == 0:
                    num_nodes_in_wing_cross_section += 1  # ensure odd number for symmetry

                # Spanwise discretisation
                num_ribs = 2 + self.var["num_floating_ribs"]
                rib_spacing = (self.var["fairing_span"]/(1+self.var["pre_strain"])) / (2 * (num_ribs - 1))
                num_bay_units, rem = divmod(rib_spacing, self.RVE_variables["lx"])
                if rem > 0.0:
                    print(
                        f"\t\tWARNING: bay spacing not an integer multiple of cell size. Num cells {num_bay_units}, Remainder {rem:.4f}"
                    )
                if self.var["element_size"] is None:
                    num_nodes_in_wing_rib_bay = int(num_bay_units)
                else:
                    num_nodes_in_wing_rib_bay = int(
                        rib_spacing / self.var["element_size"]
                    )
                if num_nodes_in_wing_rib_bay % 2 == 0:
                    num_nodes_in_wing_rib_bay += 1  # ensure odd number for symmetry

                # Check if the airfoil is closed (last point equals first point)
                is_closed_profile = np.allclose(
                    wing_cross_section_coords[-1], wing_cross_section_coords[0]
                )

                # Create the first rib as a reference
                points_to_use = (
                    wing_cross_section_coords[:-1]
                    if is_closed_profile
                    else wing_cross_section_coords
                )
                first_rib_nodes = []
                for i, node in enumerate(points_to_use):
                    first_rib_nodes.append(CAD.addPoint(node[0], 0, node[1]))

                # If closed profile, reference the first node to close the loop
                if is_closed_profile:
                    first_rib_nodes.append(first_rib_nodes[0])

                # Create line for the first rib
                first_rib_line = CAD.addPolyline(first_rib_nodes)
                # Calculate and print the length of the first rib line
                CAD.synchronize()  # Ensure the geometry is synchronized
                CAD.mesh.setTransfiniteCurve(
                    first_rib_line, num_nodes_in_wing_cross_section
                )

                # Create subsequent ribs using translation of the entire line
                rib_line_tags = [first_rib_line]
                TE_line_tags = []
                surface_tags = []
                for i in range(1, num_ribs):
                    previous_rib_position = (i - 1) * rib_spacing
                    new_rib_position = i * rib_spacing

                    # Copy and translate the whole line
                    new_rib_line = CAD.copy([(1, first_rib_line)])[0][1]
                    CAD.translate(
                        [(1, new_rib_line)], 0, new_rib_position, 0
                    )  # Translate to new position

                    # Set transfinite
                    CAD.mesh.setTransfiniteCurve(
                        new_rib_line, num_nodes_in_wing_cross_section
                    )

                    # Store the new rib line
                    rib_line_tags.append(new_rib_line)

                    # Synchronise
                    CAD.synchronize()

                    # Trailing edges
                    if is_closed_profile:
                        temp_nodes = MODEL.getEntitiesInBoundingBox(
                            (points_to_use[0][0] - eps),
                            (previous_rib_position - eps),
                            (points_to_use[0][1] - eps),
                            (points_to_use[0][0] + eps),
                            (new_rib_position + eps),
                            (points_to_use[0][1] + eps),
                            0,
                        )
                        TE_line_tags.append(
                            CAD.addPolyline([node[1] for node in temp_nodes])
                        )
                    else:
                        temp_nodes_top = MODEL.getEntitiesInBoundingBox(
                            (points_to_use[0][0] - eps),
                            (previous_rib_position - eps),
                            (points_to_use[0][1] - eps),
                            (points_to_use[0][0] + eps),
                            (new_rib_position + eps),
                            (points_to_use[0][1] + eps),
                            0,
                        )
                        temp_nodes_bottom = MODEL.getEntitiesInBoundingBox(
                            (points_to_use[-1][0] - eps),
                            (previous_rib_position - eps),
                            (points_to_use[-1][1] - eps),
                            (points_to_use[-1][0] + eps),
                            (new_rib_position + eps),
                            (points_to_use[-1][1] + eps),
                            0,
                        )
                        TE_line_tags.append(
                            [
                                CAD.addPolyline([node[1] for node in temp_nodes_top]),
                                CAD.addPolyline(
                                    [node[1] for node in temp_nodes_bottom]
                                ),
                            ]
                        )
                    for line in TE_line_tags[-1]:
                        CAD.mesh.setTransfiniteCurve(line, num_nodes_in_wing_rib_bay)

                    # Curve Loop
                    temp_TE = (
                        [TE_line_tags[-1], TE_line_tags[-1]]
                        if is_closed_profile
                        else TE_line_tags[-1]
                    )
                    temp_loop = CAD.addCurveLoop(
                        [temp_TE[0], rib_line_tags[-1], -temp_TE[1], -rib_line_tags[-2]]
                    )
                    surface_tags.append(CAD.addSurfaceFilling([temp_loop]))
                    CAD.mesh.setTransfiniteSurface(surface_tags[-1])

                    # Synchronize
                    CAD.synchronize()

                # Define physical groups
                # Remove existing physical groups
                CAD.removePhysicalGroups([])
                # Ribs
                for i, rib in enumerate(rib_line_tags[:-1]):
                    CAD.addPhysicalGroup(1, [rib], name=f"RIB_{i}")
                # Pivoting rib
                CAD.addPhysicalGroup(1, [rib_line_tags[-1]], name="RIB_P")
                # All ribs
                CAD.addPhysicalGroup(1, rib_line_tags, name="RIBS")
                # Trailing Edge
                if is_closed_profile:
                    CAD.addPhysicalGroup(1, [TE_line_tags], name="TE")
                else:
                    CAD.addPhysicalGroup(
                        1, [tag[0] for tag in TE_line_tags], name="TE_TOP"
                    )
                    CAD.addPhysicalGroup(
                        1, [tag[1] for tag in TE_line_tags], name="TE_BOTTOM"
                    )
                # Surface
                CAD.addPhysicalGroup(2, surface_tags, name="SHELL_EQUIVALENT")
                CAD.addPhysicalGroup(2, surface_tags, name="OUTER_SURFACE")

                # Synchronize
                CAD.synchronize()

            def create_mesh() -> None:
                """
                Generate the mesh  and saves it in ABAQUS format.
                """
                # Mesh generation
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                MESH.generate(2)

                # Printing options
                gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -100)
                gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -111)

                # Saving geometry mesh file
                gmsh.write(
                    os.path.join(
                        self.directory.case_folder,
                        "mesh",
                        f"{self.case_number}_fairing_mesh.inp",
                    )
                )

            def create_abaqus_geometry(bool_plot=False) -> None:
                """
                Generate the ABAQUS geometry with section, material, orientation rigid-body properties.
                """
                # load GMSH mesh data
                self.load_mesh_data("mesh") 

                # add top and bottom surface element sets
                create_surface_elsets(self.aerofoil_coords["mid"])

                # initialise a list
                lines = []

                # print nodes
                lines.append("*NODE")
                lines.extend(
                    [
                        line
                        for node in self.mesh_data["nodes"]
                        for line in Utils.ReadWriteOps.format_lines(node)
                    ]
                )

                # print elements
                for type, element in self.mesh_data["elements"].items():
                    lines.append(f"*ELEMENT, TYPE={type}")
                    lines.extend(
                        [
                            line
                            for element in self.mesh_data["elements"][type]
                            for line in Utils.ReadWriteOps.format_lines(element)
                        ]
                    )

                # print sets
                for set_name, set_items in self.mesh_data["elsets"].items():
                    lines.append(f"*ELSET, ELSET={set_name}")
                    lines.extend(Utils.ReadWriteOps.format_lines(set_items))
                for set_name, set_items in self.mesh_data["nsets"].items():
                    lines.append(f"*NSET, NSET={set_name}")
                    lines.extend(Utils.ReadWriteOps.format_lines(set_items))

                # print orientation
                lines.extend(
                    [
                        "*ORIENTATION, NAME=MAT-AXIS, DEFINITION=COORDINATES, SYSTEM=CYLINDRICAL",
                        f"{self.var['fairing_chord'] / 2}, {self.var['fairing_span'] / 4}, 0.0, {self.var['fairing_chord'] / 2}, {self.var['fairing_span'] / 3}, 0.0",
                        "1, 90.000000",
                    ]
                )

                # define section
                if self.var["model_fidelity_settings"]["equivalent"]["bool_isotropic"]:
                    lines.extend(
                        [
                            "*SHELL SECTION, ELSET=SHELL_EQUIVALENT, MATERIAL=MAT-CORE, ORIENTATION=MAT-AXIS",
                            f"{self.RVE_variables['lz']}, 5",
                        ]
                    )
                    lines.extend(
                        [
                            "*MATERIAL, NAME=MAT-CORE",
                            "*ELASTIC",
                            f"{self.RVE_variables['core_material'][0]}, {self.RVE_variables['core_material'][1]}",
                            "*DENSITY",
                            "1",
                        ]
                    )
                else:
                    ABDK = self.RVE_variables["stiffness"]
                    # lower left half of the symmetric matrix
                    # indexed from top->bottom and left->right
                    shell_general_section_data = [
                        ABDK["A11"],
                        ABDK["A12"],
                        ABDK["A22"],
                        ABDK["A16"],
                        ABDK["A26"],
                        ABDK["A66"],
                        0.0,
                        0.0,
                        0.0,
                        ABDK["D11"],
                        0.0,
                        0.0,
                        0.0,
                        ABDK["D12"],
                        ABDK["D22"],
                        0.0,
                        0.0,
                        0.0,
                        ABDK["D16"],
                        ABDK["D26"],
                        ABDK["D66"],
                    ]
                    # Format all values to 6 significant figures
                    shell_general_section_data = [
                        f"{v:.6f}" for v in shell_general_section_data
                    ]
                    # shell section definition
                    lines.append(
                        "*SHELL GENERAL SECTION, ELSET=SHELL_EQUIVALENT, ORIENTATION=MAT-AXIS, DENSITY=1"
                    )
                    lines.extend(Utils.ReadWriteOps.format_lines(shell_general_section_data, 8))
                    # shear properties: this must be immediately after shell section definition
                    if "K11" in ABDK:
                        shell_general_shear_data = [
                            ABDK["K11"][0],
                            ABDK["K22"][0],
                            ABDK["K12"][0],
                        ]
                        lines.append("*TRANSVERSE SHEAR STIFFNESS")
                        lines.extend(Utils.ReadWriteOps.format_lines(shell_general_shear_data, 8))

                # create trailing edge rigid body elements
                if all(set in self.mesh_data["nsets"] for set in ["TE_TOP", "TE_BOTTOM"]):
                    # node indices
                    top_nodes_idx = np.setdiff1d(
                        self.mesh_data["nsets"]["TE_TOP"], self.mesh_data["nsets"]["RIBS"], assume_unique=True
                    ) - 1 # node number starts from 1 but the index starts from 0
                    bottom_nodes_idx = np.setdiff1d(
                        self.mesh_data["nsets"]["TE_BOTTOM"], self.mesh_data["nsets"]["RIBS"], assume_unique=True
                    ) - 1 # node number starts from 1 but the index starts from 0

                    # arranged node indices along Y-axis
                    top_nodes_idx = top_nodes_idx[np.argsort(self.mesh_data["nodes"][top_nodes_idx][:,2], axis=0)]
                    bottom_nodes_idx = bottom_nodes_idx[np.argsort(self.mesh_data["nodes"][bottom_nodes_idx][:,2], axis=0)]

                    # Create rigid body constraints
                    if top_nodes_idx.size == bottom_nodes_idx.size:
                        lines.append("*MPC")
                        for top_idx, bottom_idx in zip(top_nodes_idx, bottom_nodes_idx):
                            lines.append(f"BEAM, {top_idx+1}, {bottom_idx+1}")

                if bool_plot:
                    # plots for debugging
                    top_coords = self.mesh_data["nodes"][np.ix_(top_nodes_idx,[2,3])]
                    bottom_coords = self.mesh_data["nodes"][np.ix_(bottom_nodes_idx,[2,3])]
                    Utils.Plots.node_coupling(top_coords, bottom_coords, xlabel="Y", ylabel="Z", save_path=os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_trailing_edge_coupling.png"))

                    slave_coords = self.mesh_data["nodes"][np.ix_(self.mesh_data["nsets"]["RIB_P"]-1,[1,3])]
                    master_coords = self.mesh_data["nodes"][np.ix_(self.mesh_data["nsets"]["PIVOT"]-1,[1,3])] * np.ones_like(slave_coords)
                    Utils.Plots.node_coupling(master_coords, slave_coords, xlabel="X", ylabel="Z", save_path=os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_pivoting_rib_coupling.png"))

                # saving file
                with open(
                    os.path.join(
                        self.directory.case_folder,
                        "inp",
                        f"{self.case_number}_fairing_mesh.inp",
                    ),
                    "w",
                ) as f:
                    f.write("\n".join(lines))

            # Get cross-section shape
            get_aerofoil_coords(num_points=None)
            offset_aerofoil_coords(num_averaging=1, window=3, bool_plot=True)

            # Create shell geometry
            create_shell_geometry()

            # Add reference nodes
            create_reference_nodes()

            # Create mesh
            create_mesh()

            # Create geometry
            create_abaqus_geometry()

        def create_explicit_model():

            def create_tailored_geometry():
                """
                Generate tailored geometry for explicit fairing model.
                """
                # Get reference case and field
                reference_case = self.var["model_fidelity_settings"]["explicit"]["reference_case"]
                reference_field = self.var["model_fidelity_settings"]["explicit"]["reference_field"]

                # Generate tailored geometry
                tailored = Tailoring.Lattice(self.directory, self.case_number, reference_case, reference_field)
                tailored.analysis(0.5e-2 * self.var["fairing_chord"])

            def create_shell_geometry():

                geometry = Utils.ReadWriteOps.read_mesh_data(os.path.join(self.directory.case_folder, "mesh", f"{self.case_number}_fairing_geometry.inp"))

                # create points
                points = np.empty((geometry["nodes"].shape[0],), dtype=int)
                for i, node in enumerate(geometry["nodes"]):
                    points[i] = CAD.addPoint(node[1], node[2], node[3])

                # Check if all points are created
                assert np.all(points == geometry["nodes"][:, 0]), "ERROR: Some points were not created. They likely got merged. Increase the tolerance in Tailoring.py"

                #  create elements
                if geometry["elements"].keys() != {"S3R"}:
                    raise ValueError(f"ERROR: Unsupported element types. Recieved {geometry['elements'].keys()}, expected {{'S3R'}}")

                # get unique edges
                signed_indices, unique_edges = Utils.GeoOps.edge_pairs(geometry["elements"]["S3R"][:, 1:])

                # create edges
                surface_edges = np.empty((unique_edges.shape[0],), dtype=int)
                for i, edge in enumerate(unique_edges):
                    surface_edges[i] = CAD.addLine(edge[0], edge[1])

                # create surfaces using signed reference to unique edges
                surface = np.empty((geometry["elements"]["S3R"].shape[0],), dtype=int)
                for i, element_edges in enumerate(signed_indices):
                    signs = element_edges[:,0]
                    indices = element_edges[:,1]
                    surface[i] = CAD.addPlaneSurface(
                        [
                            CAD.addCurveLoop((signs*surface_edges[indices]))
                        ]
                    )

                # Check if all surfaces are created
                assert np.all(surface == geometry["elements"]["S3R"][:, 0]), "ERROR: Some surfaces were not created."

                # Synchronize
                CAD.synchronize()

                # Remove existing physical groups
                MODEL.removePhysicalGroups([])

                # Create physical groups for element sets
                for name, elset in geometry["elsets"].items():
                    MODEL.addPhysicalGroup(2, elset, name=name)

                # Synchronize
                CAD.synchronize()

                # Create seperate physical group for each rib
                assert self.var["num_floating_ribs"] == 0, "ERROR: Explicit model not implemented for floating ribs yet"
                num_ribs = 2 + self.var["num_floating_ribs"]
                rib_spacing = (self.var["fairing_span"]/(1+self.var["pre_strain"])) / (2 * (num_ribs - 1)) # spacing between ribs
                ribs_elements = geometry["elsets"]["RIBS"] # element tags for all ribs
                model_bounds = np.array( MODEL.get_bounding_box(-1,-1) ) # modeling bounds
                tolerance = 1e-2
                for i  in range(num_ribs):
                    # Get elements at the rib location
                    model_bounds[[1,4]] = rib_spacing * i
                    model_bounds[:3] -= tolerance
                    model_bounds[3:] += tolerance
                    local_rib_elements = np.array(MODEL.getEntitiesInBoundingBox(*model_bounds, dim=2), dtype=int)[:,1]
                    # Filter only rib elements
                    local_rib_elements = np.intersect1d(local_rib_elements, ribs_elements)
                    # Create physical group
                    MODEL.addPhysicalGroup(2, local_rib_elements, name=["RIB_0", "RIB_P"][i])

                # Create physical groups for Materials
                # CORE
                MODEL.addPhysicalGroup(
                    2,
                    np.r_[*(
                        geometry["elsets"][elset]
                        for elset in ['STRINGERS', 'CHEVRONS']
                    )],
                    name="CORE",
                )
                # facesheet
                facesheet_elsets = np.intersect1d(np.array(["OUTER_SURFACE", "INNER_SURFACE"]), np.array(list(geometry["elsets"].keys())))
                if facesheet_elsets.size != 0:
                    MODEL.addPhysicalGroup(
                        2,
                        np.r_[*(
                            geometry["elsets"][elset]
                            for elset in ["OUTER_SURFACE", "INNER_SURFACE"]
                        )],
                        name="FACESHEET",
                    )

                # Synchronize
                CAD.synchronize()

            def create_mesh() -> None:
                """
                Generate the mesh  and saves it in ABAQUS format.
                """
                # Mesh generation
                gmsh.option.setNumber("Mesh.MeshSizeMax", self.var["element_size"])
                gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
                MESH.generate(2)
                MESH.optimize("Laplace2D") 


                # Printing options
                gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -100)
                gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -111)

                # Saving geometry mesh file
                gmsh.write(
                    os.path.join(
                        self.directory.case_folder,
                        "mesh",
                        f"{self.case_number}_fairing_mesh.inp",
                    )
                )

            def create_abaqus_geometry(bool_plot=False) -> None:
                """
                Generate the ABAQUS geometry with section, material, orientation rigid-body properties.
                """

                # load GMSH mesh data
                self.load_mesh_data("mesh")

                # Pre-processing
                # Remove rib elements
                for type in self.mesh_data["elements"].keys():
                    self.mesh_data["elements"][type] = self.mesh_data["elements"][type][
                        ~np.isin(
                            self.mesh_data["elements"][type][:, 0],
                            self.mesh_data["elsets"]["RIBS"],
                            assume_unique=True,
                        )
                    ]
                # Remove ribs elsets
                for elset in list(self.mesh_data["elsets"].keys()):
                    if elset.startswith("RIB"):
                        del self.mesh_data["elsets"][elset]
                # Enforce rib nodes location
                self.mesh_data["nodes"][np.isin(self.mesh_data["nodes"][:, 0], self.mesh_data["nsets"]["RIB_0"], assume_unique=True)][:,2] = 0.0
                self.mesh_data["nodes"][np.isin(self.mesh_data["nodes"][:, 0], self.mesh_data["nsets"]["RIB_P"], assume_unique=True)][:,2] = (self.var["fairing_span"]/(1+self.var["pre_strain"]))

                # add top and bottom surface element sets
                get_aerofoil_coords(num_points=None)
                create_surface_elsets(self.aerofoil_coords["reference"])

                # initialise a list
                lines = []

                # print nodes
                lines.append("*NODE")
                lines.extend(
                    [
                        line
                        for node in self.mesh_data["nodes"]
                        for line in Utils.ReadWriteOps.format_lines(node)
                    ]
                )

                # print elements
                for type, element in self.mesh_data["elements"].items():
                    lines.append(f"*ELEMENT, TYPE={type}")
                    lines.extend(
                        [
                            line
                            for element in self.mesh_data["elements"][type]
                            for line in Utils.ReadWriteOps.format_lines(element)
                        ]
                    )

                # print sets
                for set_name, set_items in self.mesh_data["elsets"].items():
                    lines.append(f"*ELSET, ELSET={set_name}")
                    lines.extend(Utils.ReadWriteOps.format_lines(set_items))
                for set_name, set_items in self.mesh_data["nsets"].items():
                    lines.append(f"*NSET, NSET={set_name}")
                    lines.extend(Utils.ReadWriteOps.format_lines(set_items))

                # print orientation
                lines.extend(
                    [
                        "*ORIENTATION, NAME=MAT-AXIS, DEFINITION=COORDINATES, SYSTEM=CYLINDRICAL",
                        f"{self.var['fairing_chord'] / 2}, {self.var['fairing_span'] / 4}, 0.0, {self.var['fairing_chord'] / 2}, {self.var['fairing_span'] / 3}, 0.0",
                        "1, 90.000000",
                    ]
                )

                # define section and material for each part
                materials = np.intersect1d(np.array(["CORE", "FACESHEET"]), np.array(list(self.mesh_data["elsets"].keys())))

                # define facesheet sections
                if "FACESHEET" in materials:
                    facesheet_sections = np.intersect1d(np.array(["INNER_SURFACE", "OUTER_SURFACE"]), np.array(list(self.mesh_data["elsets"].keys())))
                    offsets = {
                        "INNER_SURFACE": 0.5,
                        "OUTER_SURFACE": -0.5,
                    }
                    if facesheet_sections.size != 0:
                        for set_name in facesheet_sections: 
                            lines.extend(
                                [
                                    f"*SHELL SECTION, ELSET={set_name}, MATERIAL=MAT-FACESHEET, ORIENTATION=MAT-AXIS, OFFSET={offsets[set_name]}",
                                    f"{self.RVE_variables['facesheet_thickness']}, 5",
                                ]
                            )

                # define core sections
                thickness = {
                    "STRINGERS": self.RVE_variables['rib_thickness'],
                    "CHEVRONS": self.RVE_variables['chevron_thickness'],
                }
                for set_name in ["STRINGERS", "CHEVRONS"]: 
                    lines.extend(
                        [
                            f"*SHELL SECTION, ELSET={set_name}, MATERIAL=MAT-CORE",
                            f"{thickness[set_name]}, 5",
                        ]
                    )

                # define materials
                for set_name in materials:
                    lines.extend(
                        [
                            f"*MATERIAL, NAME=MAT-{set_name}",
                            "*ELASTIC",
                            f"{self.RVE_variables[f'{set_name.lower()}_material'][0]}, {self.RVE_variables[f'{set_name.lower()}_material'][1]}",
                            "*DENSITY",
                            "1",

                        ]
                    )

                # saving file
                with open(
                    os.path.join(
                        self.directory.case_folder,
                        "inp",
                        f"{self.case_number}_fairing_mesh.inp",
                    ),
                    "w",
                ) as f:
                    f.write("\n".join(lines))

            # Create tailored geometry
            create_tailored_geometry()

            # Create shell geometry
            create_shell_geometry()

            # Add reference nodes
            create_reference_nodes()

            # Create mesh
            create_mesh()

            # Create geometry
            create_abaqus_geometry()

        print(f"\tStarting GMSH for {self.directory.case_name} - {self.case_number}")

        # Initialize GMSH
        gmsh.initialize()

        # Set GMSH options
        eps = np.finfo(np.float32).eps  # machine epsilon
        gmsh.option.setNumber(
            "General.Verbosity", 0
        )  # Reduce verbosity, print error and warnings

        # Set GMSH model name
        MODEL = gmsh.model
        CAD = MODEL.geo
        MESH = MODEL.mesh

        # GMSH processes
        match self.var["model_fidelity"]:
            case "equivalent":
                create_equivalent_model()
            case "explicit":
                create_explicit_model()
            case _:
                raise ValueError(f"ERROR: Undefined model_fidelity. Recieved {self.var['model_fidelity']}")

        # Show the model, if needed
        if bool_show:
            gmsh.fltk.run()

        gmsh.finalize()

class FairingAnalysis(FairingGeometry):

    def __init__(self, variables=None, RVE_identifier=None, directory=Utils.Directory(), case_number: int = 0):
        super().__init__(variables, RVE_identifier, directory, case_number)

    def create_loading_steps(self):
        """
        Add loading steps to the simulation.
        """

        def define_solver_step(name, keywords, data_lines, solver, Output_Requested, max_incr_size = 1, max_num_incr = 1000):
            """
            A helper function to define the solver step.
            """
            # Defining Folding Step
            match solver:
                case "newton":
                    lines.append(f"*STEP, NAME={name}, NLGEOM=YES, INC={max_num_incr}")
                    lines.append("*STATIC")  # STATIC SOLVER, IMPLICIT
                    lines.append(f"{max_incr_size}, 1, 1E-9, {max_incr_size}")
                    lines.append("*"+",".join(keywords))
                case 'linear':
                    lines.append(f"*STEP, NAME={name}, NLGEOM=NO, INC={max_num_incr}")
                    lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
                    lines.append(f"{max_incr_size}, 1, 1E-9, {max_incr_size}")
                    lines.append("*"+",".join(keywords))
                case 'dynamic':
                    lines.append(f"*AMPLITUDE, NAME={name}, DEFINITION=SMOOTH STEP")
                    lines.append("0, 0, 1, 1")
                    lines.append(f"*STEP, NAME={name}, NLGEOM=YES, INC={max_num_incr}")
                    lines.append("*DYNAMIC, APPLICATION=QUASI-STATIC")
                    lines.append(f"{max_incr_size}, 1, 1E-9, {max_incr_size}")
                    lines.append("*" + ",".join(keywords+[f"AMPLITUDE={name}"])) 
                case "explicit":
                    lines.append(f"*AMPLITUDE, NAME={name}, DEFINITION=SMOOTH STEP")
                    lines.append("0, 0, 1, 1")
                    lines.append(f"*STEP, NAME={name}, NLGEOM=YES")
                    lines.append("*DYNAMIC, EXPLICIT, ELEMENT BY ELEMENT")
                    lines.append(f" , 1, , {max_incr_size}")
                    lines.append(
                        f"*VARIABLE MASS SCALING, DT={max_incr_size}, TYPE=BELOW MIN, FREQUENCY=1"
                    )
                    lines.append("*" + ",".join(keywords + [f"AMPLITUDE={name}"]))
                case _:
                    raise(f"ERROR: Recieved undefined solver type: {solver}")
            # loading
            for line in data_lines:
                lines.append(line)

            # OUTPUT REQUESTS
            if not Output_Requested:
                lines.extend(Output_Requests)
                Output_Requested = True
            lines.append("*END STEP")

            return Output_Requested

        # initialise a list
        lines = []

        # add rigid body for ribs
        # floating rib
        if self.var["num_floating_ribs"]>0:
            for rib_i in range(self.var["num_floating_ribs"]):
                lines.append(f"*RIGID BODY, REF NODE=FLOAT_{rib_i+1}, TIE NSET=RIB_{rib_i+1}")
        # pivoting rib
        lines.append("*RIGID BODY, REF NODE=PIVOT, TIE NSET=RIB_P")

        # boundary conditions
        lines.extend(["*BOUNDARY", "RIB_0, ENCASTRE"])
        # pivoting rib constraints
        if self.var["pre_strain"] != 0.0:
            for dof in [1, 3, 5, 6]:
                lines.append(f"PIVOT, {dof}, {dof}")
        else:
            for dof in [1, 2, 3, 5, 6]:
                lines.append(f"PIVOT, {dof}, {dof}")

        # Output request
        Output_Requests = [
            "*OUTPUT, FIELD",
            "*ELEMENT OUTPUT, ELSET=OUTER_SURFACE, DIRECTION=YES",
            "SE, SK", # In Material Axis
            "*OUTPUT, HISTORY", 
            "*NODE OUTPUT, NSET=PIVOT",  
            "RF, RM, U, UR",  #  In global axis
            "*OUTPUT, FIELD, VAR=PRESELECT",
            "*OUTPUT, HISTORY, VAR=PRESELECT",
        ]
        Output_Requested = False

        steps = []

        # pre-strain step
        if self.var["pre_strain"] !=0.0:
            value = self.var['pre_strain'] * ((self.var["fairing_span"]/(1+self.var["pre_strain"]))) / 2
            Output_Requested = define_solver_step(
                name="PRESTRAIN",
                keywords=["BOUNDARY"],
                data_lines=[f"PIVOT, 2, 2, {value}"],
                solver="newton" if self.var["solver"]!="explicit" else "explicit",
                Output_Requested=Output_Requested,
                max_incr_size=self.var["solver_increment_settings"]["max_incr_size"],
                max_num_incr=self.var["solver_increment_settings"]["max_num_incr"]
            )

        # pressure
        if self.var["bool_pressure"] == True:
            Output_Requested = define_solver_step(
                name="PRESSURE",
                keywords=["DSLOAD"],
                data_lines=[f"PIVOT, 2, 2, {value}"],
                solver="newton"
                if self.var["solver"] != "explicit"
                else "explicit",
                Output_Requested=Output_Requested,
                max_incr_size=self.var["solver_increment_settings"][
                    "max_incr_size"
                ],
                max_num_incr=self.var["solver_increment_settings"][
                    "max_num_incr"
                ],
            )

        # folding step
        Output_Requested = define_solver_step(
            name="FOLDING",
            keywords=["BOUNDARY"],
            data_lines=[f"PIVOT, 4, 4, {self.var['rotation_angle']}"],
            solver=self.var["solver"],
            Output_Requested=Output_Requested,
            max_incr_size=min(
                np.deg2rad(1.0) / self.var["rotation_angle"],
                self.var["solver_increment_settings"]["max_incr_size"],
            ),
            max_num_incr=self.var["solver_increment_settings"]["max_num_incr"],
        )

        # saving file
        with open(
            os.path.join(
                self.directory.case_folder,
                "inp",
                "fairing_loading.inp",
            ),
            "w",
        ) as f:
            f.write("\n".join(lines))

    @Utils.logger
    def run_abaqus(
        self, abaqus_path: str = "C:\\SIMULIA\\Commands\\abaqus", num_core: int = 4
    ):
        """Runs the Abaqus analysis for the generated FR model."""

        print(f"\tStarting Abaqus for {self.directory.case_name} - {self.case_number}")

        # Create loading steps
        self.create_loading_steps()

        # Assemble the final input file by joining geometry and loading steps
        geometry_file = os.path.join(
            self.directory.case_folder, "inp", f"{self.case_number}_fairing_mesh.inp"
        )
        loading_file = os.path.join(
            self.directory.case_folder, "inp", "fairing_loading.inp"
        )
        self.job_file = os.path.join(
            self.directory.case_folder, "inp", f"{self.case_number}_FR.inp"
        )
        with open(self.job_file, "w") as outfile:
            for fname in [geometry_file, loading_file]:
                with open(fname) as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")

        # Job definition
        self.log_file = os.path.join(
            self.directory.case_folder, "log", f"{self.case_number}_FR_Abaqus.log"
        )
        self.run_folder = os.path.join(
            self.directory.abaqus_folder, f"{self.case_number}"
        )
        os.makedirs(self.run_folder, exist_ok=True)

        try:
            # Run the Abaqus analysis
            command = (
                f"{abaqus_path} analysis double=both job={self.case_number}_FR input={self.job_file} cpus="
                f"{num_core} mp_mode=thread license_model=LEGACY interactive" 
            )
            Utils.run_subprocess(command, self.run_folder, self.log_file)
            # os.system("cd " + self.run_folder + "&&" + "echo y | " + command)
        except:
            traceback.print_exc()
            with open(self.log_file, "a") as f:
                f.write(
                    f"ERROR: Cannot run Abaqus for fairing case number {self.case_number}\n"
                )

        # Copy result files
        for ext in ["odb", "msg", "dat", "sta"]:
            src = os.path.join(self.run_folder, f"{self.case_number}_FR.{ext}")
            dst = os.path.join(
                self.directory.case_folder, ext, f"{self.case_number}_FR.{ext}"
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
                        f"{self.case_number}_FR_Abaqus.log",
                    ),
                    "a",
                ) as f:
                    f.write(
                        f"ERROR: Cannot copy result file {ext} for case number {self.case_number}\n"
                    )

    def extract_fairing_data(
        self, abaqus_path: str = "C:\\SIMULIA\\Commands\\abaqus.bat"
    ):

        # Job definition
        self.log_file = os.path.join(
            self.directory.case_folder, "log", f"{self.case_number}_FR_Abaqus.log"
        )
        self.run_folder = os.path.join(
            self.directory.abaqus_folder, f"{self.case_number}"
        )
        os.makedirs(self.run_folder, exist_ok=True)

        # Extract results using a Python script
        script_file = os.path.abspath("abaqus_fairing_data.py")
        command = f"{abaqus_path} python {script_file} -- {self.directory.case_folder} {self.case_number}"
        Utils.run_subprocess(command, self.run_folder, self.log_file)

    def evaluate_fairing_distortion(self, surface_nodes_U):
        """
        Evaluate the distortion of the fairing surface as an area weighted average.
        """
        # load surface mesh data saved in grid data function
        surface_mesh = Utils.ReadWriteOps.load_object(
            os.path.join(
                self.directory.case_folder,
                "data",
                f"{self.case_number}_fairing_mesh_data",
            ),
            "pickle",
        )
        surface_nodes_set = surface_mesh["surface_nodes"]
        surface_nodes_coords = surface_mesh["surface_nodes_coords"]
        surface_elements_set = surface_mesh["surface_elements"]
        surface_elements_nodes = surface_mesh["surface_elements_nodes"]
        top_surface_elements_set = surface_mesh["top_surface_elements"]
        bottom_surface_elements_set = surface_mesh["bottom_surface_elements"]


        # index of surface nodes in surface_nodes_set
        node_idx_in_set = Utils.indices(surface_nodes_set, surface_elements_nodes.flatten()).reshape(surface_elements_nodes.shape)

        # index of top and bottom surface elements in surface_elements_set
        top_surface_elements_indices = np.argwhere(np.isin(surface_elements_set, top_surface_elements_set)).squeeze()
        bottom_surface_elements_indices = np.argwhere(np.isin(surface_elements_set, bottom_surface_elements_set)).squeeze()

        increments = len(list(surface_nodes_U.values())[0])
        fairing_distortion = np.zeros((increments))
        max_vertical_displacement = np.zeros((increments))
        for increment in range(increments):

            # surface nodes displacements
            surface_nodes_displacements = np.array(
                [surface_nodes_U[node_num][increment] for node_num in surface_nodes_set],
                dtype=np.float32
            ) # ordered to surface_nodes_set
            displaced_surface_nodes = surface_nodes_coords + surface_nodes_displacements

            # Maximum vertical displacement
            max_vertical_displacement[increment] = np.max(
                abs(surface_nodes_displacements[:, 2])
            )

            # surface element centroids
            surface_elements_centroids_displacement = (
                surface_nodes_displacements[node_idx_in_set][:, :, :].mean(
                    axis=1
                )
            )

            # Element surface area
            surface_element_areas = np.zeros(surface_elements_set.shape[0])
            for i, elem_nodes in enumerate(surface_elements_nodes):
                coords = displaced_surface_nodes[node_idx_in_set[i]]
                # Compute the area using the cross product of two edges
                if coords.shape[0] == 3: # triangular element
                    vec1 = coords[1] - coords[0]
                    vec2 = coords[2] - coords[0]
                elif coords.shape[0] == 4: # quadrilateral element
                    vec1 = coords[2] - coords[0]
                    vec2 = coords[3] - coords[1]
                surface_element_areas[i] = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            total_surface_area = np.sum(surface_element_areas)

            # Calculate average thickness reduction
            area_weighted_top_U3 = np.sum(surface_element_areas[top_surface_elements_indices]*surface_elements_centroids_displacement[top_surface_elements_indices, 2])
            area_weighted_bottom_U3 = np.sum(surface_element_areas[bottom_surface_elements_indices]*surface_elements_centroids_displacement[bottom_surface_elements_indices, 2])
            average_thickness_reduction = (-area_weighted_top_U3 + area_weighted_bottom_U3)/total_surface_area

            # # debugging prints
            # print("---")
            # print(f"Increment {increment+1}/{increments}")
            # print("Weighted Top U3:", area_weighted_top_U3)
            # print("Weighted Bottom U3:", area_weighted_bottom_U3)
            # print("Surface Area:", total_surface_area)
            # print("Average Thickness Reduction:", average_thickness_reduction)

            # update output
            fairing_distortion[increment] = average_thickness_reduction

        return fairing_distortion, max_vertical_displacement

    def post_process_results(self):
        # load fairing data
        fairing_data =Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "data", f"{self.case_number}_fairing_data"), "pickle")

        # Check hinge node DOF
        tolerance = 1e-6
        U = fairing_data.hinge_node["U"]
        UR = fairing_data.hinge_node["UR"]
        fixed_DOF = np.column_stack(
            (
                U[np.ix_(np.arange(U.shape[0]), [0, 2])],
                UR[np.ix_(np.arange(UR.shape[0]), [1, 2])],
            )
        )
        if np.any(np.abs(fixed_DOF) > tolerance):
            raise ValueError("Hinge node has non-zero displacement in a fixed degree of freedom. (i.e., either X, Z traslation or Y, Z rotation)")

        # Calculate hinge rotation and torque
        distortion, max_vertical_displacement = self.evaluate_fairing_distortion(fairing_data.surface_nodes_U)
        self.FairingResponse = {
            "Rotation": np.rad2deg(fairing_data.hinge_node["UR"][:, 0]),  # [deg]
            "Torque": fairing_data.hinge_node["RM"][:, 0],  # [Nm]
            "Distortion": distortion,
            "Maximum Displacement": max_vertical_displacement,
            # "Chord": self.var["fairing_chord"],  # [m]
        }

        # save in excel file
        df = pd.DataFrame(self.FairingResponse)
        df.to_excel(os.path.join(self.directory.case_folder, "data", f"{self.case_number}_fairing_response.xlsx"), index=False)

        # Plotting
        Utils.Plots.fairing_response(
            {self.case_number: self.FairingResponse},
            save_path=os.path.join(
                self.directory.case_folder,
                "fig",
                f"{self.case_number}_fairing_response.png",
            ),
            show=False,
        )

        if self.var["model_fidelity"] == "equivalent":
            # Trace Lattice
            trace_data = Tailoring.Tracer(self.directory, self.case_number)
            trace_data.analysis()

    def reduce_step_size(self, factor):
        """Reduce the step size of the solver by a given factor and saves the updated variables."""

        # update step size
        self.var["solver_increment_settings"]["max_incr_size"] *= factor

        # saving updated variables
        Utils.ReadWriteOps.save_object(
            self.var,
            os.path.join(self.directory.case_folder, "input", f"{self.case_number}_FR"),
            method="json",
        )

    def check_convergence(self):
        """Check the convergence of the simulation by reading the .sta file."""

        # read sta file
        with open(
                os.path.join(self.directory.case_folder, "sta", f"{self.case_number}_FR.sta"), "r"
            ) as f:
            sta_file = f.read().strip().splitlines()

        # check if last line indicates successful completion
        if sta_file[-1].strip() == "THE ANALYSIS HAS COMPLETED SUCCESSFULLY":
            return True
        else:
            return False  

    @Utils.logger
    def analysis(self):
        print(f"Starting {self.__class__.__name__} analysis {self.directory.case_name} - {self.case_number}")

        # Generate mesh
        self.generate_mesh()

        # Run simulation
        self.run_abaqus(num_core=10)

        # If not converged, reduce step size and re-run
        num_attempts = int(
            self.var["solver_increment_settings"]["max_incr_size"]
            / self.var["solver_increment_settings"]["max_incr_size"]
        )
        for attempt in range(num_attempts):
            # Check convergence
            if self.check_convergence():
                print("Simulation converged successfully.")
                break
            else:
                print(f"Simulation did not converge. Attempt {attempt+1}/{num_attempts}. Reducing step size and re-running.")
                self.reduce_step_size(factor=0.1)
                self.run_abaqus(num_core=10)

        # Extract results
        self.extract_fairing_data()

        # Post-processing
        self.post_process_results()


if __name__ == "__main__":
    directory = Utils.Directory(case_name="r3_tailoring_without_pressure")

    # RVE
    RVE = RVE(
        directory=directory,
        case_number=0
    )
    RVE.analysis()

    # Fairing definition
    fairing = FairingAnalysis(
        variables={
            "rotation_angle": Utils.Units.deg2rad(40.0),
            "solver":"newton",
            "model_fidelity_settings":{
                "equivalent":{
                    "bool_isotropic": False, # [True, False], if true core properites use, else equivalent panel properties
                }
            },
        },  
        directory=directory,
        case_number=0,
        RVE_identifier=0
    )
    fairing.analysis()

    # # debuigging
    # fairing.generate_mesh()
    # fairing.extract_fairing_data()
    # fairing.post_process_results()

    # Spatially tailored model
    for count, field_angle in enumerate([0, 5, 10, 15], start=1):
        tailored = FairingAnalysis(
            variables={
                "element_size": Utils.Units.mm2m(2.0),  # selected 2mm
                "rotation_angle": Utils.Units.deg2rad(40.0),
                "solver": "dynamic",
                "solver_increment_settings": {
                    "max_incr_size": 1e-2,
                    "min_incr_size": 1e-5,
                    "max_num_incr": 1000,
                },
                "model_fidelity": "explicit",  # either of ["equivalent", "explicit", "fullscale"]
                "model_fidelity_settings": {
                    "equivalent": {
                        "bool_isotropic": True,  # [True, False], if true core properites use, else equivalent panel properties
                    },
                    "explicit": {
                        "reference_case": 0,  # int, case number of the reference case from which the explicit model is generated
                        "reference_field": field_angle,  # int, rotation angle for the folding wingtip from whose deformation the field is extracted
                    },
                },
            },
            RVE_identifier=0,
            directory=directory,
            case_number=count,
        )
        tailored.analysis()

        # tailored.generate_mesh()
        # tailored.run_abaqus(num_core=10)
        # tailored.extract_fairing_data()
        # tailored.post_process_results()


    # ## Test case
    # tailored = FairingAnalysis(
    #     variables={
    #         "element_size": 0.02,
    #         "rotation_angle": Utils.Units.deg2rad(40.0),
    #         "solver": "dynamic",  # either of ['linear', "newton", "riks", "dynamic", explicit]
    #         "solver_increment_settings": {
    #             "max_incr_size": 1e-2,  # maximum increment size
    #             "min_incr_size": 1e-5,  # minimum increment size
    #             "max_num_incr": 1000,  # maximum number of increments
    #         },
    #         "model_fidelity": "explicit",  # either of ["equivalent", "explicit", "fullscale"]
    #         "model_fidelity_settings": {
    #             "equivalent": {
    #                 "bool_isotropic": True,  # [True, False], if true core properites use, else equivalent panel properties
    #             },
    #             "explicit": {
    #                 "reference_case": 0,  # int, case number of the reference case from which the explicit model is generated
    #                 "reference_field": 0,  # int, rotation angle for the folding wingtip from whose deformation the field is extracted
    #             },
    #         },
    #     },
    #     RVE_identifier=0,
    #     directory=directory,
    #     case_number=1,
    # )
    # tailored.analysis()

    # tailored.generate_mesh()
    # tailored.run_abaqus(num_core=10)
    # tailored.extract_fairing_data()
    # tailored.post_process_results()

