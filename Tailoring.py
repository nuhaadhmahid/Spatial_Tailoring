
# Tailoring.py

import Utils
from Fairing import FairingData

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import Delaunay
from scipy import interpolate



class Lattice:
    def __init__(self, directory=Utils.Directory(), case_number:int= 0):
        self.directory = directory
        self.case_number = case_number

        # Data holders
        self.trace_lines = {}
        self.lattice_lines = {}

        # Load data
        self.load_field_data()
        self.load_cell_data()
        self.load_element_data()

    def load_field_data(self):
        fairing_data: FairingData = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "data", f"{self.case_number}_fairing_data"), "pickle")

        self.shell_equivalent_SE = fairing_data.shell_equivalent_SE
        self.shell_equivalent_SK = fairing_data.shell_equivalent_SK
        self.rotation_angles = fairing_data.hinge_node["UR"][:, 0]

    def load_cell_data(self):
        RVE_derived = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "input", f"{self.case_number}_UC_derived"), "json")
        RVE_input = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "input", f"{self.case_number}_UC"), "json")

        self.cell_dimensions = [RVE_derived["lx"], RVE_derived["ly"], RVE_derived["lz"]]
        self.chevron_angle = RVE_input["chevron_angle"]

    def load_element_data(self):
        mesh_data = Utils.ReadWriteOps.load_object(
            os.path.join(self.directory.case_folder, "data", f"{self.case_number}_fairing_mesh_data"), 
            "pickle"
        )
        self.surface_nodes = mesh_data["surface_nodes"]
        self.surface_nodes_coords = mesh_data["surface_nodes_coords"].astype(np.float32)
        self.surface_elements = mesh_data["surface_elements"]
        self.surface_elements_nodes = mesh_data["surface_elements_nodes"]
        self.surface_element_centroids = mesh_data["surface_element_centroids"].astype(np.float32)
        self.corner_element = mesh_data["corner_element"]
        self.corner_node = mesh_data["corner_node"]
        self.element_grid_shape = mesh_data["element_grid_shape"]

    def create_element_grid(self, bool_plot=False):
        """
        Construct a grid of element indices starting from the top trailing edge and inboard rib corner.
        """

        # find next element in the grid in the given direction using the shared nodes
        def next_element_index(current_element_index, current_node_index, next_node_index):
            """
            Find the next element index in the grid by checking shared nodes. Given that node numbering is consistent for all elements, the shared nodes can be used to identify the next element (e.g, for a quad element, the right edge is defined by node indices [1, 2], while the left edge is defined by node indices [0, 3] -  so current_node_index is set to [1, 2] and next_node_index is set to [2, 3] to find the next element to the right).
            Args:
                current_element_index (int): Current element index.
                current_node_index (list): Indices of the current element nodes shared with the next element.
                next_node_index (list): Indices of next element nodes to be checked.
            """
            # nodes to be checked other elements
            propagating_node_numbers = self.surface_elements_nodes[current_element_index, current_node_index] 
            # find neighbouring element with shared nodes
            next_element_index = np.argwhere((self.surface_elements_nodes[:, next_node_index] == propagating_node_numbers).all(axis=1)).squeeze()
            # next_element_index = next_element_index[next_element_index != current_element_index]

            return next_element_index

        # initilise element grid
        self.element_index_grid = np.empty(self.element_grid_shape, dtype=np.int32)

        # element grid sorting
        VR_edge_index, VL_edge_index = np.array([1,2]), np.array([0,3]) # mapping from current cell to next cell
        HT_edge_index, HB_edge_index = np.array([3,2]), np.array([0,1])
        for i in range(self.element_grid_shape[0]): # looping over each chordwise column of cells
            if i==0: 
                self.element_index_grid[i,0] = np.argwhere(self.surface_elements==self.corner_element).squeeze()
            else: 
                self.element_index_grid[i,0] = next_element_index(self.element_index_grid[i-1,0], VR_edge_index, VL_edge_index)
            for j in range(self.element_grid_shape[1]): # looping over each cells of chordwise column
                if j==0: 
                    pass
                else: 
                    self.element_index_grid[i,j] = next_element_index(self.element_index_grid[i,j-1], HT_edge_index, HB_edge_index)

        # Plotting
        if bool_plot:
            Utils.Plots.grid(
                {
                    "coords":self.surface_element_centroids[self.element_index_grid],
                    "labels":self.surface_elements[self.element_index_grid]
                },
                os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_centroids_grid.svg"),
                show=False
            )

    def create_node_grid(self, bool_plot=False):
        """
        Construct a grid of node indices based on the element grid.
        """
        # create node grid
        self.node_index_grid = np.zeros((self.element_grid_shape[0]+1, self.element_grid_shape[1]+1), dtype=np.int32)

        # corner
        self.node_index_grid[0,0] = Utils.indices(
            self.surface_nodes, self.corner_node
        )
        # Rib 0
        self.node_index_grid[0, 1:] = Utils.indices(
            self.surface_nodes,
            self.surface_elements_nodes[self.element_index_grid[0, :].ravel(), 3]
        )
        # # top TE
        self.node_index_grid[1:,0] = Utils.indices(
            self.surface_nodes,
            self.surface_elements_nodes[self.element_index_grid[:, 0].ravel(), 1]
        )
        # rest
        self.node_index_grid[1:,1:] = Utils.indices(
            self.surface_nodes,
            self.surface_elements_nodes[self.element_index_grid[:, :].ravel(), 2]
        ).reshape(self.node_index_grid[1:,1:].shape)

        # Plotting
        if bool_plot:
            Utils.Plots.grid(
                {
                    "coords":self.surface_nodes_coords[self.node_index_grid],
                    "labels":self.surface_nodes[self.node_index_grid]
                },
                os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_nodes_grid.svg"),
                show=False
            )

    def flatten_grid(self, bool_plot=False):
        """
        Flattening the grid along the surface.
        """

        # Flattening 3D node grid to 2D
        self.nodes_grid_2D = np.zeros(*np.c_[*self.node_index_grid.shape, 2], dtype=np.float32)

        # Moving origin to leading edge
        FE_j_index = np.linalg.norm(self.surface_nodes_coords[self.node_index_grid][0, :], axis=1).argmin()

        # Cumulative sum of differences up to each index (along axis 0, span)
        self.nodes_grid_2D[1:, :, 1] = np.add.accumulate(
            np.linalg.norm(
                np.diff(
                    self.surface_nodes_coords[self.node_index_grid], 
                    axis=0
                    ), 
                axis=2
            ),
            axis=0,
        )

        # Cumulative sum of differences up to each index (along axis 1, chord)
        diffs = np.linalg.norm(
            np.diff(self.surface_nodes_coords[self.node_index_grid], axis=1),
            axis=2
        )
        # bottom
        self.nodes_grid_2D[:, FE_j_index+1:, 0] = -np.cumsum(diffs[:,FE_j_index:], axis=1)
        # top
        cum_dists = np.cumsum(diffs[:, :FE_j_index-1:-1], axis=1)  # reverse
        self.nodes_grid_2D[:, :FE_j_index, 0] = cum_dists[:, ::-1]

        # Plotting
        if bool_plot:
            Utils.Plots.grid_split(
                {
                    "coords":self.nodes_grid_2D,
                    "labels":self.surface_nodes[self.node_index_grid]
                },
                os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_nodes_grid_2D.svg"),
                show=False
            )

        # Border nodes
        self.border_nodes_2D = np.concatenate((
            self.nodes_grid_2D[:-1, 0], # bottom
            self.nodes_grid_2D[-1, :-1], # left
            np.flip(self.nodes_grid_2D[1:, -1], axis=0), # top
            np.flip(self.nodes_grid_2D[0, 1:], axis=0)  # right
        ), axis=0)

        # element centroid grid
        self.centroids_grid_2D = np.zeros(*np.c_[*self.element_index_grid.shape, 2], dtype=np.float32)
        self.centroids_grid_2D[:, :] = 0.25 * (
            self.nodes_grid_2D[:-1, :-1]
            + self.nodes_grid_2D[:-1, 1:]
            + self.nodes_grid_2D[1:, :-1]
            + self.nodes_grid_2D[1:, 1:]
        )

        # Plotting
        if bool_plot:
            Utils.Plots.grid_split(
                {
                    "coords":self.centroids_grid_2D,
                    "labels":self.surface_elements[self.element_index_grid],
                    "border":self.border_nodes_2D
                },
                os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_centroids_grid_2D.svg"),
                show=False
            )

    @staticmethod
    def rotate_voigt(voigt_in, type, CCW_angle_rad):
        """
        Rotate a 2D voigt vector (strain or stress) by a given counter-clockwise angle. From https://web.mit.edu/course/3/3.11/www/modules/trans.pdf

        input :: 1darray: voigt_local, str:type of variable, float:CCW_angle_rad
        return :: 1darray: voigt_global
        """

        C = np.cos(CCW_angle_rad)
        S = np.sin(CCW_angle_rad)
        multiplier = 0.5 if type in ['SE', 'SK'] else 1.0

        # Rehshape to (3, n)
        voigt_local = np.atleast_2d(voigt_in.copy().T.reshape((3, -1)))

        # voigt_in = R^(-1) @ epsilon
        voigt_local[2] *= multiplier

        # voigt_out = R @ voigt_in
        R_2D_Array = np.array([
            [C**2,   S**2,   2*S*C      ],
            [S**2,   C**2,   -2*S*C     ],
            [-S*C,   S*C,    (C**2-S**2)]
        ])
        voigt_out = R_2D_Array @ voigt_local

        # voigt_out = R @ voigt_out
        voigt_out[2] /= multiplier

        # Reshape back to original shape
        voigt_out = voigt_out.reshape(voigt_in.T.shape).T

        return voigt_out

    @staticmethod
    def rotate_vector(vector_in, CCW_angle_rad):
        """
        Rotate a 2D vector by a given counter-clockwise angle in radians.

        input :: 1darray: vector_in, float:CCW_angle_rad
        return :: 1darray: vector_out
        """

        C = np.cos(CCW_angle_rad)
        S = np.sin(CCW_angle_rad)

        # Rehshape to (3, n)
        vector_local = np.atleast_2d(vector_in.copy().T.reshape((2, -1)))

        # voigt_out = R @ v
        R_2D_Array = np.array([
            [C,   S],
            [-S,   C]
        ])
        vector_out = R_2D_Array @ vector_local

        # Reshape back to original shape
        vector_out = vector_out.reshape(vector_in.T.shape).T

        return vector_out

    def create_field_vectors(self, bool_plot=False):

        # Collecting the field data
        self.shell_equivalent_SE = np.array([self.shell_equivalent_SE[number] for number in self.surface_elements])
        self.shell_equivalent_SK = np.array([self.shell_equivalent_SK[number] for number in self.surface_elements])

        # Rotating the field data to local coordinate system
        self.shell_equivalent_SE =self.rotate_voigt(self.shell_equivalent_SE, "SE", -np.pi/2)
        self.shell_equivalent_SK =self.rotate_voigt(self.shell_equivalent_SK, "SK", -np.pi/2)

        # Target deformation points
        tolerance = np.deg2rad(1e-1)
        self.indices = np.unique(np.r_[
            0, 
            np.argwhere(
                np.abs(
                    np.rad2deg(self.rotation_angles[np.newaxis,...]) 
                    - np.array([5, 10, 15, 20])[..., np.newaxis]
                ) <= tolerance
            )[:, 1]
        ]) # including undeformed and angles closest to 5, 10, 15, 20 deg
        print(f"Lattice field for folding angles (deg): {[f'{angle:.1f}' for angle in np.rad2deg(self.rotation_angles[self.indices])]}")

        # constructing the interpolation function for the principal directions of the field
        self.f_interp = {}
        # looping over each angle
        for ix in self.indices:
            # initialise
            key = f'{np.rad2deg(self.rotation_angles[ix]):.0f}'
            self.f_interp[key] = {}

            if key=='0':
                for variable in ["SE", "SK"]:
                    self.f_interp[key][f'vecs_{variable}']=[
                        lambda x: np.repeat(
                            np.array([-1,0])[np.newaxis, ...],
                            x.shape[0], axis=0
                        ),
                        lambda x: np.repeat(
                            np.array([0,-1])[np.newaxis, ...],
                            x.shape[0], axis=0
                        )
                    ]
                    self.f_interp[key][f'vals_{variable}']=[
                        lambda x: np.repeat(
                            np.array([1])[np.newaxis, ...],
                            x.shape[0], axis=0
                        ),
                        lambda x: np.repeat(
                            np.array([1])[np.newaxis, ...],
                            x.shape[0], axis=0
                        )
                    ]

            else:

                def field_to_tensor(field):
                    """
                    Convert a 2D Voigt vector to a 2x2 tensor.
                    """
                    tensor = np.zeros((field.shape[0], 2, 2), dtype=field.dtype)
                    tensor[:, 0, 0] = field[:, 0]
                    tensor[:, 0, 1] = field[:, 2] / 2
                    tensor[:, 1, 0] = field[:, 2] / 2
                    tensor[:, 1, 1] = field[:, 1]
                    return tensor

                # extracting field at the given angle
                tensor_SE = field_to_tensor(self.shell_equivalent_SE[:,ix])
                tensor_SK = field_to_tensor(self.shell_equivalent_SK[:,ix])

                # eigen decomposition
                eivanvals_SE, eiganvecs_SE = np.linalg.eig(tensor_SE)
                eivanvals_SK, eiganvecs_SK = np.linalg.eig(tensor_SK)

                # organised to the grid format
                grid_eigenvecs_SE = eiganvecs_SE[self.element_index_grid]
                grid_eigenvals_SE = eivanvals_SE[self.element_index_grid]
                grid_eigenvecs_SK = eiganvecs_SK[self.element_index_grid]
                grid_eigenvals_SK = eivanvals_SK[self.element_index_grid]

                def ensure_continuity(vectors, axis=0):
                    """
                    Ensuring continuity of the eigenvectors in the given direction. e.g., 0 for spanwise and 1 for chordwise
                    """
                    # Store the shape of the vectors array for use in indexing operations below
                    shape = vectors.shape
                    # row_offset and col_offset determine which axis to ensure continuity along
                    row_offset = 1 if axis == 0 else 0  # Offset for rows if axis==0
                    col_offset = 1 if axis == 1 else 0  # Offset for columns if axis==1
                    indices = np.array([
                        [np.arange(shape[0] - row_offset), np.arange(row_offset, shape[0])],
                        [np.arange(shape[1] - col_offset), np.arange(col_offset, shape[1])]
                    ], dtype=object)

                    # helper function for indexing
                    def f_ix(x):
                        return np.ix_(*indices[:,x])

                    # index of vector aligned with previous vector
                    index = np.abs(vectors[f_ix(0)] @ vectors[f_ix(1)].transpose(0,1,3,2)).argmax(axis=-1) 
                    index[..., 0] = np.cumsum(index[..., 0], axis=1)%2
                    index[..., 1] = (index[..., 0]+1)%2
                    assert np.all(index[...,0]!= index[...,1]) 
                    # swapping and flipping vectors to ensure continuity
                    vectors[f_ix(1)] = np.take_along_axis(
                        vectors[f_ix(1)],
                        index[:,:,:,None],
                        axis=2
                    ) 
                    sign = np.cumprod(np.sign(
                        np.einsum('...i,...i->...', vectors[f_ix(0)], vectors[f_ix(1)])
                    ), axis=1)
                    vectors[f_ix(1)] *= sign[...,np.newaxis]

                    return vectors

                # ensuring continuity of the eigenvectors in the chordwise direction
                grid_eigenvecs_SE = ensure_continuity(grid_eigenvecs_SE, 0)
                grid_eigenvecs_SK = ensure_continuity(grid_eigenvecs_SK, 0)

                # Plotting
                if bool_plot:
                    for type, vec in zip(['SE', 'SK'], [grid_eigenvecs_SE, grid_eigenvecs_SK]):
                        Utils.Plots.grid_split(
                            {
                                "coords":self.centroids_grid_2D,
                                # "labels":self.element_numbers[self.element_index_grid],
                                "vectors_coords":self.centroids_grid_2D,
                                f"vectors_{type}":vec,
                                "border": self.border_nodes_2D
                            },
                            os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_centroids_quiver_2D_{type}_{ix}.svg"),
                            show=False
                        )

                # interpolation data
                centroids = self.centroids_grid_2D.reshape(-1, 2)
                values_SE = grid_eigenvals_SE.reshape(-1, 2)
                vectors_SE = grid_eigenvecs_SE.reshape(-1, 2,2) 
                values_SK = grid_eigenvals_SK.reshape(-1, 2)
                vectors_SK = grid_eigenvecs_SK.reshape(-1, 2,2)

                # RBF interpolation
                for param, data in zip(
                    ['vals_SE', 'vecs_SE', 'vals_SK', 'vecs_SK'],
                    [values_SE, vectors_SE, values_SK, vectors_SK]
                ):
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        raise ValueError(f"NaN or Inf values found in {param} at angle {key} degrees. Check the input field data for inconsistencies.")

                    self.f_interp[key][param] = [
                        interpolate.RBFInterpolator(centroids, data[:,0], neighbors=4, kernel='linear'),
                        interpolate.RBFInterpolator(centroids, data[:,1], neighbors=4, kernel='linear')
                    ]

        # Plotting interpolated field
        if bool_plot:
            mins = np.min( self.centroids_grid_2D.reshape(-1, 2), axis=0)
            maxs = np.max( self.centroids_grid_2D.reshape(-1, 2), axis=0)
            new_grid_array = np.stack(np.meshgrid(
                np.linspace(mins[0], maxs[0], self.element_grid_shape[1]//2),
                np.linspace(mins[1], maxs[1], self.element_grid_shape[0]//2)
            ), axis=-1).reshape(-1, 2)

            for key in self.f_interp.keys():
                Utils.Plots.grid_split(
                    {
                        "coords": self.centroids_grid_2D,
                        "vectors_coords": new_grid_array,
                        "vectors_SE": np.stack((self.f_interp[key]['vecs_SE'][0](new_grid_array), self.f_interp[key]['vecs_SE'][1](new_grid_array)), axis=1),
                        "vectors_SK": np.stack((self.f_interp[key]['vecs_SK'][0](new_grid_array), self.f_interp[key]['vecs_SK'][1](new_grid_array)), axis=1),
                        "border": self.border_nodes_2D
                    },
                    os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_centroids_quiver_2D_interp_{key}.svg"),
                    show=False
                )

    @staticmethod
    def runga_kutta(f_interp, seeds, boundary, step_size=0.01, max_steps=None, directions=np.array([-1,1])):

        if max_steps is None:
            max_steps = int(np.diff(boundary, axis=0).max()//step_size//2)*2+10

        points = np.empty((seeds.shape[0], max_steps+1, seeds.shape[1]))
        seed_ix = max_steps//2
        points[:,seed_ix,:] = seeds

        # Runge-Kutta integration
        for direction in directions:
            step = step_size * direction
            for i in range(1, max_steps//2+1):
                prev_index = seed_ix+(i-1)*direction
                curr_index = seed_ix+i*direction
                k1 = f_interp(points[:,prev_index,:]) * step
                k2 = f_interp(points[:,prev_index,:] + 0.5 * k1) * step
                k3 = f_interp(points[:,prev_index,:] + 0.5 * k2) * step
                k4 = f_interp(points[:,prev_index,:] + k3) * step
                points[:,curr_index,:] = points[:,prev_index,:] + (k1 + 2*k2 + 2*k3 + k4) / 6

            # trimming at the boundary
            lines = []
            for i in range(points.shape[0]):
                # indentify points within boundary
                mask = np.all((points[i] > boundary[0]) & (points[i] < boundary[1]), axis=1)
                # strip index of continuous set of points within the boundary to groups 
                true_indices = np.where(mask)[0]
                # add index before and after the strip to groups
                groups = np.split(true_indices, np.where(np.diff(true_indices) != 1)[0] + 1)
                # Extend group if possible
                extended_groups = []
                for group in groups:
                    if group.size != 0:
                        start = max(group[0] - 1, 0)
                        end = min(group[-1] + 1, mask.shape[0]-1)
                        extended_groups.append(np.arange(start, end + 1))
                # append line
                for group in extended_groups:
                    lines.append(points[i, group])

        # return points
        return lines

    @staticmethod
    def resample_lines(lines_1, lines_2):
        """
        Resample lines by finding all intersection points between two sets of polylines.
        
        Parameters:
            lines_1 (list): First set of polylines, each as np.ndarray of points with shape (n, 2)
            lines_2 (list): Second set of polylines, each as np.ndarray of points with shape (m, 2)
        
        Returns:
            list: List of arrays containing intersection points for each line in lines_1
        """
        # return new_lines_1
        num_lines_1, num_lines_2 = len(lines_1), len(lines_2)

        # Assembling inputs array
        i_lines_1, j_lines_2 = np.meshgrid(np.arange(num_lines_1), np.arange(num_lines_2), indexing='ij')
        iargs = np.stack((i_lines_1, j_lines_2), axis=2, dtype=object).reshape(-1,2)
        args = list(map(lambda x: (lines_1[x[0]], lines_2[x[1]]), iargs))

        # parallel execution
        num_workers = max((os.cpu_count() - 2, 1))
        with ThreadPoolExecutor(num_workers) as executor:
            results = list(executor.map(lambda ab: Utils.GeoOps.intersection_point(*ab), args))

        # reshaping results index to meshgrid
        idx = np.arange(num_lines_1*num_lines_2).reshape(num_lines_1, num_lines_2)

        # extracting new lines_1
        new_lines_1 = []
        for i in range(num_lines_1):
            line = np.array([results[k][0] for k in idx[i,:]]) # Get all results for the current line
            line = line[~np.all(np.isnan(line), axis=1), :] # Remove rows where all elements are NaN
            if line.shape[0] > 0:
                new_lines_1.append(line)

        # extracting new lines_2
        new_lines_2 = []
        for j in range(num_lines_2):
            line = np.array([results[k][0] for k in idx[:, j]]) # Get all results for the current column
            line = line[~np.all(np.isnan(line), axis=1), :] # Remove rows where all elements are NaN
            if line.shape[0] > 0:
                new_lines_2.append(line)

        return new_lines_1, new_lines_2
    
    @staticmethod
    def trimming_line(lines, boundary, limits):
        """
        Trim lines at the boundary by finding intersections and keeping only the portions inside.
        
        Parameters:
            lines (list): List of polylines to trim, each as np.ndarray of points with shape (n, 2)
            boundary (list): List of boundary lines, each as np.ndarray of points with shape (n, 2)
            limits (np.ndarray): Array of shape (2, 2) containing min and max coordinates of boundary
            
        Returns:
            list: List of trimmed lines that are inside the boundary
        """
        new_lines = []
        for i, line in enumerate(lines):
            # Check which points are inside the boundary
            bool_inside = np.all(np.concatenate((line >= limits[0], line <= limits[1]), axis=1), axis=1)
            
            # Find segments that cross the boundary
            transition_indexes = np.where(bool_inside[:-1] != bool_inside[1:])[0]
            
            # If no transitions, check if the line is completely inside or outside
            if transition_indexes.size == 0:
                if np.any(bool_inside):
                    # Line is completely inside, keep it
                    new_lines.append(line)
                    
                continue
                
            # Find intersection points with the boundary
            intersection_points = []
            for index in transition_indexes:
                for b_line in boundary:
                    point, _ = Utils.GeoOps.intersection_point(line[index:index+2, :], b_line)
                    if not np.any(np.isnan(point)):
                        # Store both the intersection point and its position in the original line
                        intersection_points.append((point, index))
            
            # Sort intersection points by their position in the original line
            intersection_points.sort(key=lambda x: x[1])
            
            # If we have too many intersections, the line might be too close to the boundary and cause numerical issues, so we skip it
            intersection_threshold = 10
            if len(intersection_points) >= intersection_threshold:
                print(f"WARNING: Line {i} is removed due to close proximity to the border with {len(intersection_points)} intersections")
                continue
                
            # Process line segments between intersections
            if len(intersection_points) >= 2 and len(intersection_points) < intersection_threshold:
                # Extract just the points from the sorted list
                points = [p[0] for p in intersection_points]
                
                # Determine which segments are inside
                segments = []
                for j in range(0, len(points) - 1, 2):
                    if j + 1 < len(points):  # Ensure we have a pair of points
                        idx1 = intersection_points[j][1]
                        idx2 = intersection_points[j+1][1]
                        
                        # Create segment: start intersection, points in between, end intersection
                        if idx2 > idx1 + 1:
                            segment = np.vstack([
                                [points[j]],
                                line[idx1+1:idx2+1],
                                [points[j+1]]
                            ])
                            segments.append(segment)
                
                # Add all valid segments to the result
                new_lines.extend(segments)
                    
        return new_lines

    @staticmethod
    def exert_boundary(lines_1, lines_2, border_nodes):
        """
        Definiing boundary and trimming lines at the boundary
        """
        # boundary definition
        limits = np.array([np.min(border_nodes, axis=0), np.max(border_nodes, axis=0)])
        boundary_lines = np.array([
            np.array([[limits[0,0],limits[0,1]], [limits[1,0],limits[0,1]]]),
            np.array([[limits[1,0],limits[0,1]], [limits[1,0],limits[1,1]]]),
            np.array([[limits[1,0],limits[1,1]], [limits[0,0],limits[1,1]]]),
            np.array([[limits[0,0],limits[1,1]], [limits[0,0],limits[0,1]]])
            ], dtype=object)
        
        new_boundary_lines = np.empty((boundary_lines.shape[0], boundary_lines.shape[1]+1, 2), dtype=object)
        for i, edge in enumerate(boundary_lines):
            new_edge = np.empty((edge.shape[0]+1,2))
            for j in range(edge.shape[0]):
                if j==0: new_edge[j,:]=edge[j]
                else: 
                    new_edge[2*j-1,:] = np.sum(edge[j-1:j+1],axis=0)/2
                    new_edge[2*j,:] = edge[j]
            new_boundary_lines[i,:] = new_edge
        boundary_lines = new_boundary_lines

        # trimming at the boundary
        lines_1 = Lattice.trimming_line(lines_1, boundary_lines, limits)
        lines_2 = Lattice.trimming_line(lines_2, boundary_lines, limits)

        # boundary
        edge_nodes = np.array([line[index,:] for lines in [lines_1, lines_2] for line in lines for index in [0,-1]])
        boundary_corners = np.array([
            [limits[0,0],limits[0,1]], 
            [limits[1,0],limits[0,1]], 
            [limits[1,0],limits[1,1]], 
            [limits[0,0],limits[1,1]] ])
        border_nodes = np.concatenate((boundary_corners, edge_nodes))
        # edges
        tolerance=0.0005
        b = border_nodes[np.abs(border_nodes[:,1]-limits[0,1])<tolerance,:]
        t = border_nodes[np.abs(border_nodes[:,1]-limits[1,1])<tolerance,:]
        l = border_nodes[np.abs(border_nodes[:,0]-limits[0,0])<tolerance,:]
        r = border_nodes[np.abs(border_nodes[:,0]-limits[1,0])<tolerance,:]

        # sorting
        ribs = np.array((
            np.array((b[np.argsort(b[:,0], axis=0, kind='stable'),:])),
            np.array((t[np.argsort(t[:,0], axis=0, kind='stable'),:]))
        ), dtype=object)
        # trailing edge
        trailing_edge = np.empty(2, dtype=object)
        trailing_edge[0] = np.array((l[np.argsort(l[:,1], axis=0, kind='stable'),:]))
        trailing_edge[1] = np.array((r[np.argsort(r[:,1], axis=0, kind='stable'),:]))

        return lines_1, lines_2, ribs, trailing_edge

    @Utils.logger
    def trace_streamlines(self, field_type="SE", clearance=0.2, starting_points=None, bool_plot = False):

        # boundary for tracing
        boundary = np.stack((
            np.min(self.border_nodes_2D, axis=0) - clearance,
            np.max(self.border_nodes_2D, axis=0) + clearance
        ),axis=0)

        # starting point for tracing - centre of the grid
        if starting_points is None:
            start_point = np.stack((
                self.centroids_grid_2D.reshape(-1, 2).min(axis=0),
                self.centroids_grid_2D.reshape(-1, 2).max(axis=0)
            ), axis=0).mean(axis=0, keepdims=True).astype(np.float64)

        self.trace_lines[field_type] = {}
        for key in self.f_interp.keys():
            
            # tracing panel ribs
            seeds_1 = self.runga_kutta(
                self.f_interp[key][f'vecs_{field_type}'][1],
                start_point,
                boundary,
                self.cell_dimensions[0] 
            )
            traced_lines_1 = self.runga_kutta(
                self.f_interp[key][f'vecs_{field_type}'][0],
                seeds_1[0],
                boundary,
                self.cell_dimensions[1]
            )

            # tracing panel morphing direction
            seeds_2 = self.runga_kutta(
                self.f_interp[key][f'vecs_{field_type}'][0],
                start_point,
                boundary,
                self.cell_dimensions[1] 
            )
            traced_lines_2 = self.runga_kutta(
                self.f_interp[key][f'vecs_{field_type}'][1],
                seeds_2[0],
                boundary,
                self.cell_dimensions[0]
            )

            # resample at intersection point
            lines_1, lines_2 = self.resample_lines(traced_lines_1, traced_lines_2)
            self.trace_lines[field_type][key] = {
                            "lines_1": lines_1,
                            "lines_2": lines_2
                        }

            if bool_plot:
                Utils.Plots.grid_split(
                    {
                        "coords": self.centroids_grid_2D,
                        "border": self.border_nodes_2D,
                        "lines": self.trace_lines[field_type][key]
                    },
                    os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_trace_2D_{field_type}_{key}.svg"),
                    show=False
                )

    def create_lattice(self,field_type="SE", bool_plot = False):

        lines_set = self.trace_lines[field_type]

        for key, lines in lines_set.items():
            lines_1 = lines["lines_1"]
            lines_2 = lines["lines_2"]

            # introducing chevrons - NOTE: check the line being used for chevrons
            Tan60 = np.tan(self.chevron_angle) # unit cell chevron angle
            for line_i, line in enumerate(lines_2):
                chevron_line = []
                # Check if line has enough points   
                if line.shape[0]<2: continue
                # Adding chevron points
                for seg_i in range(line.shape[0]-1):
                    if seg_i==0: 
                        chevron_line.append(line[seg_i]) # additing initial point
                    v1 = np.diff(line[seg_i:seg_i+2], axis=0).ravel()
                    mag_v1 = np.linalg.norm(v1) #+ np.finfo(float).eps
                    norm_v1 = v1/mag_v1 # segment direction
                    norm_v2 = np.array([-norm_v1[1], norm_v1[0]]) # pendicular direction
                    mag_v2 = Tan60 * (mag_v1/2) 
                    point = line[seg_i] + v1/2 + (mag_v2*norm_v2) #*(-1)**line_i
                    chevron_line.extend([point, line[seg_i+1]])
                # adding chevron line to the spanwise
                chevron_line = Utils.GeoOps.bisect_line(np.array(chevron_line))
                lines_2[line_i] = chevron_line

            # exerting boundary
            chordwise, spanwise, ribs, trailing_edge = Lattice.exert_boundary(lines_1, lines_2, self.border_nodes_2D)

            # merging the two trailing egdes
            trailing_edge_y = np.sort(np.concatenate(tuple(edge[:,1].astype(np.float32) for edge in trailing_edge), axis=0, dtype=np.float32))
            for i in range(trailing_edge.shape[0]):
                # merging the trailing edge
                trailing_edge[i] = np.column_stack((
                    np.ones(trailing_edge_y.shape)*trailing_edge[i][0,0],
                    trailing_edge_y
                ))
    

            # data collection
            self.lattice_lines[key] = {
                "Chevrons": spanwise,
                "Stringers": chordwise,
                "Ribs": ribs,
                "TE_top": trailing_edge[0][np.newaxis, ...],
                "TE_bottom": trailing_edge[1][np.newaxis, ...],
            }

            if bool_plot:
                Utils.Plots.grid_split(
                    {
                        "coords": self.centroids_grid_2D,
                        # "border": self.border_nodes_2D,
                        "lines": self.lattice_lines[key]
                    },
                    os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_lattice_2D_{field_type}_{key}.svg"),
                    show=False
                )

        # Save
        Utils.ReadWriteOps.save_object(
            self.lattice_lines,
            os.path.join(self.directory.case_folder, "data", f"{self.case_number}_lattice_lines"),
            method="pickle"
        )

    @Utils.logger
    def analysis(self):
        self.create_element_grid()
        self.create_node_grid()
        self.flatten_grid()
        self.create_field_vectors(bool_plot=True)
        self.trace_streamlines("SE", bool_plot=True)
        # self.trace_streamlines("SK")
        self.create_lattice(bool_plot=True)

class Mesh:
    def __init__(self, dim: int = 3) -> None:

        if dim not in [2, 3]:
            raise ValueError("dim must be either 2 or 3")

        self.dim = dim
        self.nodes = np.empty((0, dim), dtype=float)
        self.beams = np.empty((0, 2), dtype=int)
        self.triangles = np.empty((0, 3), dtype=int)
        self.nsets = {}
        self.beam_sets = {}
        self.triangle_sets = {}
        self.groups = {}
        self._intersection_nodes_indices = np.empty((0), dtype=int)  # Indices of nodes that are intersection points

    def add_or_merge_nodes(self, new_nodes: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
        """
        Adds multiple nodes to the nodes array, merging close nodes within a given tolerance.

        Parameters:
            new_nodes (np.ndarray): A 2D array of shape (n, dim) representing the nodes to be added.
            tolerance (float): Distance threshold for merging nodes.

        Returns:
            np.ndarray: Indices in self.nodes where each new node was incorporated or merged.
        """
        if not isinstance(new_nodes, np.ndarray) or new_nodes.ndim != 2 or new_nodes.shape[1] != self.dim:
            raise ValueError(f"new_nodes must be a 2D numpy array with shape (n, {self.dim})")

        incorporated_indices = []
        for node in new_nodes:
            if self.nodes.shape[0] == 0:
                self.nodes = np.vstack([self.nodes, node])
                incorporated_indices.append(0)
                continue

            # Efficient distance calculation using broadcasting
            dists = (np.sum((self.nodes - node) ** 2, axis=1))**0.5 # Euclidean distance
            close_idxs = np.where(dists < tolerance)[0]

            if close_idxs.size == 0:
                self.nodes = np.vstack([self.nodes, node])
                incorporated_indices.append(len(self.nodes) - 1)
            else:
                # Merge all close nodes and the new node
                idx = close_idxs[0]
                avg = np.mean(np.vstack([self.nodes[close_idxs], node]), axis=0)
                self.nodes[idx] = avg
                incorporated_indices.append(idx)

                # Remove other merged nodes (except the first)
                delete_node_indices = close_idxs[1:]
                if delete_node_indices.size > 0:
                    self.replace_nodes(delete_node_indices, idx)
                    # Adjust previously returned indices if necessary
                    for i, inc_idx in enumerate(incorporated_indices[:-1]):
                        decrement = np.sum(delete_node_indices < inc_idx)
                        incorporated_indices[i] -= decrement

                # Warning
                if self.dim==3:
                    print(f"WARNING: Merged {close_idxs.size} duplicate node(s) into node index {idx}.")

        return np.array(incorporated_indices, dtype=int)

    def replace_nodes(self, remove_node_indices: np.ndarray, replace_index: int) -> None:
        """
        Removes merged nodes from self.nodes and updates beams and triangles accordingly.
        """
        self.nodes = np.delete(self.nodes, remove_node_indices, axis=0)
        # Update indices in beams and triangles efficiently
        if self.beams.shape[0] > 0:
            # Replace deleted indices with replace_index
            for del_idx in sorted(remove_node_indices, reverse=True):
                self.beams[self.beams == del_idx] = replace_index
            # Decrement indices greater than deleted
            for del_idx in sorted(remove_node_indices, reverse=True):
                self.beams[self.beams > del_idx] -= 1
            # Remove degenerate beams (where both nodes are the same)
            degenerate_mask = self.beams[:, 0] == self.beams[:, 1]
            num_deleted_beams = np.count_nonzero(degenerate_mask)
            if num_deleted_beams > 0:
                print(f"WARNING: Deleted {num_deleted_beams} degenerate beam(s) after node merge.")
            self.beams = self.beams[~degenerate_mask]
        if self.triangles.shape[0] > 0:
            # Replace deleted indices with replace_index
            for del_idx in sorted(remove_node_indices, reverse=True):
                self.triangles[self.triangles == del_idx] = replace_index
            # Decrement indices greater than deleted
            for del_idx in sorted(remove_node_indices, reverse=True):
                self.triangles[self.triangles > del_idx] -= 1
            # Remove degenerate triangles (where two or more nodes are the same)
            degenerate_mask = (
                (self.triangles[:, 0] == self.triangles[:, 1]) |
                (self.triangles[:, 1] == self.triangles[:, 2]) |
                (self.triangles[:, 0] == self.triangles[:, 2])
            )
            num_deleted_triangles = np.count_nonzero(degenerate_mask)
            if num_deleted_triangles > 0:
                print(f"WARNING: Deleted {num_deleted_triangles} degenerate triangle(s) after node merge.")
            self.triangles = self.triangles[~degenerate_mask]
        # Update indices in groups as well
        for group_key, group_lines in self.groups.items():
            for line_idx, line in enumerate(group_lines):
                # Replace deleted indices with replace_index
                for del_idx in sorted(remove_node_indices, reverse=True):
                    line[line == del_idx] = replace_index
                # Decrement indices greater than deleted
                for del_idx in sorted(remove_node_indices, reverse=True):
                    line[line > del_idx] -= 1
                # Remove degenerate lines (length < 2)
                if line.size < 2 or np.all(line == line[0]):
                    print(f"WARNING: Deleted degenerate line in group '{group_key}' after node merge.")
                    group_lines[line_idx] = np.array([], dtype=int)
            # Remove empty lines from group
            self.groups[group_key] = np.array([l for l in group_lines if l.size > 1], dtype=object)

        # Update indices in _intersection_nodes_indices as well
        for del_idx in sorted(remove_node_indices, reverse=True):
            self._intersection_nodes_indices[self._intersection_nodes_indices == del_idx] = replace_index
        for del_idx in sorted(remove_node_indices, reverse=True):
            self._intersection_nodes_indices[self._intersection_nodes_indices > del_idx] -= 1
        # Remove duplicates and sort
        self._intersection_nodes_indices = np.unique(self._intersection_nodes_indices)

    def create_grouped_lines_nodes(self, key,  new_lines: np.ndarray, tolerance: float = 1e-4):
        """
        Adds multiple lines to the lattice.

        Parameters:
            new_lines (np.ndarray): A 1D array of of objects where each object is a 2D array in form (n, 2) representing the nodes of the line to be added.
        """

        new_indices = []
        for line in new_lines:
            if not isinstance(line, np.ndarray) or line.ndim != 2 or line.shape[0] < 2 or line.shape[1] != 2:
                raise ValueError("Each line must be a 2D numpy array with shape (n, 2) where n >= 2")

            # Use the Mesh class to add nodes
            node_indices = self.add_or_merge_nodes(line, tolerance)

            # Add the new line's node indices to the collection as an object array
            # Preserve original order while removing duplicates
            _, unique_indices = np.unique(node_indices, return_index=True)
            node_indices = node_indices[np.sort(unique_indices)]
            new_indices.append(node_indices)
            # Create beams for each line and add to the mesh
            if node_indices.size > 1:
                beam_pairs = np.column_stack((node_indices[:-1], node_indices[1:]))
                beam_indices = self.add_beams(beam_pairs)
                # Add to beam_sets for this group key
                if key not in self.beam_sets:
                    self.beam_sets[key] = np.empty((0,), dtype=int)
                self.beam_sets[key] = np.concatenate([self.beam_sets[key], beam_indices])

        # Store the new indices in the groups dictionary
        self.groups[key] = np.array(new_indices, dtype=object)

    def add_beams(self, new_beams: np.ndarray) -> np.ndarray:
        """
        Adds multiple beams to the beams array and returns indices in self.beams corresponding to new_beams.

        Parameters:
            new_beams (np.ndarray): A 2D array of shape (n, 2) representing the beams to be added.

        Returns:
            np.ndarray: Indices in self.beams where each new beam was incorporated.
        """
        if not isinstance(new_beams, np.ndarray) or new_beams.ndim != 2 or new_beams.shape[1] != 2:
            raise ValueError("new_beams must be a 2D numpy array with shape (n, 2)")

        # Normalize all beams by sorting node indices
        incoming = np.sort(new_beams, axis=1)
        if self.beams.shape[0] > 0:
            existing = np.sort(self.beams, axis=1)
            # Use structured arrays for fast row-wise comparison
            dtype = [('n0', incoming.dtype), ('n1', incoming.dtype)]
            existing_view = existing.view(dtype)
            incoming_view = incoming.view(dtype)
            # Find unique incoming beams not already present
            # np.in1d is deprecated, use np.isin instead
            is_unique = ~np.isin(incoming_view, existing_view).squeeze()
            num_duplicates = np.count_nonzero(~is_unique)
            if num_duplicates > 0:
                print(f"WARNING: Discarded {num_duplicates} duplicate beam(s).")
            added_beams = new_beams[is_unique]
            self.beams = np.vstack([self.beams, added_beams])
            # Build a lookup for all beams (existing + new) for fast index retrieval
            all_beams_sorted = np.sort(self.beams, axis=1)
            all_beams_view = all_beams_sorted.view(dtype)
            # For each new_beam, find its index in self.beams
            result_indices = np.empty(new_beams.shape[0], dtype=int)
            for i, beam in enumerate(incoming_view):
                idx = np.where(all_beams_view == beam)[0]
                result_indices[i] = idx[0] if idx.size > 0 else -1
            return result_indices
        else:
            self.beams = new_beams.copy()
            return np.arange(self.beams.shape[0])

    def add_triangles(self, new_triangles: np.ndarray) -> np.ndarray:
        """
        Adds multiple triangles to the triangles array and returns indices in self.triangles corresponding to new_triangles.

        Parameters:
            new_triangles (np.ndarray): A 2D array of shape (n, 3) representing the triangles to be added.

        Returns:
            np.ndarray: Indices in self.triangles where each new triangle was incorporated.
        """
        if not isinstance(new_triangles, np.ndarray) or new_triangles.ndim != 2 or new_triangles.shape[1] != 3:
            raise ValueError("new_triangles must be a 2D numpy array with shape (n, 3)")

        # Normalize all triangles by sorting node indices
        incoming = np.sort(new_triangles, axis=1)
        if self.triangles.shape[0] > 0:
            existing = np.sort(self.triangles, axis=1)
            # Use structured array for fast comparison
            dtype = [('n0', new_triangles.dtype), ('n1', new_triangles.dtype), ('n2', new_triangles.dtype)]
            existing_view = existing.view(dtype)
            incoming_view = incoming.view(dtype)
            # Find unique incoming triangles not already present
            is_unique = ~np.isin(incoming_view, existing_view).squeeze()
            num_duplicates = np.count_nonzero(~is_unique)
            if num_duplicates > 0:
                print(f"WARNING: Discarded {num_duplicates} duplicate triangle(s).")
            added_triangles = new_triangles[is_unique]
            self.triangles = np.vstack([self.triangles, added_triangles])
            # Build a lookup for all triangles (existing + new) for fast index retrieval
            all_triangles_sorted = np.sort(self.triangles, axis=1)
            all_triangles_view = all_triangles_sorted.view(dtype)
            # For each new_triangle, find its index in self.triangles
            result_indices = np.empty(new_triangles.shape[0], dtype=int)
            for i, tri in enumerate(incoming_view):
                idx = np.where(all_triangles_view == tri)[0]
                result_indices[i] = idx[0] if idx.size > 0 else -1
            return result_indices
        else:
            self.triangles = new_triangles.copy()
            return np.arange(self.triangles.shape[0])

    def remove_beams(self, beam_indices: np.ndarray) -> None:
        """
        Removes beams at the specified indices and updates beam_sets accordingly.

        Parameters:
            beam_indices (np.ndarray): Array of indices of beams to remove.
        """
        if not isinstance(beam_indices, np.ndarray):
            beam_indices = np.array(beam_indices, dtype=int)
        beam_indices = np.unique(beam_indices)
        if beam_indices.size == 0:
            return

        # Remove beams from self.beams
        self.beams = np.delete(self.beams, beam_indices, axis=0)

        # Update beam_sets
        for key, indices in self.beam_sets.items():
            # Remove indices that were deleted
            mask = ~np.isin(indices, beam_indices)
            updated_indices = indices[mask]
            # Decrement indices greater than deleted
            for idx in sorted(beam_indices):
                updated_indices[updated_indices > idx] -= 1
            self.beam_sets[key] = updated_indices

    def triangulate_2D(
        self,
        node_indexes: np.ndarray,
        positions: np.ndarray = np.array([0, 1], dtype=int),
    ) -> None:
        """
        Performs triangulation on the current nodes and updates the triangles array.
        """
        if not isinstance(node_indexes, np.ndarray) or node_indexes.ndim != 1 or node_indexes.shape[0] < 3:
            raise ValueError(f"node_indexes must be a 1D numpy array with at least 3 elements. Got shape {node_indexes.shape}.")

        if node_indexes.shape[0] >= 3:
            tri = Delaunay(self.nodes[np.ix_(node_indexes, positions)])  # Triangulate using specified positions
            triangles = np.asarray([node_indexes[sub_indexes] for sub_indexes in tri.simplices], dtype=int)
            self.add_triangles(triangles)
        else:
            raise ValueError("Not enough nodes for triangulation.")

    def check_mesh_integrity(self):
        """
        Checks for unused nodes, degenerate beams, and degenerate triangles in the mesh.
        Prints warnings for any issues found.
        Returns:
            dict: Summary of unused nodes, degenerate beams, and degenerate triangles.
        """
        unused_nodes = []
        degenerate_beams = []
        degenerate_triangles = []

        node_count = self.nodes.shape[0]
        used_in_beams = set(self.beams.flatten()) if self.beams.size > 0 else set()
        used_in_triangles = set(self.triangles.flatten()) if self.triangles.size > 0 else set()
        used_nodes = used_in_beams.union(used_in_triangles)
        unused_nodes = [i for i in range(node_count) if i not in used_nodes]

        if len(unused_nodes) > 0:
            print(f"WARNING: {len(unused_nodes)} unused node(s) found: {unused_nodes}")

        if self.beams.size > 0:
            for idx, beam in enumerate(self.beams):
                if beam[0] == beam[1]:
                    degenerate_beams.append(idx)
            if len(degenerate_beams) > 0:
                print(f"WARNING: {len(degenerate_beams)} degenerate beam(s) found at indices: {degenerate_beams}")

        if self.triangles.size > 0:
            for idx, tri in enumerate(self.triangles):
                if len(set(tri)) < 3:
                    degenerate_triangles.append(idx)
            if len(degenerate_triangles) > 0:
                print(f"WARNING: {len(degenerate_triangles)} degenerate triangle(s) found at indices: {degenerate_triangles}")

        # Check if all beams exist as an edge in the triangles
        beam_edges = np.sort(self.beams, axis=1)
        triangle_edges = np.sort(np.concatenate([
            self.triangles[:, [0, 1]],
            self.triangles[:, [1, 2]],
            self.triangles[:, [2, 0]]
        ], 0), axis=1)
        # Find beams that are not present in triangle edges
        missing_beam_edges = self.beams[~np.isin(beam_edges, triangle_edges).all(axis=1)]
        if missing_beam_edges.shape[0] > 0:
            print(f"WARNING: {len(missing_beam_edges)} beam(s) do not exist as an edge in any triangle: {missing_beam_edges}")

        return {
            "unused_nodes": np.asarray(unused_nodes, dtype=int),
            "degenerate_beams": np.asarray(degenerate_beams, dtype=int),
            "degenerate_triangles": np.asarray(degenerate_triangles, dtype=int),
            "missing_beam_edges_in_triangles": np.asarray(missing_beam_edges, dtype=int)
        }

    def plot_2D(self, keys: dict = {}, label: bool = False, save_path=None, show=False):
        """
        Plots the 2D mesh with options to plot groups, nodes, beams and triangles.
        """
        def groups(group_sets):
            """
            Plots 2D groups.    
            """
            # Determine which groups to plot
            keys = group_sets if len(group_sets) > 0 else list(self.groups.keys())
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            labelled_node = []
            for i, ax in enumerate(axes):
                for row, group_key in enumerate(keys):
                    group_lines = self.groups[group_key]
                    for idx, line_indices in enumerate(group_lines):
                        if line_indices.shape[0] > 0:
                            line_indices = np.array(line_indices, dtype=int)
                            line_coords = self.nodes[line_indices]
                            ax.plot(line_coords[:, 0], line_coords[:, 1], color=colours[row], marker='.', markersize=0.15,
                                    linestyle='-', linewidth=0.5, label=group_key if idx == 0 else None)
                            if label:
                                # LINE LABEL
                                # Find the node inside the boundary and closest to the middle
                                boundary_x_min, boundary_x_max = x_top_lim if i == 0 else x_bot_lim
                                boundary_y_min, boundary_y_max = y_lim
                                # Mask for nodes inside the boundary
                                inside_mask = (
                                    (line_coords[:, 0] >= boundary_x_min) & (line_coords[:, 0] <= boundary_x_max) &
                                    (line_coords[:, 1] >= boundary_y_min) & (line_coords[:, 1] <= boundary_y_max)
                                )
                                if np.any(inside_mask):
                                    # Find the node closest to the mean of the inside nodes
                                    inside_coords = line_coords[inside_mask]
                                    center = np.mean(inside_coords, axis=0)
                                    squared_dists = np.sum((inside_coords - center)**2, axis=1)
                                    closest_idx = np.argmin(squared_dists)
                                    label_x, label_y = inside_coords[closest_idx]
                                    ax.text(label_x, label_y, f"l:{idx}", color=colours[row],
                                            ha='center', va='top', fontsize=3)
                                # NODE LABEL
                                # Label each node in the line with its index in the line, but only once per node
                                for node_idx, (x, y) in enumerate(line_coords[inside_mask]):
                                    global_node_idx = line_indices[inside_mask][node_idx]
                                    if global_node_idx not in labelled_node:
                                        labelled_node.append(global_node_idx)
                                        ax.text(x, y, f"{global_node_idx}", color='black', ha='center', va='bottom', fontsize=2)

        def nodes(node_sets):
            """
            2D scatter plot of nodes.
            """
            # Determine which nodes to plot
            if node_sets and all(k in self.nsets for k in node_sets):
                keys = node_sets
                nodes_by_key = {k: self.nodes[self.nsets[k]] for k in keys}
            else:
                keys = ["Nodes"]
                nodes_by_key = {"Nodes": self.nodes}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            for i, ax in enumerate(axes):
                for row, key in enumerate(keys):
                    nodes = nodes_by_key[key]
                    ax.scatter(
                        nodes[:, 0], nodes[:, 1],
                        color=colours[row], s=20, marker='.',
                        label=key, alpha=0.6
                    )
                    if label:
                        # LABEL NODES (inside boundary)
                        boundary_x_min, boundary_x_max = x_top_lim if i == 0 else x_bot_lim
                        boundary_y_min, boundary_y_max = y_lim
                        inside_mask = (
                            (nodes[:, 0] >= boundary_x_min) & (nodes[:, 0] <= boundary_x_max) &
                            (nodes[:, 1] >= boundary_y_min) & (nodes[:, 1] <= boundary_y_max)
                        )
                        inside_nodes = nodes[inside_mask]
                        for idx, (x, y) in enumerate(inside_nodes):
                            ax.text(x, y, f"n:{idx}", color=colours[row],
                                    ha='center', va='bottom', fontsize=2)

        def beams(beam_sets):
            """
            2D line plot of beams.
            """

            # Determine which beams to plot
            if beam_sets and all(k in self.beam_sets for k in beam_sets):
                keys = beam_sets
                beams_by_key = {k: self.beams[self.beam_sets[k]] for k in keys}
            else:
                keys = ["Beams"]
                beams_by_key = {"Beams": self.beams}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            for i, ax in enumerate(axes):
                for row, key in enumerate(keys):
                    beams = beams_by_key[key]
                    for idx, beam in enumerate(beams):
                        beam = np.array(beam, dtype=int)
                        beam_coords = self.nodes[beam]
                        ax.plot(
                            beam_coords[:, 0], beam_coords[:, 1],
                            color=colours[row], marker='.', markersize=0.5,
                            linestyle='-', linewidth=0.5, label=key if idx == 0 else None
                        )
                        if label:
                            # LABEL BEAM (at midpoint inside boundary)
                            boundary_x_min, boundary_x_max = x_top_lim if i == 0 else x_bot_lim
                            boundary_y_min, boundary_y_max = y_lim
                            inside_mask = (
                                (beam_coords[:, 0] >= boundary_x_min) & (beam_coords[:, 0] <= boundary_x_max) &
                                (beam_coords[:, 1] >= boundary_y_min) & (beam_coords[:, 1] <= boundary_y_max)
                            )
                            if np.any(inside_mask):
                                inside_coords = beam_coords[inside_mask]
                                center = np.mean(inside_coords, axis=0)
                                label_x, label_y = center
                                ax.text(label_x, label_y, f"b:{idx}", color=colours[row],
                                        ha='center', va='top', fontsize=2)

        def triangles(triangle_sets):
            """
            2D patch plot of triangles.
            """
            # Determine which triangles to plot
            if triangle_sets and all(k in self.triangle_sets for k in triangle_sets):
                keys = triangle_sets
                triangles_by_key = {k: self.triangles[self.triangle_sets[k]] for k in keys}
            else:
                keys = ["Triangles"]
                triangles_by_key = {"Triangles": self.triangles}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            for i, ax in enumerate(axes):
                for row, key in enumerate(keys):
                    triangles = triangles_by_key[key]
                    for idx, triangle in enumerate(triangles):
                        pts = self.nodes[triangle][:, :2]
                        polygon = patches.Polygon(
                            pts,
                            closed=True,
                            facecolor=colours[row],
                            alpha=0.2,
                            edgecolor='k',
                            linewidth=0.5,
                            label=key if idx == 0 else None
                        )
                        ax.add_patch(polygon)
                        if label:
                            # LABEL TRIANGLE (at centroid if inside boundary)
                            boundary_x_min, boundary_x_max = x_top_lim if i == 0 else x_bot_lim
                            boundary_y_min, boundary_y_max = y_lim
                            centroid = np.mean(pts, axis=0)
                            if (
                                (centroid[0] >= boundary_x_min) and (centroid[0] <= boundary_x_max) and
                                (centroid[1] >= boundary_y_min) and (centroid[1] <= boundary_y_max)
                            ):
                                ax.text(centroid[0], centroid[1], f"t:{idx}", color=colours[row],
                                        ha='center', va='center', fontsize=2)

        fig, axes = plt.subplots(2)

        try:
            group_sets = keys["group_sets"]
            groups(group_sets)
        except KeyError:
            pass

        try:
            node_sets = keys["node_sets"]
            nodes(node_sets)
        except KeyError:
            pass

        try:
            beam_sets = keys["beam_sets"]
            beams(beam_sets)
        except KeyError:
            pass

        try:
            triangle_sets = keys["triangle_sets"]
            triangles(triangle_sets)
        except KeyError:
            pass

        for i, ax in enumerate(axes):
            ax.set_xlabel('$\\eta$-Axis', fontsize=9)
            ax.set_ylabel('$\\zeta$-Axis', fontsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_aspect('equal')

        # Determine plot limits with padding
        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        diff = (maxs - mins).min()
        padding = 0.1 * diff  # 10% padding on each side
        mins -= padding
        maxs += padding
        x_top_lim = [0.0, maxs[0]]
        x_bot_lim = [mins[0], 0.0]
        y_lim = [mins[1], maxs[1]]

        # Set titles and limits
        axes[0].set_title('Top Surface', loc='left', fontsize=9)
        axes[1].set_title('Bottom Surface', loc='left', fontsize=9)
        axes[0].set_xlim(x_top_lim)
        axes[1].set_xlim(x_bot_lim)
        axes[0].set_ylim(y_lim)
        axes[1].set_ylim(y_lim)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.95), ncol=len(labels), frameon=False)

        fig.tight_layout()

         # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_3D(self, keys: dict = {}, label: bool = False, save_path=None, show=False):
        """
        Plots the 3D nodes with options to plot nodes, beams and triangles.
        """
        def nodes(node_sets):
            """
            3D scatter plot of nodes.
            """
            # Determine which nodes to plot
            if node_sets and all(k in self.nsets for k in node_sets):
                keys = node_sets
                nodes_by_key = {k: self.nodes[self.nsets[k]] for k in keys}
            else:
                keys = ["Nodes"]
                nodes_by_key = {"Nodes": self.nodes}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            for row, key in enumerate(keys):
                nodes = nodes_by_key[key]
                ax.scatter(
                    nodes[:, 0], nodes[:, 1], nodes[:, 2],
                    color=colours[row], s=2, marker='.',
                    alpha=0.6, label=key if row == 0 else None
                )
                if label:
                    for idx, (x, y, z) in enumerate(nodes):
                        ax.text(x, y, z, f"n:{idx}", color=colours[row], fontsize=6)

        def beams(beam_sets):
            # Determine which beams to plot
            if beam_sets and all(k in self.beam_sets for k in beam_sets):
                keys = beam_sets
                beams_by_key = {k: self.beams[self.beam_sets[k]] for k in keys}
            else:
                keys = ["Beams"]
                beams_by_key = {"Beams": self.beams}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            for row, key in enumerate(keys):
                beams = beams_by_key[key]
                for idx, beam in enumerate(beams):
                    beam = np.array(beam, dtype=int)
                    beam_coords = self.nodes[beam]
                    ax.plot(
                        beam_coords[:, 0], beam_coords[:, 1], beam_coords[:, 2],
                        color=colours[row], marker='.', markersize=0.5,
                        linestyle='-', linewidth=0.5, label=key if idx == 0 else None
                    )
                    if label:
                        center = np.mean(beam_coords, axis=0)
                        ax.text(center[0], center[1], center[2], f"b:{idx}", color=colours[row], fontsize=6)            

        def triangles(triangle_sets):
            # Determine which triangles to plot
            if triangle_sets and all(k in self.triangle_sets for k in triangle_sets):
                keys = triangle_sets
                triangles_by_key = {k: self.triangles[self.triangle_sets[k]] for k in keys}
            else:
                keys = ["Triangles"]
                triangles_by_key = {"Triangles": self.triangles}
            num_groups = len(keys)
            cmap = plt.get_cmap('rainbow', num_groups)
            colours = [cmap(i) for i in range(num_groups)]

            points = []
            for row, key in enumerate(keys):
                triangles = triangles_by_key[key]
                for idx, triangle in enumerate(triangles):
                    pts = self.nodes[triangle]
                    ax.plot_trisurf(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        color=colours[row], alpha=0.3, edgecolor='k', linewidth=0.5, label=key if idx == 0 else None
                    )
                    points.extend(pts)
                    if label:
                        centroid = np.mean(pts, axis=0)
                        ax.text(centroid[0], centroid[1], centroid[2], f"t:{idx}", color=colours[row], fontsize=6)

        fig = plt.figure(figsize=(7.5,4))
        ax = fig.add_subplot(111, projection='3d')

        try:
            node_sets = keys["node_sets"]
            nodes(node_sets)
        except KeyError:
            pass

        try:
            beam_sets = keys["beam_sets"]
            beams(beam_sets)
        except KeyError:
            pass

        try:
            triangle_sets = keys["triangle_sets"]
            triangles(triangle_sets)
        except KeyError:
            pass

        # decorations
        ax.set_xlabel("X [m]", fontsize=9)
        ax.set_ylabel("Y [m]", fontsize=9)
        ax.set_zlabel("Z [m]", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.xaxis.labelpad = 25

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',fontsize=9, bbox_to_anchor=(1, 1), ncol=len(labels), frameon=False)

        ax.yaxis.set_ticks(np.arange(np.min(self.nodes[:,1]), np.max(self.nodes[:,1])+0.01, 0.2))
        ax.zaxis.set_ticks(np.arange(-0.1, 0.1 +0.01, 0.1))
        ax.set_box_aspect((np.ptp(self.nodes[:,0]),np.ptp(self.nodes[:,1]),np.ptp(self.nodes[:,2]),), zoom=1.1) 
        plt.subplots_adjust(top = 1, bottom = 0, right = 1.15, left = 0, hspace = 0, wspace = 0)

         # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def write_mesh(self, filename: str, FR_INPUTS: dict, UC_INPUTS: dict):
        """
        Writes the mesh to a file using the specified serialization method.

        Parameters:
            filename (str): The base file path (without extension) where the mesh will be saved.
        """
        def format_lines(data_list, items_per_line=16):
            lines : list[str] = []
            for i in range(0, len(data_list), items_per_line):
                chunk = data_list[i:i + items_per_line]
                lines.append(", ".join(map(str, chunk)) + ("," if i + items_per_line < len(data_list) else ""))
            return lines

        lines = []

        # nodes
        lines.append("*NODE")
        for i, node in enumerate(self.nodes):
            i = i + 1  # Abaqus uses 1-based indexing
            lines.append("%i, %f, %f, %f" % (i, *node))

        # beams
        lines.append("*ELEMENT, TYPE=%s" % ("B31"))
        for i, beam in enumerate(self.beams):
            i = i + 1  # Abaqus uses 1-based indexing
            beam = beam + 1  # Abaqus uses 1-based indexing
            lines.append("%i, %i, %i" % (i, beam[0], beam[1]))

        # triangles
        num_beam_elements = self.beams.shape[0]
        lines.append("*ELEMENT, TYPE=%s" % ("S3"))
        # outside
        for i, triangle in enumerate(self.triangles):
            i = i + 1 + num_beam_elements # Abaqus uses 1-based indexing
            triangle = triangle + 1  # Abaqus uses 1-based indexing
            lines.append("%i, %i, %i, %i" % (i, triangle[0], triangle[1], triangle[2]))
        # inside
        num_triangles = self.triangles.shape[0]
        for i, triangle in enumerate(self.triangles):
            i = i + 1 + num_beam_elements + num_triangles
            triangle = triangle + 1
            lines.append("%i, %i, %i, %i" % (i, triangle[0], triangle[1], triangle[2]))

        # node sets
        for set_name, set_items in self.nsets.items():
            lines.append("*NSET, NSET=%s" % (set_name))
            set_items = set_items + 1  # Abaqus uses 1-based indexing
            lines.extend(format_lines(set_items))
        # beam sets
        for set_name, set_items in self.beam_sets.items():
            set_items = set_items + 1  # Abaqus uses 1-based indexing
            lines.append("*ELSET, ELSET=%s" % (set_name))
            lines.extend(format_lines(set_items))
        # triangle sets
        # outside
        lines.append("*ELSET, ELSET=SHELL_OUTSIDE")
        lines.extend(format_lines(np.arange(num_triangles).astype(int) + 1 + num_beam_elements))
        # inside
        lines.append("*ELSET, ELSET=SHELL_INSIDE")
        lines.extend(format_lines(np.arange(num_triangles).astype(int) + 1 + num_beam_elements + num_triangles))
        # all shell elements
        lines.append("*ELSET, ELSET=SHELL")
        lines.extend(format_lines(np.arange(1 + num_beam_elements,num_triangles*2 + 1).astype(int)))

        # surfaces
        lines.append("*SURFACE, TYPE=ELEMENT, NAME=SHELL_OUTSIDE_SURF")
        lines.append("SHELL_OUTSIDE, SPOS")

        # dimensions
        dimensions = {
            "RIBS" : [UC_INPUTS['RIB_THICKNESS'], UC_INPUTS["CORE_THICKNESS"]],
            "TE" : UC_INPUTS["TE_DIMENSIONS"], # NOTE: Flat TE cross-section
            "STRINGERS" : [UC_INPUTS['RIB_THICKNESS'], UC_INPUTS["CORE_THICKNESS"]],
            "CHEVRONS" : [UC_INPUTS['CHEVRON_THICKNESS'], UC_INPUTS["CORE_THICKNESS"]],
        }
        # section
        for section in ["RIBS", "STRINGERS", "CHEVRONS"]: # Fairing
            lines.append("*BEAM SECTION, ELSET=%s, MATERIAL=MATERIAL_CORE, SECTION=RECT"%(section))
            lines.append("%f, %f"%tuple(dimensions[section]))
            lines.append("0, 1, 0")
        for section in ["TE"]: # Trailing Edge
            lines.append("*BEAM SECTION, ELSET=%s, MATERIAL=MATERIAL_FACESHEET, SECTION=RECT"%(section))
            lines.append("%f, %f"%tuple(dimensions[section]))
            lines.append("1, 0, 0")

        # facesheet section
        # outside
        offset = (UC_INPUTS["FACESHEET_THICKNESS"] + UC_INPUTS["CORE_THICKNESS"]) / (2.0 * UC_INPUTS["FACESHEET_THICKNESS"])
        lines.append("*SHELL SECTION, ELSET=SHELL_OUTSIDE, MATERIAL=MATERIAL_FACESHEET, OFFSET=%f"%(-offset))
        lines.append("%f, 5" % UC_INPUTS["FACESHEET_THICKNESS"])  # THICKNESS
        # inside
        lines.append("*SHELL SECTION, ELSET=SHELL_INSIDE, MATERIAL=MATERIAL_FACESHEET, OFFSET=%f"%(offset))
        lines.append("%f, 5" % UC_INPUTS["FACESHEET_THICKNESS"])  # THICKNESS

        # material properties
        for key in ["MATERIAL_CORE", "MATERIAL_FACESHEET"]:
            lines.append("*MATERIAL, NAME=%s" % key)
            lines.append("*ELASTIC")
            lines.append("%f, %f " % tuple(item for item in UC_INPUTS[key]))
            lines.extend(["*DENSITY", "1"])

        # rigid body ribs
        lines.append("*RIGID BODY, REF NODE=ANODE_1, TIE NSET=RIB_1")
        lines.append("*RIGID BODY, REF NODE=ANODE_2, TIE NSET=RIB_2")

        # Initial boundary conditions
        lines.append("*BOUNDARY")
        lines.append("ANODE_1, ENCASTRE") # FIXING ANCHOR NODE 1
        if FR_INPUTS['PRE_STRAIN_ACROSS_HINGE']!=0.0: # Assuming model is a symmetric half-model
            for dof in [1, 3, 5, 6]:
                lines.append("ANODE_2, %i, %i"%(dof, dof))
        else:
            lines.append("ANODE_2, 1, 3")
            lines.append("ANODE_2, 5, 6")

        # output requests
        Output_Requests = [
            "*OUTPUT, FIELD", # FIELD OUTPUT
            "*ELEMENT OUTPUT, ELSET=SHELL, DIRECTION=YES",
            "SE, SF", # SECTION STRAIN & FORCE FOR ELEMENT
            "*OUTPUT, HISTORY", # HISTORY OUTPUT
            "*NODE OUTPUT, NSET=ANODE_2", # ANCHOR NODE 2
            "RF, RM, U, UR", #  MOMENTS AND ROATIONS IN GLOBAL X AND Y
            "*OUTPUT, FIELD, VAR=PRESELECT",
            "*OUTPUT, HISTORY, VAR=PRESELECT",
        ]
        Output_Requested = False

        # Adding step: Pre-strain
        PreStrain = FR_INPUTS["PRE_STRAIN_ACROSS_HINGE"]
        Span = FR_INPUTS["SPAN"]
        Multiplier = 1/2 if FR_INPUTS['MODEL_SCALE'] == "HALF" else 1
        if FR_INPUTS['PRE_STRAIN_ACROSS_HINGE']!=0.0:
            if FR_INPUTS['SOLVER']=='NEWTON':
                lines.append("*STEP, NAME=PRESTRAIN, NLGEOM=YES, INC=500")
                lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
                lines.append("1, 1, 1E-9, 1")
                lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
            elif FR_INPUTS['SOLVER']=='LINEAR':
                lines.append("*STEP, NAME=PRESTRAIN, NLGEOM=NO, INC=500")
                lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
                lines.append("1, 1, 1E-9, 1")
                lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
            elif FR_INPUTS['SOLVER']=='RIKS':
                lines.append("*STEP, NAME=PRESTRAIN, NLGEOM=YES, INC=500")
                lines.append("*STATIC, RIKS") # RIKS SOLVER, IMPLICIT
                lines.append("1, , 1E-9, 1, , ANODE_2, 4, %f" % (FR_INPUTS['FOLDING_ANGLE_X']))
                # LOADING : FOLDING
                lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
            elif FR_INPUTS['SOLVER']=='DYNAMIC':
                lines.append("*AMPLITUDE, NAME=AMP-1, DEFINITION=SMOOTH STEP")
                lines.append("0, 0, 1, 1")
                lines.append("*STEP, NAME=PRESTRAIN, NLGEOM=YES, INC=500")
                lines.append("*DYNAMIC, APPLICATION=QUASI-STATIC")
                lines.append("1, 1, 1E-9, 1")
                lines.append("*BOUNDARY, AMPLITUDE=AMP-1") # DYNAMIC, IMPLICIT
            else:
                raise("Error: Unsupported solver type '%s'." % FR_INPUTS['SOLVER'])
            # LOADING VALUE
            lines.append("ANODE_2, 2, 2, %f"%(PreStrain*Span*Multiplier))
            # OUTPUT REQUESTS
            if not Output_Requested:
                lines.extend(Output_Requests)
                Output_Requested = True
            lines.append("*END STEP")

        # ADDING STEP : PRESSURE STEP
        if FR_INPUTS['SOLVER']=='NEWTON':
            lines.append("*STEP, NAME=PRESSURE, NLGEOM=YES, INC=500")
            lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
            lines.append("1, 1, 1E-9, 1")
            lines.append("*DSLOAD")
        elif FR_INPUTS['SOLVER']=='LINEAR':
            lines.append("*STEP, NAME=PRESSURE, NLGEOM=NO, INC=500")
            lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
            lines.append("1, 1, 1E-9, 1")
            lines.append("*DSLOAD")
        elif FR_INPUTS['SOLVER']=='RIKS':
            lines.append("*STEP, NAME=PRESSURE, NLGEOM=YES, INC=500")
            lines.append("*STATIC, RIKS") # RIKS SOLVER, IMPLICIT
            lines.append("1, , 1E-9, 1, , ANODE_2, 4, %f" % (FR_INPUTS['FOLDING_ANGLE_X']))
            # LOADING : FOLDING
            lines.append("*DSLOAD")
        elif FR_INPUTS['SOLVER']=='DYNAMIC':
            lines.append("*AMPLITUDE, NAME=AMP-1, DEFINITION=SMOOTH STEP")
            lines.append("0, 0, 1, 1")
            lines.append("*STEP, NAME=PRESSURE, NLGEOM=YES, INC=500")
            lines.append("*DYNAMIC, APPLICATION=QUASI-STATIC")
            lines.append("1, 1, 1E-9, 1")
            lines.append("*DSLOAD, AMPLITUDE=AMP-1") # DYNAMIC, IMPLICIT
        else:
            raise("Error: Unsupported solver type '%s'." % FR_INPUTS['SOLVER'])
        # LOADING VALUE
        lines.append("SHELL_OUTSIDE_SURF, P, %f" % (-11866.000000)) # FIXME: Pressure value
        # OUTPUT REQUESTS
        if not Output_Requested:
            lines.extend(Output_Requests)
            Output_Requested = True
        lines.append("*END STEP")

        # ADDING STEP : FOLDING STEP
        # Defining increments to ensure all sampling angle are found
        if len(FR_INPUTS['SAMPLING_ROTATIONS'])!=0:
            max_increment = np.gcd.reduce(np.asarray(FR_INPUTS['SAMPLING_ROTATIONS']))/(FR_INPUTS['FOLDING_ANGLE_HINGE']*Multiplier)
        else: max_increment = 0.1
        refiner = 0.01 if all([int(FILE_ID)==0,int(ITERATION_ID)==0]) else 0.1 # reducing stepsize in dynamic analysis
        # Defining Folding Step
        if FR_INPUTS['SOLVER']=='NEWTON':
            lines.append("*STEP, NAME=FOLDING, NLGEOM=YES, INC=1000")
            lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
            lines.append("%f, 1, 1E-9, %f"%(max_increment,max_increment))
            lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
        elif FR_INPUTS['SOLVER']=='LINEAR':
            lines.append("*STEP, NAME=FOLDING, NLGEOM=NO, INC=1000")
            lines.append("*STATIC") # STATIC SOLVER, IMPLICIT
            lines.append("%f, 1, 1E-9, %f"%(max_increment,max_increment))
            lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
        elif FR_INPUTS['SOLVER']=='RIKS':
            lines.append("*STEP, NAME=FOLDING, NLGEOM=YES, INC=1000")
            lines.append("*STATIC, RIKS") # RIKS SOLVER, IMPLICIT
            lines.append("%f, , 1E-9, %f, , ANODE_2, 4, %f" % (max_increment, max_increment, FR_INPUTS['FOLDING_ANGLE_X']))
            # LOADING : FOLDING
            lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
        elif FR_INPUTS['SOLVER']=='DYNAMIC':
            lines.append("*AMPLITUDE, NAME=AMP-2, DEFINITION=SMOOTH STEP")
            lines.append("0, 0, 1, 1")
            lines.append("*STEP, NAME=FOLDING, NLGEOM=YES, INC=1000")
            lines.append("*DYNAMIC, APPLICATION=QUASI-STATIC")
            lines.append(" %f, 1, 1E-9, %f"%(max_increment*refiner,max_increment*refiner))
            lines.append("*BOUNDARY, AMPLITUDE=AMP-2") # DYNAMIC, IMPLICIT
        else:
            raise("Error: Unsupported solver type '%s'." % FR_INPUTS['SOLVER'])
        # LOADING VALUE
        lines.append("ANODE_2, 4, 4, %f" %
            (FR_INPUTS['FOLDING_ANGLE_X']/2.0 if FR_INPUTS['MODEL_SCALE'] == "HALF" else FR_INPUTS['FOLDING_ANGLE_X'])
        )
        # OUTPUT REQUESTS
        if not Output_Requested:
            lines.extend(Output_Requests)
            Output_Requested = True
        lines.append("*END STEP")

        # writing
        with open(filename+".inp", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


class Tailored:

    def __init__(
        self,
        directory: Utils.Directory,
        case_number: int,
        lattice_data: Lattice | None = None,
        tolerance=1e-4,
    ):
        self.directory = directory
        self.case_number = case_number
        self.tolerance = tolerance
        self.lattice_data = lattice_data

    @Utils.logger
    def create_fairing_2D(self, bool_plot=False):

        # Load lattice data
        if self.lattice_data is None:
            self.lattice_lines = Utils.ReadWriteOps.load_object(
                os.path.join(self.directory.case_folder, "data", f"{self.case_number}_lattice_lines"),
                method="pickle"
            )
        else:
            self.lattice_lines = self.lattice_data.lattice_lines

        # Create 2D mesh for each increment
        self.mesh2D = {}
        for increment_key, grouped_lines in self.lattice_lines.items():
            self.mesh2D[increment_key] = Mesh(2)

            # Create nodes and groups from lines
            for group_name, group_lines in grouped_lines.items():
                self.mesh2D[increment_key].create_grouped_lines_nodes(group_name, group_lines, tolerance=self.tolerance)

            if bool_plot:
                self.mesh2D[increment_key].plot_2D(
                    {"group_sets": []},
                    False,
                    os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_tailored_2D_{increment_key}.svg")
                )

            # Triangulate
            self.mesh2D[increment_key].triangulate_2D(np.arange(self.mesh2D[increment_key].nodes.shape[0]))

            if bool_plot:
                self.mesh2D[increment_key].plot_2D(
                    {"group_sets": [], "triangle_sets": []},
                    False,
                    os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_tailored_2D_{increment_key}.svg")
                ) 

    def mapping_2D_to_3D(self, bool_plot=False):

        # Interpolation data
        self.node_coords_3D = self.lattice_data.surface_nodes_coords[self.lattice_data.node_index_grid].reshape((-1,3))
        self.node_coords_2D = self.lattice_data.nodes_grid_2D.reshape((-1,2))

        # Create RBF interpolator
        self.f_interp_2D_to_3D = interpolate.RBFInterpolator(self.node_coords_2D, self.node_coords_3D, neighbors=4, kernel='linear')

        # Map 2D nodes to 3D
        self.mesh3D = {}
        for increment_key in self.mesh2D.keys():
            self.mesh3D[increment_key] = Mesh(3)

            # Map nodes
            indices = self.mesh3D[increment_key].add_or_merge_nodes(
                self.f_interp_2D_to_3D(self.mesh2D[increment_key].nodes)
                                                                    )
            # check if same number of indices returned
            assert indices.shape[0] == self.mesh2D[increment_key].nodes.shape[0], "Error in node mapping."

            # Map 2D node indices to 3D node indices for beams and triangles
            self.mesh3D[increment_key].beams = self.mesh2D[increment_key].beams.copy()
            self.mesh3D[increment_key].triangles = self.mesh2D[increment_key].triangles.copy()

            # map over groups
            for group_key, group_lines in self.mesh2D[increment_key].groups.items():
                new_lines = []
                for line in group_lines:
                    mapped_line = indices[line.astype(int)]
                    new_lines.append(mapped_line)
                self.mesh3D[increment_key].groups[group_key] = np.array(new_lines, dtype=object)

            # using groups create sets for beams
            sorted_beams = np.sort(self.mesh3D[increment_key].beams, axis=1)
            for group_key, group_lines in self.mesh3D[increment_key].groups.items():
                beam_indices = []
                for line in group_lines:
                    if line.size > 1:
                        beam_pairs = np.column_stack((line[:-1], line[1:]))
                        sorted_pair = np.sort(beam_pairs, axis=1)
                        exist = np.isin(sorted_beams, sorted_pair).all(axis=1)
                        beam_indices.extend(np.where(exist)[0])
                        if not np.any(exist):
                            raise ValueError(f"Beam pairs does not exist in mesh.beams: {beam_pairs[exist]}")
                # Store the beam indices in the mesh's beam_sets
                self.mesh3D[increment_key].beam_sets[group_key] = np.array(beam_indices, dtype=int)

            if bool_plot:
                self.mesh3D[increment_key].plot_3D(
                    {"group_sets": [], "triangle_sets": []},
                    label=False,
                    save_path=os.path.join(self.directory.case_folder, "fig", f"{self.case_number}_tailored_3D_{increment_key}.svg"),
                )

    def validate_midplane_mesh(self):

        for increment_key in self.mesh3D.keys():
            # check mesh
            degenerates = self.mesh3D[increment_key].check_mesh_integrity()
            if degenerates['unused_nodes'].size > 0:
                print("Unused nodes:")
                print(self.mesh3D[increment_key].nodes[degenerates['unused_nodes']])
            if degenerates['degenerate_beams'].size > 0:
                self.mesh3D[increment_key].remove_beams(degenerates['degenerate_beams'])
            if degenerates['degenerate_triangles'].size > 0:
                print(self.mesh3D[increment_key].triangles[degenerates['degenerate_triangles']])
            if degenerates['missing_beam_edges_in_triangles'].size > 0:
                print(f"WARNING: {degenerates['missing_beam_edges_in_triangles'].size} beam edges missing in triangles:")
            degenerates = self.mesh3D[increment_key].check_mesh_integrity()
            if degenerates['unused_nodes'].size > 0 and degenerates['degenerate_beams'].size > 0 and degenerates['degenerate_triangles'].size > 0 and degenerates['missing_beam_edges_in_triangles'].size > 0:
                raise ValueError("Mesh integrity check failed, degenerates found.")

            # check repeated sets
            beam_sets = ["TE_top", "TE_bottom", "Ribs","Chevrons","Stringers"]
            for i, key_i in enumerate(beam_sets):
                for j, key_j in enumerate(beam_sets[i+1:]):
                    repeat_index = np.isin(self.mesh3D[increment_key].beam_sets[key_j], self.mesh3D[increment_key].beam_sets[key_i])
                    if np.any(repeat_index):
                        print(f"WARNING: Beams of '{key_i}' repeated in '{key_j}' are removed from '{key_j}'.")
                        self.mesh3D[increment_key].beam_sets[key_j] = self.mesh3D[increment_key].beam_sets[key_j][~repeat_index]

            



if __name__ == "__main__":
    # Example
    directory = Utils.Directory(case_name="test_case_4")
    case_number = 0

     #%%
    # Trace Lattice
    lattice_data = Lattice(directory, case_number)
    lattice_data.analysis()

    #%%
    # Tailored Geometry
    tailored = Tailored(directory, case_number, lattice_data=lattice_data)
    tailored.create_fairing_2D(bool_plot=False)
    tailored.mapping_2D_to_3D(bool_plot=False)
    tailored.validate_midplane_mesh()
