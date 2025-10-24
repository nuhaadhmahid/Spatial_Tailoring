# Tailoring.py
from numpy.char import capitalize
from numpy.strings import lower
import Utils
from Utils import FairingData
from Mesh import Mesh

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import interpolate

class Tracer:
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
        print(f"\tTraced field for folding angles (deg): {[f'{angle:.0f}' for angle in np.rad2deg(self.rotation_angles[self.indices])]}")

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

    def save_grid_data(self):
        Utils.ReadWriteOps.save_object(
            {
                "surface_nodes_coords": self.surface_nodes_coords,
                "node_index_grid": self.node_index_grid,
                "nodes_grid_2D": self.nodes_grid_2D,
                "centroids_grid_2D": self.centroids_grid_2D,
                "element_index_grid": self.element_index_grid
            },
            os.path.join(self.directory.case_folder, "data", f"{self.case_number}_grid_data"),
            "pickle"
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
            intersection_threshold = 6
            if len(intersection_points) > intersection_threshold:
                print(f"\tWARNING: Line {i} is removed due to close proximity to the border with {len(intersection_points)} intersections")
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
        lines_1 = Tracer.trimming_line(lines_1, boundary_lines, limits)
        lines_2 = Tracer.trimming_line(lines_2, boundary_lines, limits)

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
        trailing_edge_l = l[np.argsort(l[:,1], axis=0, kind='stable'),:]
        trailing_edge_r = r[np.argsort(r[:,1], axis=0, kind='stable'),:]
        y_mean_l = trailing_edge_l[:,0].mean()
        y_mean_r = trailing_edge_r[:,0].mean()
        trailing_edge_r[:,0] = y_mean_r
        trailing_edge_l[:,0] = y_mean_l
        trailing_edge[0] = np.array(trailing_edge_r)
        trailing_edge[1] = np.array(trailing_edge_l)

        return lines_1, lines_2, ribs, trailing_edge

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

        # Save trace lines
        Utils.ReadWriteOps.save_object(
            self.trace_lines,
            os.path.join(self.directory.case_folder, "data", f"{self.case_number}_trace_lines"),
            method="pickle"
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
                chevron_line = np.array(chevron_line)
                lines_2[line_i] = chevron_line

            # exerting boundary
            chordwise, spanwise, ribs, trailing_edge = Tracer.exert_boundary(lines_1, lines_2, self.border_nodes_2D)

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
                "CHEVRONS": spanwise,
                "STRINGERS": chordwise,
                "RIBS": ribs,
                "TE_TOP": trailing_edge[0][np.newaxis, ...],
                "TE_BOTTOM": trailing_edge[1][np.newaxis, ...],
            }

            if bool_plot:
                Utils.Plots.grid_split(
                    {
                        #"coords": self.centroids_grid_2D, # centroids
                        "coords": self.nodes_grid_2D, # nodes
                        # "border": self.border_nodes_2D,
                        "lines": {name.capitalize(): values for name, values in self.lattice_lines[key].items() if name!="TE_BOTTOM" and name!="TE_TOP"} | {"TRAILING_EDGE".capitalize():np.r_[*list(self.lattice_lines[key][name]for name in ["TE_TOP", "TE_BOTTOM"])]}
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
        print(f"Starting {self.__class__.__name__} analysis {self.directory.case_name} - {self.case_number}")

        # Generating grids
        self.create_element_grid()
        self.create_node_grid()
        self.flatten_grid()
        self.save_grid_data()

        # Creating field vectors and tracing streamlines
        self.create_field_vectors(bool_plot=True)
        self.trace_streamlines("SE", bool_plot=True)
        # self.trace_streamlines("SK")
        self.create_lattice(bool_plot=True)
      
class Lattice:

    def __init__(
        self,
        directory: Utils.Directory,
        case_number: int,
        reference_case: int = 0,
        reference_field: int = 0,
        RVE_identifier: int = 0,
        trace_lines: Tracer | None = None,
        grid_data: Tracer | None = None,
        aerofoil_coords: dict | None = None,
        fairing_chord: float | None = None,
        tolerance=0.5e-3,
    ):
        self.directory = directory
        self.case_number = case_number
        self.reference_case = reference_case
        self.reference_field = reference_field
        self.RVE_identifier = RVE_identifier
        self.lattice_lines = trace_lines
        self.grid_data = grid_data
        self.aerofoil_coords = aerofoil_coords
        self.fairing_chord = fairing_chord
        self.tolerance = tolerance

    @staticmethod
    def refine_lines(lines , min_length=0.02):
        """
        Refine lines by incrementing points to ensure no segment exceeds min_length.
        Parameters:
            lines (list): List of polylines, each as np.ndarray of points with shape (n, 2)
            min_length (float): Minimum length for each segment in the polyline.
        Returns:
            list: List of refined polylines.
        """

        # Refine lines
        num_workers = min((os.cpu_count() - 2, len(lines)))    
        args = [(line, min_length) for line in lines]
        with ThreadPoolExecutor(num_workers) as executor:
            lines = list(executor.map(lambda ab: Utils.GeoOps.increment_line(*ab), args))

        return lines

    def create_fairing_2D(self, min_edge_length=0.02, bool_plot=False):
        """
        Creates a 2D mesh from lattice lines for each increment.
        Uses triangulation to create triangular elements.
        """

        # Load lattice data
        if self.lattice_lines is None:
            lattice_file = os.path.join(self.directory.case_folder, "data", f"{self.reference_case}_lattice_lines.pickle")
            if os.path.exists(lattice_file):
                self.lattice_lines = Utils.ReadWriteOps.load_object(
                    os.path.join(self.directory.case_folder, "data", f"{self.reference_case}_lattice_lines"),
                    method="pickle"
                )[f"{self.reference_field}"] 
            else:
                print("WARNING: Lattice data not found, now tracing lattice.")
                lattice = Tracer(self.directory, self.reference_case)
                lattice.analysis()
                self.lattice_lines = lattice.lattice_lines[f"{self.reference_field}"]

        if self.fairing_chord is None:
            FR_input = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "input", f"{self.case_number}_FR"), "json")
            self.fairing_chord = FR_input["fairing_chord"]

        # Refine lines
        min_length = 1e-2*self.fairing_chord
        for key in self.lattice_lines.keys():
            self.lattice_lines[key] = self.refine_lines(self.lattice_lines[key], min_length)

        # Create 2D mesh for each increment
        self.mesh2D = {}
        self.mesh2D = Mesh(2, [2,3])

        # Create nodes and groups from lines
        for group_name, group_lines in self.lattice_lines.items():
            # Add lines to mesh
            self.mesh2D.add_lines_to_mesh(0, group_lines, group_name, self.tolerance)

        if bool_plot:
            print(f"Plotting: rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_lines_2D: nodes {self.mesh2D.nodes.shape}, elements {self.mesh2D.elements[0].shape}")
            self.mesh2D.plot_2D(
                el_list_idx=0,
                bool_nodes=False,
                bool_elements=True,
                label=False,
                save_path=os.path.join(self.directory.case_folder, "fig", f"rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_lines_2D.svg")
            )

        # Triangulate
        self.mesh2D.triangulate(1, np.arange(self.mesh2D.nodes.shape[0]))

        if bool_plot:
            print(f"Plotting: rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_triangles_2D: nodes {self.mesh2D.nodes.shape}, elements {self.mesh2D.elements[1].shape}")
            self.mesh2D.plot_2D(
                el_list_idx=1,
                bool_nodes=False,
                bool_elements=True,
                label=False,
                save_path=os.path.join(self.directory.case_folder, "fig", f"rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_triangles_2D.svg")
            )

    def init_f_interp_2D_to_3D(self):
        """
        Creates an RBF interpolator to map 2D nodes to 3D space.
        Uses grid data to create the mapping.
        """
        # Load grid data
        if self.grid_data is None:
            self.grid_data = Utils.ReadWriteOps.load_object(
                os.path.join(self.directory.case_folder, "data", f"{self.reference_case}_grid_data"),
                method="pickle"
            )
        else:
            self.grid_data = self.grid_data

        # Interpolation data
        node_coords_3D = self.grid_data["surface_nodes_coords"][self.grid_data["node_index_grid"]].reshape((-1,3))
        node_coords_2D = self.grid_data["nodes_grid_2D"].reshape((-1,2))

        # Create RBF interpolator
        f_interp_2D_to_3D = interpolate.RBFInterpolator(node_coords_2D, node_coords_3D, neighbors=4, kernel='linear')

        return f_interp_2D_to_3D

    def init_f_normals_3D(self):
        # Load aerofoil coordinates
        if self.aerofoil_coords is None:
            self.aerofoil_coords = Utils.ReadWriteOps.load_object(
                os.path.join(self.directory.case_folder, "data", f"{self.reference_case}_aerofoil_coords"),
                method="pickle"
            )
        else:
            self.aerofoil_coords = self.aerofoil_coords

        # Interpolate normals at mid-plane
        coords = self.aerofoil_coords['mid']
        normals = Utils.GeoOps.normals_2D(coords)

        # RBF interpolation
        f_normals_3D = interpolate.RBFInterpolator(coords, normals, neighbors=4, kernel='linear')

        return f_normals_3D

    def load_cell_data(self):
        RVE_derived = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "input", f"{self.RVE_identifier}_UC_derived"), "json")
        RVE_input = Utils.ReadWriteOps.load_object(os.path.join(self.directory.case_folder, "input", f"{self.RVE_identifier}_UC"), "json")

        self.cell_dimensions = [RVE_derived["lx"], RVE_derived["ly"], RVE_derived["lz"]]
        self.chevron_angle = RVE_input["chevron_angle"]

    def mapping_2D_to_3D(self, bool_plot=False):
        """
        Maps 2D mesh nodes to 3D space using RBF interpolation. Merges trailing edge nodes in the inner surface.
        """

        # Initialize interpolator
        self.f_interp_2D_to_3D = self.init_f_interp_2D_to_3D()
        self.f_normals_3D = self.init_f_normals_3D()

        # load panel data
        self.load_cell_data()

        # collecting and verifying trailing edge nodes
        nset_TE_TOP = self.mesh2D.nsets()["TE_TOP"]
        nset_TE_BOTTOM = self.mesh2D.nsets()["TE_BOTTOM"]
        assert nset_TE_TOP.size == nset_TE_BOTTOM.size, "Error in trailing edge node sets."
        assert np.allclose(self.mesh2D.nodes[nset_TE_TOP, 1], self.mesh2D.nodes[nset_TE_BOTTOM, 1]), "Error in trailing edge node Y-coords."

        # Initialize 3D mesh
        self.mesh3D = Mesh(3, [3])

        # Node indices
        node_indices_2D = np.arange(self.mesh2D.nodes.shape[0])

        # Map nodes
        nodes_3D_midplane = self.f_interp_2D_to_3D(self.mesh2D.nodes)

        # Offset nodes in normal direction to create thickness
        offset = np.zeros_like(nodes_3D_midplane)
        offset[:,[0,2]] = self.f_normals_3D(nodes_3D_midplane[:,[0,2]]) * self.cell_dimensions[2] / 2.0

        # Create 3D nodes
        # For outer nodes
        node_indices_3D_outer = self.mesh3D.add_nodes(nodes_3D_midplane-offset, "OUTER_SURFACE")
        
        # For inner nodes, skipping trailing edge nodes (they will be merged later)
        node_indices_3D_inner = np.zeros(node_indices_3D_outer.shape[0], dtype=int)
        inner_nodes = nodes_3D_midplane+offset
        non_TE_mask = ~(np.isin(node_indices_2D, nset_TE_TOP) | np.isin(node_indices_2D, nset_TE_BOTTOM))
        node_indices_3D_inner[non_TE_mask] = self.mesh3D.add_nodes(inner_nodes[non_TE_mask], "INNER_SURFACE")

        # Merging trailing edge nodes
        node_TE = (inner_nodes[nset_TE_BOTTOM] + inner_nodes[nset_TE_TOP]) / 2.0
        node_indices = self.mesh3D.add_nodes(node_TE, "INNER_SURFACE")
        node_indices_3D_inner[nset_TE_BOTTOM] = node_indices
        node_indices_3D_inner[nset_TE_TOP] = node_indices

        # Check outer nodes indices are same as mid-plane nodes
        assert np.all(node_indices_3D_outer == node_indices_2D), "Error in node mapping."
        # Check inner nodes indices are same size as outer nodes (some inner nodes may be merged)
        assert np.all(node_indices_3D_inner.size == node_indices_3D_outer.size), "Error in node mapping."

        # helper function to map node indices
        def map_node_indices(elements, new_indices):
            return new_indices[elements]

        # create triangle elements
        for label, new_indices in zip(["OUTER_SURFACE", "INNER_SURFACE"], [node_indices_3D_outer, node_indices_3D_inner]):
            self.mesh3D.add_elements(
                0,
                map_node_indices(
                    self.mesh2D.elements[1], new_indices
                ),
                label,
            )

        # Create core elements
        beam_elements = self.mesh2D.elements[0]
        beam_elsets = self.mesh2D.elsets()[0]

        for label in ["STRINGERS", "CHEVRONS", "RIBS"]:

            # Map inner and outer element sets
            elements_inner = map_node_indices(beam_elements[beam_elsets[label]], node_indices_3D_inner)
            elements_outer = map_node_indices(beam_elements[beam_elsets[label]], node_indices_3D_outer)

            # Create elements in 3D mesh
            elements = []
            for el_inner, el_outer in zip(elements_inner, elements_outer):
                # new connectivity for two triangles forming a quad
                elements.extend([
                        [el_inner[0], el_outer[1], el_outer[0]],
                        [el_inner[0], el_inner[1], el_outer[1]]
                ])
            elements = np.array(elements, dtype=int)

            # Add elements to mesh
            self.mesh3D.add_elements(0, elements, label)

        if bool_plot:
            print(f"Plotting: rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_triangles_3D: nodes {self.mesh2D.nodes.shape}, elements {self.mesh2D.elements[0].shape}")

            self.mesh3D.plot_3D(
                el_list_idx=0,
                bool_nodes=False,
                bool_elements=True,
                label=False,
                save_path=os.path.join(self.directory.case_folder, "fig", f"rc{self.reference_case}_rf{self.reference_field}_cn{self.case_number}_triangles_3D.svg"),
                show=False
            )

    # NOTE: Currently not in use
    # def validate_midplane_mesh(self):

    #     for increment_key in self.mesh3D.keys():
    #         # check mesh
    #         degenerates = self.mesh3D.check_mesh_integrity()
    #         if degenerates['unused_nodes'].size > 0:
    #             print("Unused nodes:")
    #             print(self.mesh3D.nodes[degenerates['unused_nodes']])
    #         if degenerates['degenerate_beams'].size > 0:
    #             self.mesh3D.remove_beams(degenerates['degenerate_beams'])
    #         if degenerates['degenerate_triangles'].size > 0:
    #             print(self.mesh3D.triangles[degenerates['degenerate_triangles']])
    #         if degenerates['missing_beam_edges_in_triangles'].size > 0:
    #             print(f"WARNING: {degenerates['missing_beam_edges_in_triangles'].size} beam edges missing in triangles:")
    #         degenerates = self.mesh3D.check_mesh_integrity()
    #         if degenerates['unused_nodes'].size > 0 and degenerates['degenerate_beams'].size > 0 and degenerates['degenerate_triangles'].size > 0 and degenerates['missing_beam_edges_in_triangles'].size > 0:
    #             raise ValueError("Mesh integrity check failed, degenerates found.")

    #         # check repeated sets
    #         beam_sets = ["TE_top", "TE_bottom", "Ribs","Chevrons","Stringers"]
    #         for i, key_i in enumerate(beam_sets):
    #             for j, key_j in enumerate(beam_sets[i+1:]):
    #                 repeat_index = np.isin(self.mesh3D.beam_sets[key_j], self.mesh3D.beam_sets[key_i])
    #                 if np.any(repeat_index):
    #                     print(f"WARNING: Beams of '{key_i}' repeated in '{key_j}' are removed from '{key_j}'.")
    #                     self.mesh3D.beam_sets[key_j] = self.mesh3D.beam_sets[key_j][~repeat_index]

    def write_mesh(self):
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
    

        # initialise
        lines = []

        # nodes
        lines.append("*NODE")
        for i, node in enumerate(self.mesh3D.nodes):
            i += 1  # Abaqus uses 1-based indexing
            lines.append("%i, %f, %f, %f" % (i, *node))

        # elements
        element_type = {2: "B31", 3: "CPS3"}
        element_count = 0
        for elements in self.mesh3D.elements:
            # element header
            lines.append("*ELEMENT, TYPE=%s" % (element_type[elements.shape[1]]))
            for i, element in enumerate(elements):
                i += 1  # Abaqus uses 1-based indexing
                element += element_count + 1  # Offset for multiple types of elements, Abaqus uses 1-based indexing
                lines.append(", ".join(map(str, (i, *element))) + ("," if i < len(elements) else ""))
            # update element count
            element_count += elements.shape[0]

        # node sets
        for set_name, set_items in self.mesh3D.nsets().items():
            lines.append("*NSET, NSET=%s" % (set_name))
            set_items += 1  # Abaqus uses 1-based indexing
            lines.extend(format_lines(set_items))
        # element sets
        for elset in self.mesh3D.elsets():
            for set_name, set_items in elset.items():
                set_items += 1  # Abaqus uses 1-based indexing
                lines.append("*ELSET, ELSET=%s" % (set_name))
                lines.extend(format_lines(set_items))

        # write to file
        with open(os.path.join(self.directory.case_folder, "mesh", f"{self.case_number}_fairing_geometry.inp"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @Utils.logger
    def analysis(self, min_edge_length=0.02):
        print(f"Starting {self.__class__.__name__} analysis {self.directory.case_name} - {self.case_number}")
        
        # Create 2D fairing mesh
        self.create_fairing_2D(min_edge_length=min_edge_length)
        # Map 2D mesh to 3D
        self.mapping_2D_to_3D()
        # Write mesh to file
        self.write_mesh()


if __name__ == "__main__":
    # Example
    directory = Utils.Directory(case_name="test_case_9")
    case_number = 1 # current case number

    # Trace Lattice
    lattice_data = Tracer(directory, 0)
    lattice_data.analysis()

    # Reference for traced field
    reference_case = 0 # reference case for the field data - this must be an equivalent model shell model
    reference_field = 0 # rotation angle for the folding wingtip at which deformation field is extracted

    # Tailored Geometry
    tailored = Lattice(directory, case_number, reference_case, reference_field)
    # tailored.analysis()

    tailored.create_fairing_2D(bool_plot=True)
    tailored.mapping_2D_to_3D(bool_plot=True)
    tailored.write_mesh()

