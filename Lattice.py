
import Utils
from Fairing import FairingData

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
        print(f"\tLattice field for folding angles (deg): {[f'{angle:.1f}' for angle in np.rad2deg(self.rotation_angles[self.indices])]}")

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
            intersection_threshold = 10
            if len(intersection_points) >= intersection_threshold:
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
                "CHEVRONS": spanwise,
                "STRINGERS": chordwise,
                "RIBS": ribs,
                "TE_TOP": trailing_edge[0][np.newaxis, ...],
                "TE_BOTTOM": trailing_edge[1][np.newaxis, ...],
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


if __name__ == "__main__":
    # Example
    directory = Utils.Directory(case_name="test_case_7")
    case_number = 0

    # Trace Lattice
    lattice_data = Lattice(directory, case_number)
    lattice_data.analysis()