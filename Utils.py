import os
import time
import traceback
import pickle
import json
import numpy as np
import subprocess
import matplotlib.pyplot as plt


def logger(func):
    """Decorator to log the execution time of a function."""

    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            print(
                f"\tUPDATE: {func.__name__} for {args[0].directory.case_name} - {args[0].case_number} completed in "
                f"{time.time() - start:.2f}s"
            )
            return result
        except:
            print(
                f"\tUPDATE: {func.__name__} for {args[0].directory.case_name} - {args[0].case_number} failed to complete"
            )
            print(traceback.format_exc())
            return None

    return wrapper

def run_analysis(job):
    """
    Top-level function used for parallel execution of the analysis method of the given job object.
    """
    job.analysis()

def indexed_function_caller(func, num, *args):
    """
    Wrapper function to unpack arguments for multiprocessing.
    """
    if not callable(func):
        raise ValueError("func must be a callable function")

    return np.array((*args[:num], func(*args[num:])), dtype=object)

def indices(A, B):
    """
    For 1D arrays A and B, returns the indices of B in A in the order the values exist in B

    :param A: 1D numpy array
    :param B: 1D numpy array
    :return: 1D numpy array of indices
    """
    sort_idx = np.argsort(A)
    indices_B_in_A = sort_idx[np.searchsorted(A, B, sorter=sort_idx)]
    return np.array(indices_B_in_A)

def run_subprocess(command, run_folder, log_file):
    """Runs a shell command and captures its output."""
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=run_folder,
            input="y",
            text=True,
            capture_output=True,
        )
        with open(log_file, "a") as log:
            log.write(f"--- LOG TIME : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"--- COMMAND : {command}\n")
            log.write(f"--- STDOUT ---\n{process.stdout}")
            log.write(f"--- STDERR ---\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Execution failed for command:\n{command}.")
        print(f"Return code: {e.returncode}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        with open(log_file, "a") as log:
            log.write(f"ERROR: Execution failed for command:\n{command}.\n")
            log.write(f"Return code: {e.returncode}\n")
            log.write(f"--- STDOUT ---\n{e.stdout}\n")
            log.write(f"--- STDERR ---\n{e.stderr}\n")

def save_object(obj, path, method):
    """Saves an object to a file using the specified method."""
    if method == "pickle":
        with open(path + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == "json":
        with open(path + ".json", "w") as f:
            json.dump(obj, f, indent=4)
    else:
        print("ERROR: Wrong file saving method")

def load_object(path, method):
    """Loads an object from a file using the specified method."""
    if method == "pickle":
        with open(path + ".pickle", "rb") as f:
            return pickle.load(f, encoding='latin1')
    elif method == "json":
        with open(path + ".json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("ERROR: Wrong file loading method")
        return None

class GeoOps:
    """
    A class collecting functions for geometric operations
    """

    @staticmethod
    def normals_2D(points):
        """
        Calculate the normal vector at a point on a 2D curve.

        Parameters:
            points (numpy.ndarray): Array of shape (n, 2)

        Returns:
            numpy.ndarray: Unit normal vector at the point
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"Input points must be an array of shape (n, 2). Received {points.shape}"
            )

        # initialise
        normals = np.empty(points.shape)
        # edges
        edges = np.diff(points, axis=0) 
        # perpendicular vectors
        normals[0] = (-edges[0, 1], edges[0, 0]) # forward difference
        normals[1:-1,:] = np.column_stack(( # central difference
            - np.c_[edges[:-1, 1],edges[1:, 1]].mean(axis=1), 
            + np.c_[edges[:-1, 0],edges[1:, 0]].mean(axis=1),
        ))  
        normals[-1] = (-edges[-1, 1], edges[-1, 0]) # backward difference
        # normalising vector
        magnitudes = np.linalg.norm(normals, axis=1)
        normals = normals / magnitudes[:, np.newaxis]  

        return normals    

    @staticmethod
    def intersection_point(line_1, line_2):
            """
            Calculate the intersection point of two lines defined by multiple points each.
            """
            # findingg the closest indexes
            matrix = line_1[:, None]-line_2[None, :]
            dist_squared_matrix = np.sum(np.power(matrix,2), axis=2)
            index = np.unravel_index(np.argmin(dist_squared_matrix),dist_squared_matrix.shape)

            # using line segements to find intersection points
            for i in range(max([0,index[0]-1]), min([index[0]+1, line_1.shape[0]-1])): # index for line 1
                for j in range(max([0,index[1]-1]), min([index[1]+1, line_2.shape[0]-1])): # index for line 2
                    try:
                        # parametrise the line segment with range 0 to 1
                        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
                        x1, y1 = line_1[i, 0], line_1[i, 1]
                        x2, y2 = line_1[i+1, 0], line_1[i+1, 1]
                        x3, y3 = line_2[j, 0], line_2[j, 1]
                        x4, y4 = line_2[j+1, 0], line_2[j+1, 1]
                        denom = ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)) + np.finfo(float).eps # to avoid division by zero
                        t1 = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
                        t2 = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom

                        # check the parameter range to verify if the point is with the segment
                        if all([t1>=0.0, t1<=1.0, t2>=0.0, t2<=1.0]):
                            # found intersection
                            return np.array([x1+t1*(x2-x1), y1+t1*(y2-y1)]), np.array([i,j])
                        
                        # just for debugging
                        # print(t1, t2, (x1,y1), (x2,y2), (x3,y3), (x4,y4))

                    except:
                        #pass
                        traceback.print_exc()


            # no intersection
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

    @staticmethod
    def bisect_line(line):
        """
        Bisects each pair of consecutive lines by computing their midpoints and interleaving them with the original lines.
        Parameters
            lines : np.ndarray : An array of shape (N, M) representing N lines, each with M coordinates.
        Returns     
            np.ndarray : An array of shape (2*N - 1, M) containing the original lines and their midpoints interleaved.
        """

        midpoints = line[:-1] + np.diff(line, axis=0) / 2
        new_lines = np.empty(
            (line.shape[0] + midpoints.shape[0], line.shape[1]), dtype=line.dtype
        )
        new_lines[0::2] = line
        new_lines[1::2] = midpoints
        return new_lines

class Units:
    """
    A class providing unit conversion methods.
    """

    @staticmethod
    def m2mm(value: float) -> float:
        """
        Converts a value from meters to millimeters.

        Parameters:
            value (float): The value in meters to be converted.

        Returns:
            float: The equivalent value in millimeters.
        """
        return value * 1000.0

    @staticmethod
    def mm2m(value: float) -> float:
        """
        Converts a value from millimeters to meters.

        Parameters:
            value (float): The value in millimeters to be converted.

        Returns:
            float: The converted value in meters.
        """
        return value / 1000.0

    @staticmethod
    def deg2rad(value: float) -> float:
        """
        Converts an angle from degrees to radians.

        Parameters:
            value (float): Angle in degrees.

        Returns:
            float: Angle in radians.
        """
        return np.deg2rad(value)

    @staticmethod
    def rad2deg(value: float) -> float:
        """
        Converts an angle from radians to degrees.

        Parameters:
            value (float): Angle in radians.

        Returns:
            float: Angle converted to degrees.
        """
        return np.rad2deg(value)

class Directory:
    def __init__(self, case_name: str = "default_case"):
        self.case_name = case_name
        self.run_folder = os.path.abspath(os.getcwd())
        self.case_folder = os.path.join(self.run_folder, case_name)
        self.abaqus_folder = os.path.join(self.run_folder, "abaqus_temp")
        for folder in [self.case_folder, self.abaqus_folder]:
            os.makedirs(folder, exist_ok=True)
        for sub in [
            "mesh",
            "input",
            "inp",
            "msg",
            "dat",
            "odb",
            "report",
            "data",
            "fig",
            "log",
            "trace",
            "sta",
            "temp",
        ]:
            os.makedirs(os.path.join(self.case_folder, sub), exist_ok=True)
        for sub in ["inp", "data"]:
            os.makedirs(
                os.path.join(self.case_folder, sub, "serialised"), exist_ok=True
            )

class Plots:

    @staticmethod
    def node_coupling(master_coords, slave_coords, xlabel="", ylabel="", save_path=None, show=False):
        """
        For each pair of master and slave coordinates, draws a line connecting them,
        and scatters the master and slave nodes with different colors. The plot is saved
        to a file in the specified case folder.
        Parameters:
            master_coords (np.ndarray): Array of shape (N, 2) containing coordinates of master nodes.
            slave_coords (np.ndarray): Array of shape (N, 2) containing coordinates of slave nodes.
            xlabel (str, optional): Label for the x-axis. Default is an empty string.
            ylabel (str, optional): Label for the y-axis. Default is an empty string.
            save_path (str, optional): Path to save the figure. If None, figure is not saved.
            show (bool, optional): Whether to display the plot. Default is False.
        """
        plt.figure(figsize=(8, 6))
        for m, s in zip(master_coords, slave_coords):
            plt.plot([m[0], s[0]], [m[1], s[1]], 'k-', lw=1)
        plt.scatter(master_coords[:, 0], master_coords[:, 1], color='blue', label='Master nodes')
        plt.scatter(slave_coords[:, 0], slave_coords[:, 1], color='red', label='Slave nodes')
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.title('Kinematic Coupling Node Pairs')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def aerofoils(aerofoils, save_path=None, show=False):
        """
        Plots multiple aerofoils on the same graph.
        Parameters:
            aerofoils (list or dict): A list or dictionary of aerofoil coordinates.
                          Each item should be a 2D array with shape (n, 2) where n is the number of points.
            save_path (str, optional): Path to save the figure. If None, figure is not saved.
            show (bool, optional): Whether to display the plot. Default is False.
        """

        plt.figure(figsize=(10, 6))

        # Check if input is a list or dictionary
        if isinstance(aerofoils, list):
            for i, coords in enumerate(aerofoils):
                plt.plot(coords[:, 0], coords[:, 1], label=f"Aerofoil {i}")
        elif isinstance(aerofoils, dict):
            for name, coords in aerofoils.items():
                plt.plot(coords[:, 0], coords[:, 1], label=str(name))
        else:
            raise TypeError(
                "Input must be a list or dictionary of aerofoil coordinates"
            )

        # Format the plot
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(aerofoils) if isinstance(aerofoils, list) else len(aerofoils.keys()),
            frameon=False,
        )
        plt.gca().set_aspect("equal")

        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def fairing_response(FairingResponse, save_path=None, show=False):
        """
        Plots the fairing response for a given dataset.

        Parameters:
            FairingResponse (dict or list): The fairing response data to plot.
            save_path (str, optional): Path to save the figure. If None, figure is not saved.
            show (bool, optional): Whether to display the plot. Default is False.
        """
        fig, ax1 = plt.subplots(figsize=(3, 3))
        num_colours = len(FairingResponse) if isinstance(FairingResponse, (list, tuple)) else 1
        colourmap = plt.get_cmap('rainbow', num_colours)
        colours = list(colourmap(i) for i in range(num_colours))

        for i, data in enumerate([FairingResponse]):
            rotation = data["Rotation"]
            torque = data["Torque"]
            distortion = Units.m2mm(data["Distortion"])


            ax1.plot(rotation, torque, color=colours[i], linestyle='-', label="Torque")
            ax1.set_xlabel("Rotation [deg]")
            ax1.set_ylabel("Torque [Nm] (solid)")

            ax2 = ax1.twinx()
            ax2.plot(rotation, distortion, color=colours[i], linestyle='--', label="Distortion")
            ax2.set_ylabel("Distortion [mm] (dashed)")

            # Save the figure if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

    @staticmethod
    def grid(coords_labels, save_path=None, show=False):

        coords = coords_labels["coords"]
        labels = coords_labels["labels"]

        # dimensions
        space_dim = coords[0,0,:].shape[0]

        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the grid lines (chordwise and spanwise)
        for i in range(coords.shape[0]):
            ax.plot(*(coords[i, :, axis] for axis in range(space_dim)),
                    color='g', marker='o', markersize=1, linewidth=0.5)
        for j in range(coords.shape[1]):
            ax.plot(*(coords[:, j, axis] for axis in range(space_dim)),
                    color='r', marker='o', markersize=1, linewidth=0.5)

        # Show element numbers next to centroids
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                ax.text(
                    *coords[i, j, :],
                    f"{labels[i, j]}, ({i},{j})",
                    fontsize=2,
                    color='k'
                )

        plt.gca().set_aspect("equal")
        plt.tight_layout()
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        if space_dim == 3:
            ax.set_zlabel('Z [m]')
        ax.set_title('Element Centroid Grid in 3D')

         # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def grid_split(coords_labels, save_path=None, show=False):

        try:
            coords = coords_labels["coords"]
            bool_coords = True
        except:
            bool_coords = False

        try:
            labels = coords_labels["labels"]
            bool_label = True
        except:
            bool_label = False

        try:
            border = coords_labels["border"]
            bool_border = True
        except:
            bool_border = False

        try:
            vectors_coord = coords_labels["vectors_coords"]
            vectors_SE = coords_labels["vectors_SE"]
            bool_quiver_SE = True
        except:
            bool_quiver_SE = False

        try:
            vectors_coord = coords_labels["vectors_coords"]
            vectors_SK = coords_labels["vectors_SK"]
            bool_quiver_SK = True
        except:
            bool_quiver_SK = False

        try:
            lines = coords_labels["lines"]
            bool_lines = True
        except:
            bool_lines = False


        fig, ax=plt.subplots(2)

        # Find min and max for each coordinate axis (axis=2)
        mins = coords.min(axis=(0, 1))
        maxs = coords.max(axis=(0, 1))
        diff = (maxs - mins).min()
        padding = 0.1 * diff  # 10% padding on each side
        mins -= padding
        maxs += padding

        # Plot the grid lines (chordwise and spanwise)
        for i_ax, _ in enumerate(ax):

            ax[i_ax].set_xlabel('$\\eta$ [m]')
            ax[i_ax].set_ylabel('$\\zeta$ [m]')
            x_min = [0, mins[0]][i_ax]
            x_max = [maxs[0], 0][i_ax]
            y_min, y_max = mins[1], maxs[1]
            ax[i_ax].set_xlim([x_min, x_max])
            ax[i_ax].set_ylim([y_min, y_max])
            ax[i_ax].set_aspect("equal")

            # mask
            mask = np.logical_and.reduce((
                coords[:, :, 0] >= x_min,
                coords[:, :, 0] <= x_max,
                coords[:, :, 1] >= y_min,
                coords[:, :, 1] <= y_max
            ))

            if bool_coords:
                # Plot the grid lines (chordwise and spanwise)
                for i in range(coords.shape[0]):
                    ax[i_ax].plot(*(coords[i, :, axis] for axis in range(2)),
                        color='gray', marker='.', markersize=0.15, linewidth=0.15, alpha=0.25)
                for j in range(coords.shape[1]):
                    ax[i_ax].plot(*(coords[:, j, axis] for axis in range(2)),
                        color='gray', marker='.', markersize=0.15, linewidth=0.15, alpha=0.25)

            if bool_quiver_SE:
                # Plot vectors at each grid point
                ax[i_ax].quiver(
                    *(vectors_coord[..., axis].ravel() for axis in range(2)),
                    *(vectors_SE[...,0, axis].ravel() for axis in range(2)),
                    color='green',
                    angles='xy', width=0.001, scale_units='xy', scale=75
                )
                ax[i_ax].quiver(
                    *(vectors_coord[..., axis].ravel() for axis in range(2)),
                    *(vectors_SE[..., 1, axis].ravel() for axis in range(2)),
                    color='magenta',
                    angles='xy', width=0.001, scale_units='xy', scale=75
                )

            if bool_quiver_SK:
                # curvature
                ax[i_ax].quiver(
                    *(vectors_coord[..., axis].ravel() for axis in range(2)),
                    *(vectors_SK[...,0, axis].ravel() for axis in range(2)),
                    color='orange',
                    angles='xy', width=0.001, scale_units='xy', scale=75
                )
                ax[i_ax].quiver(
                    *(vectors_coord[..., axis].ravel() for axis in range(2)),
                    *(vectors_SK[..., 1, axis].ravel() for axis in range(2)),
                    color='pink',
                    angles='xy', width=0.001, scale_units='xy', scale=75
                )

            # Show element numbers next to where mask is True
            if bool_label:
                for i in range(coords.shape[0]):
                    for j in range(coords.shape[1]):
                        if mask[i, j]:
                            position = coords[i, j]
                            label = f"{i},{j}"#{labels[i, j]}"
                            ax[i_ax].text(
                                *position,
                                label,
                                fontsize=2,
                                color='k'
                            )  

            # show borders
            if bool_border:
                ax[i_ax].plot(
                    *(np.r_[border[:, axis], border[0, axis]] for axis in range(2)),
                    color='black', marker='o', markersize=0.15, linewidth=0.5
                )

            if bool_lines:
                num_groups = len(lines)
                cmap = plt.get_cmap('rainbow', num_groups)
                colours = [cmap(i) for i in range(num_groups)]
                for i, (key, onetype) in enumerate(lines.items()):
                    for j, oneline in enumerate(onetype):
                        ax[i_ax].plot(
                            *(oneline[:, axis] for axis in range(2)),
                            color=colours[i], marker='o', markersize=0.15, linewidth=0.5,
                            label=f"{key}" if j==0 else None
                        )


        fig.tight_layout()

         # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
