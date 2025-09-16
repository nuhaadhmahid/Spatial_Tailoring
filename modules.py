from typing import Any
import pickle
import json
import numpy as np
from scipy.spatial import Delaunay
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import multiprocessing as mp
import traceback


class Utils:

    @staticmethod
    def save_object(obj:Any, path:str, method:str):
        """
        Saves a Python object to a file using the specified serialization method.

        Parameters:
            obj (any): The Python object to be saved.
            path (str): The file path (without extension) where the object will be saved.
            method (str): The serialization method to use. Supported values are:
                - 'pickle': Saves the object using Python's pickle module (binary format).
                - 'json': Saves the object in JSON format (text format).
        """
        match method:
            case "pickle":
                try:
                    with open(path + ".pickle", "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                except (pickle.PickleError, OSError) as ex:
                    print("ERROR: pickle save failed: ", ex)
            case "json":
                try:
                    with open(path + ".json", "w", encoding="utf-8") as f:
                        json.dump(obj, f, sort_keys=True, indent=4)
                except (TypeError, OSError, json.JSONDecodeError) as ex:
                    print("ERROR: json save failed: ", ex)
            case _:
                print("ERROR: Wrong fail saving method")

    @staticmethod
    def load_object(path:str, method:str) -> Any:
        """
        Loads a Python object from a file using the specified serialization method.

        Parameters:
            path (str): The base file path (without extension) from which to load the object.
            method (str): The serialization method to use ('pickle' or 'json').

        Returns:
            obj: The loaded Python object, or None if loading fails.
        """

        obj = None  # Ensure obj is always defined
        match method:
            case "pickle":
                try:
                    with open(path + ".pickle", "rb") as f:
                        obj = pickle.load(f, encoding="latin1")
                except (pickle.PickleError, OSError) as ex:
                    print("ERROR: pickle load failed: ", ex)
            case "json":
                try:
                    with open(path + ".json", "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except (TypeError, OSError, json.JSONDecodeError) as ex:
                    print("ERROR: json load failed: ", ex)
            case _:
                print("ERROR: Wrong fail saving method")
        return obj
    
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

    def plot_groups_2D(self, filename: str, group_keys: list[str] = [], label: bool = False):
        """
        Plots groups of lines from mesh.groups. Each group contains lines as arrays of node indices.
        Optionally, only specified group_keys are plotted. Plots on two subplots (top/bottom surface).

        Parameters:
            filename (str): Output filename (without extension).
            group_keys (list[str], optional): List of group keys to plot. If None, plot all groups.
            label (bool): Whether to label the lines with their index.
        """
        fig, axes = plt.subplots(2)
        keys = group_keys if len(group_keys) > 0 else list(self.groups.keys())
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        x_top_lim = [0.0, 1.68]
        x_bot_lim = [-1.68, 0.0]
        y_lim = [-0.05, 0.4135]
        labelled_node = []
        for i, ax in enumerate(axes):
            for row, group_key in enumerate(keys):
                group_lines = self.groups[group_key]
                for idx, line_indices in enumerate(group_lines):
                    if line_indices.shape[0] > 0:
                        line_indices = np.array(line_indices, dtype=int)
                        line_coords = self.nodes[line_indices]
                        ax.plot(line_coords[:, 0], line_coords[:, 1], color=colours[row], marker='.', markersize=0.5,
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
            ax.set_xlabel('$\\eta$-Axis', fontsize=9)
            ax.set_ylabel('$\\zeta$-Axis', fontsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_aspect('equal')

        axes[0].set_title('Top Surface', loc='left', fontsize=9)
        axes[1].set_title('Bottom Surface', loc='left', fontsize=9)
        axes[0].set_xlim(x_top_lim)
        axes[1].set_xlim(x_bot_lim)
        axes[0].set_ylim(y_lim)
        axes[1].set_ylim(y_lim)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.925), ncol=len(labels), frameon=False)

        fig.tight_layout()
        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)

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

class Panel2D:

    def __init__(
        self,
        chevron_length: float = 0.0,
        chevron_angle_rad: float = 0.0,
        chevron_spacing: float = 0.0,
        mesh: Mesh = Mesh(2),
        trace_data = {}
    ) -> None:
        """
        Initializes the Lattice class with parameters for chevron length, angle, spacing, and nodes.
        """
        self.chevron_length = chevron_length
        self.chevron_angle_rad = chevron_angle_rad
        self.chevron_spacing = chevron_spacing
        self.mesh = mesh

    def plot_nodes(self, filename: str, nset_keys: list[str] = [], label: bool = False):
        """
        Plots nodes from mesh.nodes or mesh.nsets. Optionally, only specified nset_keys are plotted.
        Plots on two subplots (top/bottom surface), similar to plot_groups_2D.

        Parameters:
            filename (str): Output filename (without extension).
            nset_keys (list[str], optional): List of nset keys to plot. If empty, plot all nodes.
            label (bool): Whether to label the nodes with their index.
        """
        fig, axes = plt.subplots(2)
        # Determine which nodes to plot
        if nset_keys and all(k in self.mesh.nsets for k in nset_keys):
            keys = nset_keys
            nodes_by_key = {k: self.mesh.nodes[self.mesh.nsets[k]] for k in keys}
        else:
            keys = ["all"]
            nodes_by_key = {"all": self.mesh.nodes}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        x_top_lim = [0.0, 1.68]
        x_bot_lim = [-1.68, 0.0]
        y_lim = [-0.05, 0.4135]
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
            ax.set_xlabel('$\\eta$-Axis', fontsize=9)
            ax.set_ylabel('$\\zeta$-Axis', fontsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_aspect('equal')

        axes[0].set_title('Top Surface', loc='left', fontsize=9)
        axes[1].set_title('Bottom Surface', loc='left', fontsize=9)
        axes[0].set_xlim(x_top_lim)
        axes[1].set_xlim(x_bot_lim)
        axes[0].set_ylim(y_lim)
        axes[1].set_ylim(y_lim)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.925), ncol=len(labels), frameon=False)

        fig.tight_layout()
        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)
    
    def plot_beams(self, filename: str, beam_set_keys: list[str] = [], label: bool = False):
        """
        Plots beams from mesh.beams or mesh.beam_sets. Optionally, only specified beam_set_keys are plotted.
        Plots on two subplots (top/bottom surface), similar to plot_groups_2D.

        Parameters:
            filename (str): Output filename (without extension).
            beam_set_keys (list[str], optional): List of beam set keys to plot. If empty, plot all beams.
            label (bool): Whether to label the beams with their index.
        """
        fig, axes = plt.subplots(2)
        # Determine which beams to plot
        if beam_set_keys and all(k in self.mesh.beam_sets for k in beam_set_keys):
            keys = beam_set_keys
            beams_by_key = {k: self.mesh.beams[self.mesh.beam_sets[k]] for k in keys}
        else:
            keys = ["all"]
            beams_by_key = {"all": self.mesh.beams}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        x_top_lim = [0.0, 1.68]
        x_bot_lim = [-1.68, 0.0]
        y_lim = [-0.05, 0.4135]
        for i, ax in enumerate(axes):
            for row, key in enumerate(keys):
                beams = beams_by_key[key]
                for idx, beam in enumerate(beams):
                    beam = np.array(beam, dtype=int)
                    beam_coords = self.mesh.nodes[beam]
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
            ax.set_xlabel('$\\eta$-Axis', fontsize=9)
            ax.set_ylabel('$\\zeta$-Axis', fontsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_aspect('equal')

        axes[0].set_title('Top Surface', loc='left', fontsize=9)
        axes[1].set_title('Bottom Surface', loc='left', fontsize=9)
        axes[0].set_xlim(x_top_lim)
        axes[1].set_xlim(x_bot_lim)
        axes[0].set_ylim(y_lim)
        axes[1].set_ylim(y_lim)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.925), ncol=len(labels), frameon=False)

        fig.tight_layout()
        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)
        
    def plot_triangles(self, filename: str, triangle_set_keys: list[str] = [], label: bool = False):
        """
        Plots triangles from mesh.triangles or mesh.triangle_sets. Optionally, only specified triangle_set_keys are plotted.
        Plots on two subplots (top/bottom surface), similar to plot_groups_2D.

        Parameters:
            filename (str): Output filename (without extension).
            triangle_set_keys (list[str], optional): List of triangle set keys to plot. If empty, plot all triangles.
            label (bool): Whether to label the triangles with their index.
        """
        
        fig, axes = plt.subplots(2)
        # Determine which triangles to plot
        if triangle_set_keys and all(k in self.mesh.triangle_sets for k in triangle_set_keys):
            keys = triangle_set_keys
            triangles_by_key = {k: self.mesh.triangles[self.mesh.triangle_sets[k]] for k in keys}
        else:
            keys = ["all"]
            triangles_by_key = {"all": self.mesh.triangles}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        x_top_lim = [0.0, 1.68]
        x_bot_lim = [-1.68, 0.0]
        y_lim = [-0.05, 0.4135]
        for i, ax in enumerate(axes):
            for row, key in enumerate(keys):
                triangles = triangles_by_key[key]
                for idx, triangle in enumerate(triangles):
                    pts = self.mesh.nodes[triangle][:, :2]
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
            ax.set_xlabel('$\\eta$-Axis', fontsize=9)
            ax.set_ylabel('$\\zeta$-Axis', fontsize=9)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_aspect('equal')
            ax.autoscale()

        axes[0].set_title('Top Surface', loc='left', fontsize=9)
        axes[1].set_title('Bottom Surface', loc='left', fontsize=9)
        axes[0].set_xlim(x_top_lim)
        axes[1].set_xlim(x_bot_lim)
        axes[0].set_ylim(y_lim)
        axes[1].set_ylim(y_lim)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.925), ncol=len(labels), frameon=False)

        fig.tight_layout()
        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)

class Wing3D:
    def __init__(self) -> None:
        """
        Initializes the Nodes class with an array of nodes.
        """
        self.mesh = Mesh(dim=3)  # Initialize with 3D mesh

    def mapping_2D_to_3D(self, fun_interpolation, mesh_2D):

        # Merge trailing edge nodes
        trailing_edge_indices = mesh_2D.groups['Trailing_Edge']
        other_indices = np.empty((0), dtype=int)
        for i, trailing_edge_index in enumerate(trailing_edge_indices):
            if i==0:
                first_indices = trailing_edge_index.astype(int)
            else:
                other_indices = np.append(other_indices, trailing_edge_index.astype(int))

        # Mapping nodes from 2D to 3D and creating them while avoiding repeating trailing edge nodes
        nodes_3D = fun_interpolation(mesh_2D.nodes)
        index_to_map = ~np.isin(np.arange(nodes_3D.shape[0]), other_indices)
        nodes_to_map = nodes_3D[index_to_map] # unique nodes to create
        nodes_to_map = self.mesh.add_or_merge_nodes(nodes_to_map, tolerance=1e-4)
        index_3D = np.zeros(mesh_2D.nodes.shape[0], dtype=int) # initialize index array
        index_3D[index_to_map] = nodes_to_map 
        index_3D[other_indices] = index_3D[first_indices] # mapping index of repeated nodes to original at TE

        # Map 2D node indices to 3D node indices for beams and triangles
        self.mesh.beams = np.array([[index_3D[i0], index_3D[i1]] for i0, i1 in mesh_2D.beams], dtype=int)
        self.mesh.triangles = np.array([[index_3D[i0], index_3D[i1], index_3D[i2]] for i0, i1, i2 in mesh_2D.triangles], dtype=int)

        # map over groups
        for group_key, group_lines in mesh_2D.groups.items():
            new_lines = []
            for line in group_lines:
                mapped_line = index_3D[line.astype(int)]
                new_lines.append(mapped_line)
            self.mesh.groups[group_key] = np.array(new_lines, dtype=object)

        # using groups create sets for beams
        sorted_beams = np.sort(self.mesh.beams, axis=1)
        for group_key, group_lines in self.mesh.groups.items():
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
            self.mesh.beam_sets[group_key] = np.array(beam_indices, dtype=int)

    def plot_nodes(self, filename: str, nset_keys: list[str] = [], label: bool = False):
        """
        Plots 3D nodes from mesh.nodes or mesh.nsets. Optionally, only specified nset_keys are plotted.

        Parameters:
            filename (str): Output filename (without extension).
            nset_keys (list[str], optional): List of nset keys to plot. If empty, plot all nodes.
            label (bool): Whether to label the nodes with their index.
        """
        fig = plt.figure(figsize=(7.5,4))
        ax = fig.add_subplot(111, projection='3d')
        # Determine which nodes to plot
        if nset_keys and all(k in self.mesh.nsets for k in nset_keys):
            keys = nset_keys
            nodes_by_key = {k: self.mesh.nodes[self.mesh.nsets[k]] for k in keys}
        else:
            keys = ["all"]
            nodes_by_key = {"all": self.mesh.nodes}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        points = []
        for row, key in enumerate(keys):
            nodes = nodes_by_key[key]
            ax.scatter(
                nodes[:, 0], nodes[:, 1], nodes[:, 2],
                color=colours[row], s=3, marker='o',
                alpha=0.6, label=key if row == 0 else None
            )
            points.extend(nodes)
            if label:
                for idx, (x, y, z) in enumerate(nodes):
                    ax.text(x, y, z, f"n:{idx}", color=colours[row], fontsize=6)
            
        points = np.array(points)
        ax.yaxis.set_ticks(np.arange(np.min(points[:,1]), np.max(points[:,1])+0.01, 0.2))
        ax.zaxis.set_ticks(np.arange(-0.1, 0.1 +0.01, 0.1))

        # decorations
        ax.set_xlabel("Chord, X", fontsize=9)
        ax.set_ylabel("Span, Y", fontsize=9)
        ax.set_zlabel("Thickness, Z", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.xaxis.labelpad = 25

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',fontsize=9, bbox_to_anchor=(1, 1), ncol=len(labels), frameon=False)

        ax.set_box_aspect((np.ptp(points[:,0]),np.ptp(points[:,1]),np.ptp(points[:,2]),), zoom=1.1) 
        plt.subplots_adjust(top = 1, bottom = 0, right = 1.15, left = 0, hspace = 0, wspace = 0)

        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_beams(self, filename: str, beam_set_keys: list[str] = [], label: bool = False):
        """
        Plots 3D beams from mesh.beams or mesh.beam_sets. Optionally, only specified beam_set_keys are plotted.

        Parameters:
            filename (str): Output filename (without extension).
            beam_set_keys (list[str], optional): List of beam set keys to plot. If empty, plot all beams.
            label (bool): Whether to label the beams with their index.
        """
        fig = plt.figure(figsize=(7.5, 4))
        ax = fig.add_subplot(111, projection='3d')
        # Determine which beams to plot
        if beam_set_keys and all(k in self.mesh.beam_sets for k in beam_set_keys):
            keys = beam_set_keys
            beams_by_key = {k: self.mesh.beams[self.mesh.beam_sets[k]] for k in keys}
        else:
            keys = ["all"]
            beams_by_key = {"all": self.mesh.beams}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        points = []
        for row, key in enumerate(keys):
            beams = beams_by_key[key]
            for idx, beam in enumerate(beams):
                beam = np.array(beam, dtype=int)
                beam_coords = self.mesh.nodes[beam]
                ax.plot(
                    beam_coords[:, 0], beam_coords[:, 1], beam_coords[:, 2],
                    color=colours[row], marker='o', markersize=1.5,
                    linestyle='-', linewidth=0.5, label=key if idx == 0 else None
                )
                points.extend(beam_coords)
                if label:
                    center = np.mean(beam_coords, axis=0)
                    ax.text(center[0], center[1], center[2], f"b:{idx}", color=colours[row], fontsize=6)

        points = np.array(points)
        ax.yaxis.set_ticks(np.arange(np.min(points[:,1]), np.max(points[:,1])+0.01, 0.2))
        ax.zaxis.set_ticks(np.arange(-0.1, 0.1 +0.01, 0.1))

        # decorations
        ax.set_xlabel("Chord, X", fontsize=9)
        ax.set_ylabel("Span, Y", fontsize=9)
        ax.set_zlabel("Thickness, Z", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.xaxis.labelpad = 25

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',fontsize=9, bbox_to_anchor=(1, 1), ncol=len(labels), frameon=False)

        ax.set_box_aspect((np.ptp(points[:,0]),np.ptp(points[:,1]),np.ptp(points[:,2]),), zoom=1.1) 
        plt.subplots_adjust(top = 1, bottom = 0, right = 1.15, left = 0, hspace = 0, wspace = 0)

        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_triangles(self, filename: str, triangle_set_keys: list[str] = [], label: bool = False):
        """
        Plots 3D triangles from mesh.triangles or mesh.triangle_sets. Optionally, only specified triangle_set_keys are plotted.

        Parameters:
            filename (str): Output filename (without extension).
            triangle_set_keys (list[str], optional): List of triangle set keys to plot. If empty, plot all triangles.
            label (bool): Whether to label the triangles with their index.
        """
        fig = plt.figure(figsize=(7.5, 4))
        ax = fig.add_subplot(111, projection='3d')
        # Determine which triangles to plot
        if triangle_set_keys and all(k in self.mesh.triangle_sets for k in triangle_set_keys):
            keys = triangle_set_keys
            triangles_by_key = {k: self.mesh.triangles[self.mesh.triangle_sets[k]] for k in keys}
        else:
            keys = ["all"]
            triangles_by_key = {"all": self.mesh.triangles}
        num_groups = len(keys)
        cmap = plt.get_cmap('rainbow', num_groups)
        colours = [cmap(i) for i in range(num_groups)]

        points = []
        for row, key in enumerate(keys):
            triangles = triangles_by_key[key]
            for idx, triangle in enumerate(triangles):
                pts = self.mesh.nodes[triangle]
                ax.plot_trisurf(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    color=colours[row], alpha=0.3, edgecolor='k', linewidth=0.5, label=key if idx == 0 else None
                )
                points.extend(pts)
                if label:
                    centroid = np.mean(pts, axis=0)
                    ax.text(centroid[0], centroid[1], centroid[2], f"t:{idx}", color=colours[row], fontsize=6)

        points = np.array(points)
        ax.yaxis.set_ticks(np.arange(np.min(points[:,1]), np.max(points[:,1])+0.01, 0.2))
        ax.zaxis.set_ticks(np.arange(-0.1, 0.1 +0.01, 0.1))

        # decorations
        ax.set_xlabel("Chord, X", fontsize=9)
        ax.set_ylabel("Span, Y", fontsize=9)
        ax.set_zlabel("Thickness, Z", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.xaxis.labelpad = 25

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',fontsize=9, bbox_to_anchor=(1, 1), ncol=len(labels), frameon=False)

        ax.set_box_aspect((np.ptp(points[:,0]),np.ptp(points[:,1]),np.ptp(points[:,2]),), zoom=1.1) 
        plt.subplots_adjust(top = 1, bottom = 0, right = 1.15, left = 0, hspace = 0, wspace = 0)

        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=300)
        plt.close(fig)

class Lattice:
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
                        t1 = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                        t2 = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

                        # check the parameter range to verify if the point is with the segment
                        if all([t1>=0.0, t1<=1.0, t2>=0.0, t2<=1.0]):
                            # found intersection
                            return np.array([x1+t1*(x2-x1), y1+t1*(y2-y1)])
                        
                        # just for debugging
                        # print(t1, t2, (x1,y1), (x2,y2), (x3,y3), (x4,y4))

                    except:
                        #pass
                        traceback.print_exc()


            # no intersection
            return np.array([np.nan, np.nan])

    @staticmethod
    def func_wrapper(func, num, *args):
        """
        Wrapper function to unpack arguments for multiprocessing.
        """
        if not callable(func):
            raise ValueError("func must be a callable function")

        return np.array((*args[:num], func(*args[num:])), dtype=object)

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
        # Use multiprocessing to parallelize intersection calculations
        pool = mp.Pool(processes=min(mp.cpu_count(), 14))
        results = []
        for index_1, line_1 in enumerate(lines_1) :
            for index_2, line_2 in enumerate(lines_2):
                args = (line_1, line_2) # Unpack the first element (index) from the tuple
                results.append(pool.apply_async(Lattice.func_wrapper, (Lattice.intersection_point, 2, index_1, index_2, *args))) # Map tasks in parallel
        pool.close()
        pool.join() 

        # Process results
        results = np.vstack([result.get() for result in results], dtype=object)
        new_lines_1 = []
        for i in range(len(lines_1)):
            line = results[results[:, 0] == i] # Get all results for the current line
            line = line[np.argsort(line[:, 1])] # Sort by the second column (index of line_2)
            line = np.asarray(line[:,2].tolist(), dtype=float) # Convert to float array
            line = line[~np.all(np.isnan(line), axis=1), :] # Remove rows where all elements are NaN
            if line.shape[0] > 0:
                new_lines_1.append(line) 

        return new_lines_1

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
                    point = Lattice.intersection_point(line[index:index+2, :], b_line)
                    if point is not None and not np.any(np.isnan(point)):
                        # Store both the intersection point and its position in the original line
                        intersection_points.append((point, index))
            
            # Sort intersection points by their position in the original line
            intersection_points.sort(key=lambda x: x[1])
            
            # If we have too many intersections, the line might be too close to the boundary and cause numerical issues, so we skip it
            intersection_threshold = 20
            if len(intersection_points) >= intersection_threshold:
                print(f"WARNING: Line {i} is removed due to close proximity to the border with {len(intersection_points)} intersections")
                continue
                
            # Process line segments between intersections
            if len(intersection_points) >= 2:
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
    def exert_boundary(lines_1, lines_2, boundary):
        """
        Definiing boundary and trimming lines at the boundary
        """
        # boundary definition
        boundary = np.array(np.concatenate(tuple(line for line in boundary), axis=0), dtype=np.float32)
        limits = np.array([np.min(boundary, axis=0), np.max(boundary, axis=0)])
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
        boundary = np.concatenate((boundary_corners, edge_nodes))
        # edges
        tolerance=0.0005
        b = boundary[np.abs(boundary[:,1]-limits[0,1])<tolerance,:]
        t = boundary[np.abs(boundary[:,1]-limits[1,1])<tolerance,:]
        l = boundary[np.abs(boundary[:,0]-limits[0,0])<tolerance,:]
        r = boundary[np.abs(boundary[:,0]-limits[1,0])<tolerance,:]

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

    @staticmethod
    def bisect_lines(lines: np.ndarray) -> np.ndarray:
        """
        Bisects each pair of consecutive lines by computing their midpoints and interleaving them with the original lines.
        Parameters
        ----------
        lines : np.ndarray
            An array of shape (N, M) representing N lines, each with M coordinates.
        Returns
        -------
        np.ndarray
            An array of shape (2*N - 1, M) containing the original lines and their midpoints interleaved.
        """

        midpoints = lines[:-1] + np.diff(lines, axis=0) / 2
        new_lines = np.empty((lines.shape[0]+midpoints.shape[0], lines.shape[1]), dtype=lines.dtype)
        new_lines[0::2] = lines
        new_lines[1::2] = midpoints
        return new_lines

    @staticmethod
    def generate_lattice(line_data, chevron_angle_rad):

        # re-sampling the lines
        chordwise = Lattice.resample_lines(line_data["lines_1"], line_data["lines_2"]) # chordwise
        spanwise = Lattice.resample_lines(line_data["lines_2"], line_data["lines_1"]) # spanwise
        print("Lattice: Resampled")

        # introducing chevrons - NOTE: check the line being used for chevrons
        Tan60 = np.tan(chevron_angle_rad) # unit cell chevron angle
        for line_i, line in enumerate(spanwise):
            chevron_line = []
            for seg_i in range(line.shape[0]-1):
                if seg_i==0: chevron_line.append(line[seg_i]) # additing initial point
                v1 = np.diff(line[seg_i:seg_i+2], axis=0).ravel()
                mag_v1 = np.linalg.norm(v1)
                norm_v1 = v1/mag_v1 # segment direction
                norm_v2 = np.array([-norm_v1[1], norm_v1[0]]) # pendicular direction
                mag_v2 = Tan60 * (mag_v1/2) 
                point = line[seg_i] + v1/2 + (mag_v2*norm_v2) #*(-1)**line_i
                chevron_line.extend([point, line[seg_i+1]])
            # adding chevron line to the spanwise
            chevron_line = Lattice.bisect_lines(np.array(chevron_line))
            spanwise[line_i] = chevron_line
        print("Lattice: Chevrons added")

        # exerting boundary
        boundary = line_data["boundary"]
        spanwise, chordwise, ribs, trailing_edge = Lattice.exert_boundary(spanwise, chordwise, boundary)
        print("Lattice: Trimmed to boundary")

        # data collection
        flat_line_data={
            "Chevrons": spanwise,
            "Stringers": chordwise,
            "Ribs": ribs,
            "Trailing_Edge": trailing_edge,
        }

        return flat_line_data


if __name__ == "__main__":


    # Tolerance for merging nodes
    tolerance = 1e-4  

    # ARGUMENTS
    CASE_NAME = "Test_v3" # sys.argv[-3]
    FILE_ID = "0" # sys.argv[-2]
    ITERATION_ID = "1" # sys.argv[-1]
	
    # PATHS
    CODE_DIR = os.path.abspath(os.getcwd())
    RUN_DIR = os.path.join(CODE_DIR, "run_dir")
    CASE_DIR = os.path.join(CODE_DIR, CASE_NAME)

    # read inputs
    FR_INPUTS = Utils.load_object(os.path.join(CASE_DIR, "input", '%s_FR_INPUTS'%(str(FILE_ID))),'json')
    UC_INPUTS = Utils.load_object(os.path.join(CASE_DIR, "input", '%s_UC_INPUTS'%(str(FILE_ID))),'json')


    # Initialising attice
    lattice = Panel2D( 
        chevron_length=UC_INPUTS["CHEVRON_WALL_LENGTH"],
        chevron_angle_rad=UC_INPUTS["THETA_RAD"],
        chevron_spacing=UC_INPUTS["CHEVERON_SEPARATION"],
        mesh=Mesh(dim=2), 
    )
    print("Lattice: Initialised with chevron length %.2f mm, angle %.2f deg, spacing %.2f mm" % (
        Utils.Units.m2mm(lattice.chevron_length), Utils.Units.rad2deg(lattice.chevron_angle_rad), Utils.Units.m2mm(lattice.chevron_spacing)))

    # getting lattice nodes
    line_data = Utils.load_object(os.path.join(CASE_DIR,"data","distribution","%s_%s_trace"%(str(FILE_ID), str(ITERATION_ID))), 'pickle') # reload
    flat_line_data = Lattice.generate_lattice(line_data, lattice.chevron_angle_rad)

    # merging the two trailing egdes
    trailing_edge_y = np.sort(np.concatenate(tuple(edge[:,1].astype(np.float32) for edge in flat_line_data["Trailing_Edge"]), axis=0, dtype=np.float32))
    for i in range(flat_line_data["Trailing_Edge"].shape[0]):
        # merging the trailing edge
        flat_line_data["Trailing_Edge"][i] = np.column_stack((
            np.ones(trailing_edge_y.shape)*flat_line_data["Trailing_Edge"][i][0,0],
            trailing_edge_y
        ))

    # Adding lines to the mesh, grouped by keys
    for key in flat_line_data.keys():
        print("Adding lines for key:", key)
        new_lines = flat_line_data[key]
        lattice.mesh.create_grouped_lines_nodes(key, new_lines, tolerance=tolerance)
    print("Lattice: Lines added to the mesh")

    # Plot
    lattice.mesh.plot_groups_2D(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_lattice"),
        group_keys=[],
        label=True
    )

    # Beams plot
    lattice.plot_beams(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_lattice_beams"),
        beam_set_keys=[],   
        label=True
    )

    # Adding triangles
    lattice.mesh.triangulate_2D(np.arange(lattice.mesh.nodes.shape[0]))
    print("Lattice: Triangles added to the mesh")

    # Triangles plot
    lattice.plot_triangles(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_lattice_triangles"),
        triangle_set_keys=[],
        label=True
    )   

    ## Flat grid to wing profile mapping
    # loading grids
    grids = Utils.load_object(os.path.join(CASE_DIR,"inp","distribution",str(FILE_ID)+"_grids"), 'pickle') # reload

    # RBF Intepolation - Very Robust
    coordinate_fun = interpolate.RBFInterpolator(
        grids["flat_node_grid"][:,:,1:].reshape((-1,2)).astype(np.float32), 
        grids["node_grid"][:,:,1:].reshape((-1,3)).astype(np.float32), 
        neighbors=4, kernel='linear'
        )

    # Mapping
    wing = Wing3D()
    wing.mapping_2D_to_3D(coordinate_fun, lattice.mesh)
    print("Wing: Lattice 2D to 3D mapping completed")

    wing.plot_nodes(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_wing_nodes"),
        nset_keys=[],
        label=False
    )
    wing.plot_beams(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_wing_beams"),
        beam_set_keys=[],
        label=False
    )
    wing.plot_triangles(
        filename=os.path.join(CASE_DIR, "fig", f"{FILE_ID}_{ITERATION_ID}_wing_triangles"),
        triangle_set_keys=[],
        label=False
    )

    # check mesh
    degenerates = wing.mesh.check_mesh_integrity()
    if degenerates['unused_nodes'].size > 0:
        print("Unused nodes:")
        print(wing.mesh.nodes[degenerates['unused_nodes']])
    if degenerates['degenerate_beams'].size > 0:
        wing.mesh.remove_beams(degenerates['degenerate_beams'])
    if degenerates['degenerate_triangles'].size > 0:
        print(wing.mesh.triangles[degenerates['degenerate_triangles']])
    if degenerates['missing_beam_edges_in_triangles'].size > 0:
        print(f"WARNING: {degenerates['missing_beam_edges_in_triangles'].size} beam edges missing in triangles:")
    degenerates = wing.mesh.check_mesh_integrity()
    if degenerates['unused_nodes'].size > 0 and degenerates['degenerate_beams'].size > 0 and degenerates['degenerate_triangles'].size > 0 and degenerates['missing_beam_edges_in_triangles'].size > 0:
        raise ValueError("Mesh integrity check failed, degenerates found.")

    # check repeated sets
    beam_sets = ["Trailing_Edge", "Ribs","Chevrons","Stringers"]
    for i, key_i in enumerate(beam_sets):
        for j, key_j in enumerate(beam_sets[i+1:]):
            repeat_index = np.isin(wing.mesh.beam_sets[key_j], wing.mesh.beam_sets[key_i])
            if np.any(repeat_index):
                print(f"WARNING: Beams of '{key_i}' repeated in '{key_j}' are removed from '{key_j}'.")
                wing.mesh.beam_sets[key_j] = wing.mesh.beam_sets[key_j][~repeat_index]

    # Creating anchor nodes
    NODES = Utils.load_object(os.path.join(CASE_DIR, 'inp', "distribution", str(FILE_ID) +'_FR_NODES'),"pickle")
    mid_rib_index = len(NODES["BEAM"])//2 # index, not the midrib number
    anchor_node = NODES["BEAM"][mid_rib_index]
    wing.mesh.nodes = np.vstack((wing.mesh.nodes, anchor_node, anchor_node)) # adding anchor node

    # creating required nsets
    ribs_node_index = np.unique(wing.mesh.beams[wing.mesh.beam_sets["Ribs"]].reshape((-1)))
    min_x = np.min(wing.mesh.nodes[ribs_node_index, 0])
    min_y, max_y = np.min(wing.mesh.nodes[ribs_node_index, 1]), np.max(wing.mesh.nodes[ribs_node_index, 1])
    wing.mesh.nsets = {
        "NODE":np.arange(wing.mesh.nodes.shape[0]),
        "RIB_1": ribs_node_index[np.argwhere(np.abs(wing.mesh.nodes[ribs_node_index, 1]-min_y)<0.005).squeeze()],
        "RIB_2": ribs_node_index[np.argwhere(np.abs(wing.mesh.nodes[ribs_node_index, 1]-max_y)<0.005).squeeze()],
        "ANODE_1": np.array([wing.mesh.nodes.shape[0]-2]),  # first anchor node
        "ANODE_2": np.array([wing.mesh.nodes.shape[0]-1]),  # second anchor node
        "TE": np.unique(wing.mesh.beams[wing.mesh.beam_sets["Trailing_Edge"]].reshape((-1))),
    }

    # updating the elset names
    wing.mesh.beam_sets = {
        "BEAM":np.arange(wing.mesh.beams.shape[0]),
        "RIBS": wing.mesh.beam_sets["Ribs"],
        "TE": wing.mesh.beam_sets["Trailing_Edge"],
        "CHEVRONS": wing.mesh.beam_sets["Chevrons"],
        "STRINGERS": wing.mesh.beam_sets["Stringers"]
    }

    # Save the mesh
    wing.mesh.write_mesh(
        filename=os.path.join(CASE_DIR, "inp", f"{FILE_ID}_{ITERATION_ID}_FR"),
        FR_INPUTS=FR_INPUTS,
        UC_INPUTS=UC_INPUTS,
    )