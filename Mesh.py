# Mesh.py
import traceback
import Utils
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class Set:
    def __init__(self, name: str, items: np.ndarray) -> None:
        """
        A class to manage sets of elements (nodes, beams, triangles).
        Parameters:
            name (str): Name of the set.
            items (np.ndarray): Array of element indices.
        """
        if not isinstance(items, np.ndarray) or items.ndim != 1 or items.shape[0] < 1 or items.dtype != int:
            raise ValueError("elements must be a 1D numpy array with shape (n,) where n >= 1 and dtype is int")

        self.__name = name
        self.__items = items

    def __call__(self):
        return {self.__name: self.__items}

    # Getters, no setters to prevent direct modification
    @property
    def name(self) -> str:
        return self.__name
    @property
    def items(self) -> np.ndarray:  
        return self.__items

    def add_items(self, new_items: np.ndarray) -> None:
        """
        Adds new items to the set, ensuring uniqueness.
        Parameters:
            new_items (np.ndarray): Array of new element indices to add.
        """
        if not isinstance(new_items, np.ndarray) or new_items.ndim != 1:
            raise ValueError("new_items must be a 1D numpy array")

        self.__items = np.unique(np.r_[self.__items, new_items])

    def remove_items(self, remove_items: np.ndarray) -> None:
        """
        Removes items from the set.
        Parameters:
            remove_items (np.ndarray): Array of element indices to remove.
        """
        if not isinstance(remove_items, np.ndarray) or remove_items.ndim != 1:
            raise ValueError("remove_items must be a 1D numpy array")

        self.__items = np.setdiff1d(self.__items, remove_items)

class Nodes(Set):
    def __init__(self, dim: int = 3) -> None:
        """
        A class to manage nodes in 2D or 3D space.
        Parameters:
            dim (int): Dimension of the space (2 or 3).
        """
        if not isinstance(dim, int) or dim not in [2, 3]:
            raise ValueError("dim must be either 2 or 3")

        self.__dim = dim
        self.__nodes = np.empty((0, dim), dtype=float)
        self.__sets = []

    def __call__(self) -> np.ndarray:
        """Returns nodes array."""
        return self.__nodes

    # Getters, no setters to prevent direct modification
    @property
    def dim(self) -> int:
        return self.__dim
    @property
    def sets(self) -> dict:
        return reduce(lambda x, y: {**x, **y}, [nset() for nset in self.__sets], {})

    def add_or_merge_node(self, new_node: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
        """
        Adds a node to the nodes array, merging close nodes within a given tolerance.

        Parameters:
            new_node (np.ndarray): A 1D array of shape (dim,) representing the node to be added.
        """
        if not isinstance(new_node, np.ndarray) or new_node.ndim != 1 or new_node.shape[0] != self.dim or new_node.dtype != float:
            raise ValueError(f"new_node must be a 1D numpy array float with shape ({self.dim},). Got shape {new_node.shape} and dtype {new_node.dtype}.")

        
        # First node addition
        if self.__nodes.shape[0] == 0:
            self.__nodes = new_node[np.newaxis, :]
            return self.__nodes.shape[0] - 1

        # Check for close nodes
        dist = np.sum((self.__nodes - new_node) ** 2, axis=1)**0.5
        merge_idx = np.argwhere(dist < tolerance).ravel()
        
        if merge_idx.size == 0: # No close nodes, simply add
            self.__nodes = np.r_[self.__nodes, new_node[np.newaxis,:]]
            return self.__nodes.shape[0] - 1
        
        else: # Merge close nodes
            merge_idx = merge_idx[0]  # Take the first close node
            avg = np.mean(np.vstack([self.__nodes[merge_idx], new_node]), axis=0)
            self.__nodes[merge_idx] = avg
            return merge_idx

    def remove_node(self, remove_index: int) -> None:
        """
        Removes a node from self.nodes.
        """
        # Remove node
        self.__nodes = np.delete(self.__nodes, remove_index, axis=0)

        # Update indices in the sets
        for set in self.__sets:
            set = np.array(
                [
                    index - 1 if index > remove_index else index
                    for index in set
                    if index != remove_index
                ],
                dtype=int,
            )

    def add_or_merge_set(self, name: str, items: np.ndarray) -> None:
        """
        Adds a new set of nodes.
        Parameters:
            name (str): Name of the set.
            items (np.ndarray): Array of node indices to include in the set.
        """

        # check if set exists, if yes, merge items
        for s in self.__sets:
            if s.name == name:
                s.add_items(items)
                return
        
        # create new set
        new_set = Set(name, items)
        self.__sets.append(new_set)


    def exclusivize_set(self, priority_order: list[str]) -> None:
        """
        Exclusivize nodes in sets in the order of priority.
        Parameters:
            priority_order (list[str]): List of set names in order of priority.
        """
        # Initialize a list to keep track of higher priority items
        higher_priority_items = np.empty((0,), dtype=int)

        # Loop through sets in the order of priority
        for i, name in enumerate(priority_order):
            # check name exists
            if name != self.__sets[i].name:
                raise ValueError(f"Set with name '{name}' not found.")

            # Skip the first set as it has the highest priority
            if i == 0:
                higher_priority_items = np.r_[higher_priority_items, self.__sets[i].__items]
                continue  

            # Find the current set
            current_set_items = self.__sets[i].__items

            # Determine repeated items in the current set
            repeated_items = np.intersect1d(current_set_items, higher_priority_items)

            # Update the current set to only include exclusive items
            self.__sets.remove(repeated_items)

            # Update the list of higher priority items
            higher_priority_items = np.r_[higher_priority_items, self.__sets[i].__items]

class Elements(Set):
    def __init__(self, dim: int) -> None:
        """
        A class to manage elements (1D elements defined by two node indices).
        """
        if not isinstance(dim, int) or dim not in [2, 3]:
            raise ValueError("dim must be either 2 or 3")

        self.__dim = dim
        self.__elements = np.empty((0, self.__dim), dtype=int)
        self.__sets = []

    def __call__(self) -> np.ndarray:
        """Returns elements array."""
        return self.__elements

    # Getters, no setters to prevent direct modification
    @property
    def dim(self) -> int:
        return self.__dim
    @property
    def sets(self) -> dict:
        return reduce(lambda x, y: {**x, **y}, [elset() for elset in self.__sets], {})

    def add_element(self, new_element: np.ndarray) -> None:
        """
        Adds an element to the elements array.

        Parameters:
            new_element (np.ndarray): A 1D array of shape (2,) representing the element to be added.

        Returns:
            int: Index of the added or existing element.
        """
        if not isinstance(new_element, np.ndarray) or new_element.ndim != 1 or new_element.shape[0] != self.dim or new_element.dtype != int:
            raise ValueError(f"new_element must be a 1D numpy array with shape ({self.__dim},)")

        # Check if element already exists (regardless of node order)
        bool_exists = False
        if self.__elements.shape[0] > 0:
            sorted_existing = np.sort(self.__elements, axis=1)
            sorted_new = np.sort(new_element)
            bool_exists = np.all(sorted_existing == sorted_new, axis=1)

        # If element exists, merge and return index
        if np.any(bool_exists):
            return np.argwhere(bool_exists).ravel()[0]
        # Else add new element
        else:
            self.__elements = np.r_[self.__elements, new_element[np.newaxis, :]]
            return self.__elements.shape[0] - 1
        
    def remove_element(self, remove_index: int) -> None:
        """
        Removes an element from self.elements.
        """
        # Remove element
        self.__elements = np.delete(self.__elements, remove_index, axis=0)

        # Update indices in the sets
        for set in self.__sets:
            set = np.array(
                [
                    index - 1 if index > remove_index else index
                    for index in set
                    if index != remove_index
                ],
                dtype=int,
            )

    def add_or_merge_set(self, name: str, items: np.ndarray) -> None:
        """
        Adds a new set of elements.
        Parameters:
            name (str): Name of the set.
            items (np.ndarray): Array of element indices to include in the set.
        """

        # check if set exists, if yes, merge items
        for s in self.__sets:
            if s.name == name:
                s.add_items(items)
                return
        
        # create new set
        new_set = Set(name, items)
        self.__sets.append(new_set)

class Mesh (Nodes, Elements):
    """
    A class to manage a mesh consisting of nodes, beams, triangles, and associated sets.
    """
    def __init__(self, node_dim: int = 3, element_dim: list[int] = [2]) -> None:
        self.__nodes = Nodes(node_dim) # 2D or 3D nodes
        self.__elements = [Elements(dim) for dim in element_dim]

    # Getters, no setters to prevent direct modification
    @property
    def nodes(self) -> np.ndarray:
        return self.__nodes()
    @property
    def elements(self) -> list[np.ndarray]:
        return [el() for el in self.__elements]
    
    # Access sets
    def nsets(self):
        return self.__nodes.sets
    def elsets(self):
        """ 
        For each instance of Elements in the Mesh, return its Sets as a list of dictionaries."""
        return [el.sets for el in self.__elements]

    def add_nodes(self, new_nodes: np.ndarray, nset_name: str|None = None, tolerance: float = 1e-4) -> np.ndarray:
        """
        Adds multiple nodes to the nodes array, merging close nodes within a given tolerance, and creates a set for the newly added nodes.

        Parameters:
            new_nodes (np.ndarray): A 2D array of shape (n, dim) representing the nodes to be added.
            tolerance (float): Distance threshold for merging nodes.
        """

        # add nodes
        incorporated_indices = []
        for node in new_nodes:
            idx = self.__nodes.add_or_merge_node(node.astype(float), tolerance)
            incorporated_indices.append(idx)

        # filter unique indices only
        incorporated_indices = np.array(incorporated_indices, dtype=int)

        # create set
        if nset_name is not None:
            self.__nodes.add_or_merge_set(nset_name, Utils.unique_1D(incorporated_indices))

        return incorporated_indices

    def add_nset(self, name: str, items: np.ndarray) -> None:
        """
        Adds a new set of nodes.
        Parameters:
            name (str): Name of the set.
            items (np.ndarray): Array of node indices to include in the set.
        """
        self.__nodes.add_or_merge_set(name, items)

    def add_elements(self, el_list_idx: int, new_elements: np.ndarray, elset_name: str|None = None) -> np.ndarray:
        """
        Adds multiple elements to the elements array, merging close elements within a given tolerance, and creates a set for the newly added elements.

        Parameters:
            new_elements (np.ndarray): A 2D array of shape (n, m) representing the elements to be added.
            tolerance (float): Distance threshold for merging elements.
        """
        if not isinstance(el_list_idx, int) or el_list_idx < 0 or el_list_idx >= len(self.__elements):
            raise ValueError(f"index must be an integer between 0 and {len(self.__elements)-1}. Got {el_list_idx}.")

        # add elements
        incorporated_indices = []
        for element in new_elements:
            idx = self.__elements[el_list_idx].add_element(element)
            incorporated_indices.append(idx)

        # filter unique indices only
        incorporated_indices = np.array(incorporated_indices, dtype=int)

        # create set
        if elset_name is not None:
            self.__elements[el_list_idx].add_or_merge_set(elset_name, Utils.unique_1D(incorporated_indices))

        return incorporated_indices

    def triangulate(self, el_list_idx: int, node_indexes: np.ndarray, nsets_to_remove:dict|None = None) -> None:
        """
        Performs Delaunay triangulation on a subset of nodes specified by their indexes
        """
        if not isinstance(node_indexes, np.ndarray) or node_indexes.ndim != 1 or node_indexes.shape[0] < 3 or node_indexes.dtype != int:
            raise ValueError(f"node_indexes must be a 1D numpy array of integers with at least 3 elements. Got shape {node_indexes.shape}.")

        if self.__nodes.dim != 2:
            raise ValueError(f"Triangulation is only supported in 2D. Nodes have dim={self.__nodes.dim}.")

        # Perform Delaunay triangulation
        tri = Delaunay(self.nodes[node_indexes])  # Triangulate using specified positions
        triangles = np.asarray([node_indexes[sub_indexes] for sub_indexes in tri.simplices], dtype=int)

        # Remove triangles where all nodes are in the same nset
        if nsets_to_remove is not None:
            to_remove = []
            for nset_name in nsets_to_remove.keys():
                nset_nodes = nsets_to_remove[nset_name]
                # Check if all nodes of each triangle are in the current nset
                # isin returns boolean array, all with axis=1 checks if all nodes in triangle are in nset
                mask = np.all(np.isin(triangles, nset_nodes), axis=1)
                if np.any(mask):
                    print(f"\t WARNING: Removing {np.sum(mask)} triangles with all nodes in nset '{nset_name}'")
                to_remove.extend(np.where(mask)[0])
            
            # Remove duplicates and convert to array, then filter triangles
            if to_remove:
                to_remove = np.unique(to_remove)
                triangles = np.delete(triangles, to_remove, axis=0)

        # Add triangles to the mesh
        self.add_elements(el_list_idx, triangles, elset_name="triangles")

    def add_lines_to_mesh(self, el_list_idx: int, lines, set_name=None, tolerance: float = 1e-4) -> None:
        """
        Adds polylines to the mesh as connected elements, creating nodes as necessary.

        Parameters:
            lines (list): A list of polylines, each represented as a 2D numpy array of shape (n, dim).
            set_name (str|None): Name of the set to which the nodes and elements will be added. If None, no set is created.
        """

        if not isinstance(lines, (list, np.ndarray)) or len(lines) == 0 or not all(isinstance(line, np.ndarray) and line.ndim == 2 and line.shape[1] == self.__nodes.dim for line in lines):
            raise ValueError(f"lines must be a list of 2D numpy arrays with shape (n, {self.__nodes.dim})")

        for line in lines:
            indices = self.add_nodes(line, set_name, tolerance)
            unique_indices = Utils.unique_1D(indices)
            elements = np.c_[unique_indices[:-1],unique_indices[1:]]
            self.add_elements(el_list_idx, elements, set_name)
    
    # TODO: check mesh integrity (e.g. no isolated nodes, all elements refer to valid nodes, etc.)

    def plot_2D(self, el_list_idx, bool_nodes=False, bool_elements=True, label: bool = False, save_path=None, show=False):
        """
        Plots the 2D mesh with options to plot groups, nodes, beams and triangles.
        """
        def nodes():
            """
            2D scatter plot of nodes.
            """
            nsets = self.nsets()
            colours = plt.cm.rainbow(np.linspace(0,1,len(nsets)))
            for i, ax in enumerate(axes):
                for (name, nset), colour in zip(nsets.items(), colours):
                    nodes = self.nodes[nset]
                    ax.scatter(
                        nodes[:, 0], nodes[:, 1], color=colour, s=5, marker='.', label=name
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
                            ax.text(x, y, f"n:{idx}", color=colour,
                                    ha='center', va='bottom', fontsize=2)

        def elements(el_list_idx=0):
            """
            2D line plot of elements.
            """
            dim = self.__elements[el_list_idx].dim
            elsets = self.__elements[el_list_idx].sets
            colours = plt.cm.rainbow(np.linspace(0,1,len(elsets)))
            for i, ax in enumerate(axes):
                for (name, elset), colour in zip(elsets.items(), colours):
                    for count, index in enumerate(elset):
                        nodes = self.__elements[el_list_idx]()[index]
                        if dim == 2:
                            ax.plot(
                                self.nodes[nodes, 0],
                                self.nodes[nodes, 1],
                                color=colour,
                                marker='.', markersize=0.5,
                                linestyle='-', linewidth=0.5,
                                label=name if count == 0 else None,
                            )
                        elif dim == 3:
                            ax.add_patch(
                                plt.Polygon(
                                    self.nodes[nodes],
                                    closed=True,
                                    facecolor=colour,
                                    alpha=0.2,
                                    edgecolor="k",
                                    linewidth=0.5,
                                    label=name if count == 0 else None,
                                )
                            )
                        if label:
                            # LABEL ELEMENT (at midpoint inside boundary)
                            boundary_x_min, boundary_x_max = x_top_lim if i == 0 else x_bot_lim
                            boundary_y_min, boundary_y_max = y_lim
                            inside_mask = (
                                (nodes[:, 0] >= boundary_x_min) & (nodes[:, 0] <= boundary_x_max) &
                                (nodes[:, 1] >= boundary_y_min) & (nodes[:, 1] <= boundary_y_max)
                            )
                            if np.any(inside_mask):
                                inside_coords = nodes[inside_mask]
                                center = np.mean(inside_coords, axis=0)
                                label_x, label_y = center
                                ax.text(label_x, label_y, f"e:{index}", color=colour,
                                        ha='center', va='top', fontsize=2)

        fig, axes = plt.subplots(2)

        # call plotting function
        if bool_nodes:
            nodes()
        if bool_elements:
            elements(el_list_idx)

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

    def plot_3D(self, el_list_idx, bool_nodes=False, bool_elements=True, label: bool = False, save_path=None, show=False):
        """
        Plots the 2D mesh with options to plot groups, nodes, beams and triangles.
        """
        def nodes():
            """
            2D scatter plot of nodes.
            """
            nsets = self.nsets()
            colours = plt.cm.rainbow(np.linspace(0,1,len(nsets)))
            for (name, nset), colour in zip(nsets.items(), colours):
                nodes = self.nodes[nset]
                ax.scatter(
                    xs=nodes[:, 0], ys=nodes[:, 1], zs=nodes[:, 2], color=colour, s=2, marker='.', label=name
                )
                if label:
                    for idx, (x, y, z) in enumerate(nodes):
                        ax.text(x, y, z, f"n:{idx}", color=colour,
                                ha='center', va='bottom', fontsize=2)

        def elements(el_list_idx=0):
            """
            2D line plot of elements.
            """
            dim = self.__elements[el_list_idx].dim
            elsets = self.__elements[el_list_idx].sets
            colours = plt.cm.rainbow(np.linspace(0,1,len(elsets)))
            for (name, elset), colour in zip(elsets.items(), colours):
                
                # get elements
                elements = self.__elements[el_list_idx]()[elset]
                
                if dim == 2: # plot each element individually to avoid connecting lines
                    for count, element in enumerate(elements): 
                        ax.plot(
                            *(self.nodes[element, j] for j in range(dim)),
                            color=colour,
                            marker='.', markersize=0.5,
                            linestyle='-', linewidth=0.5,
                            label=name if count == 0 else None,
                        )
                elif dim == 3: # plot all triangles in one go
                    ax.plot_trisurf( 
                        *(self.nodes[:, j] for j in range(dim)),
                        triangles=elements,
                        facecolor=colour,
                        alpha=0.2,
                        edgecolor="k",
                        linewidth=0.5,
                        label=name,
                    )

                if label:
                    for count, (index, element) in enumerate(zip(elset, elements)):
                        center = np.mean(self.nodes[element], axis=0)
                        ax.text(center[0], center[1], center[2], f"e:{index}", color=colour,
                                ha='center', va='top', fontsize=2)


        fig = plt.figure(figsize=(7.5,4))
        ax = fig.add_subplot(111, projection='3d')

        # call plotting function
        if bool_nodes:
            nodes()
        if bool_elements:
            elements(el_list_idx)

        # decorations
        ax.set_xlabel("X [m]", fontsize=9)
        ax.set_ylabel("Y [m]", fontsize=9)
        ax.set_zlabel("Z [m]", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.xaxis.labelpad = 25

        # Legends
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',fontsize=9, bbox_to_anchor=(1, 1), ncol=len(labels), frameon=False)

        # Set aspect ratio and zoom
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
