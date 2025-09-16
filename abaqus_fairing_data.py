# ABAQUS SCIPTING MODULES
from abaqusConstants import *

# ABAQUS ODB MODULES
from odbAccess import *

# OTHER PYTHON PACKAGES
import os
import sys
import traceback
import numpy as np
import pickle
import json

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


class FairingData:

    def __init__(self, case_folder, case_number, hinge_node={}, surface_nodes_U={}, shell_equivalent_SE={}, shell_equivalent_SK={}):
        self.case_folder = case_folder
        self.case_number = case_number
        self.hinge_node = hinge_node
        self.surface_nodes_U = surface_nodes_U
        self.shell_equivalent_SE = shell_equivalent_SE
        self.shell_equivalent_SK = shell_equivalent_SK

    def get_frame_nodal_data(self, frame, HingeSet, OuterSurfaceSet, count):

        for key in ["RF", "RM", "U", "UR"]:
            field=frame.fieldOutputs[key]

            # hinge node
            if count == 0: # initilise list at first frame
                self.hinge_node[key] = np.empty((0, 3), dtype=np.float32)
            self.hinge_node[key] = np.concatenate(
                (self.hinge_node[key], np.atleast_2d(field.getSubset(region=HingeSet).values[0].data)),
                axis=0
            )

            # surface nodes
            if key == "U":
                
                try:
                    nodes=field.getSubset(region=OuterSurfaceSet).values
                except KeyError:
                    print("History output %s not found."%(key))
                    continue

                if len(nodes) > 0:       
                    for node in nodes:
                        if count == 0: # initilise list at first frame
                            self.surface_nodes_U[node.nodeLabel] = np.empty((0, 3), dtype=np.float32)
                        self.surface_nodes_U[node.nodeLabel] = np.concatenate((
                            self.surface_nodes_U[node.nodeLabel],
                            np.atleast_2d(node.data),
                            ), axis=0
                        )

    def get_frame_element_data(self, frame, ShellSet, count):
        for key in ['SE','SK']:
            # initialise dictoionary for field 
            try:
                field=frame.fieldOutputs[key].getSubset(region=ShellSet, position=CENTROID).values
            except KeyError:
                print("Field output %s not found."%(key))
                continue

            # component index
            if key in 'SE': 
                component_index = np.array([0,1,3])
                for element in field:
                    if count == 0: # initilise list at first frame: # initilise list at first frame
                        self.shell_equivalent_SE[element.elementLabel] = np.empty((0,3))
                    self.shell_equivalent_SE[element.elementLabel] = np.concatenate((
                        self.shell_equivalent_SE[element.elementLabel], 
                        np.atleast_2d(element.data[np.ix_(component_index)])
                        ),axis=0
                    )

            elif key in 'SK': 
                component_index = np.array([1,0,2])
                for element in field:
                    if count == 0: # initilise list at first frame: # initilise list at first frame
                        self.shell_equivalent_SK[element.elementLabel] = np.empty((0,3))
                    self.shell_equivalent_SK[element.elementLabel] = np.concatenate((
                        self.shell_equivalent_SK[element.elementLabel], 
                        np.atleast_2d(element.data[np.ix_(component_index)])
                        ),axis=0
                    )
            else:
                print("ERROR: Unknown key %s"%(key))

    def extract(self):
        # odb file
        try:
            odb_file = os.path.join(self.case_folder, "odb",  "%i_FR.odb"%(self.case_number))
            print(odb_file)
            odb=openOdb(odb_file, readOnly=False)
        except:
            print("ERROR: Unable to open ODB file")
            return

        # instance
        Instance = odb.rootAssembly.instances['PART-1-1']

        # Sets
        HingeSet = Instance.nodeSets["PIVOT"]
        try:
            ShellSet = Instance.elementSets["SHELL_EQUIVALENT"]  # Only in equivalent shell model
        except:
            ShellSet = None
        try:
            OuterSurfaceSet = Instance.nodeSets["OUTER_SURFACE"]  # Only if it exist (Shell Model)
        except:
            print("ERROR: OuterSurfaceSet not found")

        # Collect frame data
        count = 0
        for step in odb.steps.values():
            for frame in step.frames:
                self.get_frame_nodal_data(frame, HingeSet, OuterSurfaceSet, count)
                self.get_frame_element_data(frame, ShellSet, count)
                count += 1

        # close
        odb.close()

if __name__ == '__main__':

    # ARGUMENTS
    case_folder = sys.argv[-2]
    case_number = int(sys.argv[-1])

    # Derived path
    code_folder = os.path.dirname(case_folder)

    # Re-directing Abaqus output
    status_file=os.path.join(case_folder, "log", "%i_FR_LOG_Abaqus.txt"%(case_number))
    original_stdout = sys.stdout  # Save a reference to the original standard output
    f = open(status_file, "w")  # a to append
    sys.stdout = f  # Change the standard output to the file we created.

    # Extract data
    try:
        case_data = FairingData(case_folder, case_number)
        case_data.extract()
    except:
        traceback.print_exc()
        # Closing Abaqus dump file
        sys.stdout = original_stdout  # Reset the standard output to its original value
        f.close()
        # Print error message
        print("ERROR: Extraction failed. Check log file for details.")

    # Save data
    save_object(case_data, os.path.join(case_folder, "data", "%i_fairing_data"%(case_number)), method="pickle")

    # Closing Abaqus dump file
    if not f.closed:
        print("Closing log file")
        sys.stdout = original_stdout  # Reset the standard output to its original value
        f.close()
        
