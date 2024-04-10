import numpy as np
import json

# Human3.6M
JOINT_LIST = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot", "Spine", "Thorax",
    "Neck", "Head", "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist"
]

class DictTree:
    """
    A efficient tree structure that discribes nodes in dictionaries. Use the 
    capital word NODE for nodes in this tree. All NODEs are dictionaries with
    three common keys - "name": name of the node; "index": the index of the node
    in node_list; "parent" the index of the node's parent.
    """
    def __init__(self, size, root):
        """
        size: <int> the number of nodes.
        root: <dict> the root NODE.
        """
        self.root = root
        self.size = size

        # the list of NODEs
        self.node_list = [{}]*size
        self.node_list[root["index"]] = root

        # the list of distal joint indices of bones.
        self.left_bones = []
        self.right_bones = []
        self.middle_bones = []

    def create_node(self, name, idx, parent):
        """
        Create a NODE and add it to the node_list.
        name, idx, parent respectively corespond to the "name", "index",
        "parent" keys.
        """
        node = {"name":name, "index":idx, "parent":parent}
        assert self.node_list[node["index"]] == {}, "Two nodes shares one index"
        self.node_list[node["index"]] = node

    def get_conv_mat(self):
        """
        Get the conversion matrix and its inverse.
        """
        conv_mat = np.zeros((self.size*3, self.size*3))
        for i in range(self.size):
            if i == self.root["index"]:
                conv_mat[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
            else:
                p = self.node_list[i]["parent"]
                conv_mat[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
                conv_mat[3*i:3*i+3, 3*p:3*p+3] = -np.eye(3)

        self.conv_J2B = conv_mat
        self.conv_B2J = np.linalg.inv(conv_mat)

    def get_bl_mat(self, poses3D):
        """
        :pose3D: <numpy.ndarray> of n_frames x n_joints x 3, the 3D joint coordinates.
        :return: <numpy.ndarray> of n_frames x n_bones, the 3D bone length vector
        """
        n_frames = poses3D.shape[0]
        bls = poses3D.reshape(n_frames, -1) @ self.conv_J2B.T
        if self.root['index'] == 0:
            bl = bls[:, 3:]
        else:
            bl = np.concatenate((bls[:, :self.root['index']*3], bls[:, :(self.root['index']+1)*3]), axis=1)
        return np.linalg.norm(bl.reshape(n_frames, -1, 3), axis=2).reshape(n_frames, -1)

    def draw_skeleton(self, ax, pts, joint_color="#2E62A6", bone_color="auto", joint_size=4, bone_width=2):
        """
        Draw human skeleton.
        :ax:          <matplotlib.axes> the axes to draw the skeleton
        :pts:         <numpy.ndarray> of n_joints x dims
        :joint_color: <string> the color to draw joints;
        :bone_color:  <string> the color to draw bones; "auto" means to use
        different colors to distinguish left, right and middle bones.
        :return:      <matplotlib.axes> the painted axes.
        """
        Nj = pts.shape[0]
        dim = pts.shape[1]
        bone_color_list = [bone_color] * Nj
        child_list = list(range(Nj))
        child_list.remove(self.root["index"])
        if bone_color == "auto":
            for i in self.left_bones:
                bone_color_list[i] = '#F29F05'
            for i in self.right_bones:
                bone_color_list[i] = '#7C8C03'
            for i in self.middle_bones:
                bone_color_list[i] = '#2E62A6'
        if dim == 2:
            for i in child_list:
                ax.plot(*[pt.reshape(2,) for pt in np.split(
                    pts[[i, self.node_list[i]["parent"]], :], 2, axis=1)],
                    color=bone_color_list[i], linewidth=bone_width)
            ax.scatter(*np.split(pts, 2, axis=1), color=joint_color, s=joint_size)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        elif dim == 3:
            for i in child_list:
                ax.plot3D(*[pt.reshape(2,) for pt in np.split(
                    pts[[i, self.node_list[i]["parent"]], :], 3, axis=1)],
                    color=bone_color_list[i], linewidth=bone_width)
            ax.scatter3D(*np.split(pts, 3, axis=1), color=joint_color, s=joint_size)
            # achieve equal visual lengths on three axes.
            extents = np.array(
                [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))
            ax.view_init(elev=0, azim=0)
            tmp = [getattr(ax, 'set_{}ticks'.format(dim))([]) for dim in 'xyz']
            ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); ax.set_zlabel(r"$z$")

        return ax

    def get_bl_vec(self, pose3D):
        """
        :pose3D: <numpy.ndarray> of self.size x 3, the 3D joint coordinates.
        :return: <numpy.ndarray> of (self.size-1) x 1, the 3D bone length vector
        """
        return np.linalg.norm((self.conv_J2B @ pose3D.reshape(-1, 1))[3:]\
            .reshape(self.size-1, 3), axis=1).reshape(self.size-1, 1)
        
    def save(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.node_list, f)
        
    def update_limb_pairs(self):
        self.limb_pairs = np.array([[self.node_list[i]["parent"], i] for i in range(self.size) if i != self.root["index"]])
            

def create_human_tree(data_type="human3.6m"):
    """
    create human tree structure according to data_type
    return a DictTree object.
    """
    if data_type == "human3.6m" or data_type == "mhad" or data_type == "joint":
        human_tree = DictTree(17, {"name":"Hip", "index":6})
        human_tree.create_node("RHip", 2, parent=6)
        human_tree.create_node("RKnee", 1, parent=2)
        human_tree.create_node("RFoot", 0, parent=1)
        human_tree.create_node("LHip", 3, parent=6)
        human_tree.create_node("LKnee", 4, parent=3)
        human_tree.create_node("LFoot", 5, parent=4)
        human_tree.create_node("Spine", 7, parent=6)
        human_tree.create_node("Thorax", 8, parent=7)
        human_tree.create_node("Neck", 16, parent=8)
        human_tree.create_node("Head", 9, parent=16)
        human_tree.create_node("LShoulder", 13, parent=8)
        human_tree.create_node("LElbow", 14, parent=13)
        human_tree.create_node("LWrist", 15, parent=14)
        human_tree.create_node("RShoulder", 12, parent=8)
        human_tree.create_node("RElbow", 11, parent=12)
        human_tree.create_node("RWrist", 10, parent=11)
        human_tree.left_bones = [3, 4, 5, 13, 14, 15]
        human_tree.right_bones = [2, 1, 0, 12, 11, 10]
        human_tree.middle_bones = [7, 8, 16, 9]
    elif data_type == "totalcapture":
        human_tree = DictTree(16, {"name":"Hips", "index":0})
        human_tree.create_node("RightUpLeg", 1, parent=0)
        human_tree.create_node("RightLeg", 2, parent=1)
        human_tree.create_node("RightFoot", 3, parent=2)
        human_tree.create_node("LeftUpLeg", 4, parent=0)
        human_tree.create_node("LeftLeg", 5, parent=4)
        human_tree.create_node("LeftFoot", 6, parent=5)
        human_tree.create_node("Spine", 7, parent=0)
        human_tree.create_node("Neck", 8, parent=7)
        human_tree.create_node("Head", 9, parent=8)
        human_tree.create_node("LeftArm", 10, parent=8)
        human_tree.create_node("LeftForeArm", 11, parent=10)
        human_tree.create_node("LeftHand", 12, parent=11)
        human_tree.create_node("RightArm", 13, parent=8)
        human_tree.create_node("RightForeArm", 14, parent=13)
        human_tree.create_node("RightHand", 15, parent=14)
        human_tree.left_bones = [4, 5, 6, 10, 11, 12]
        human_tree.right_bones = [1, 2, 3, 13, 14, 15]
        human_tree.middle_bones = [7, 8, 9]
    human_tree.get_conv_mat()
    human_tree.update_limb_pairs()

    return human_tree


def get_inner_mat(u, v):
    return np.array([[1, 0, -u], [0, 1, -v], [-u, -v, u**2+v**2]])


if __name__ == "__main__":
    ht = create_human_tree("totalcapture")
    lines = []
    for i in range(16):
        name = ht.node_list[i]["name"]
        lines.append(f"'{name}': {i}")
    
    print(",\n".join(lines))
    for nd in ht.node_list:
        if "parent" in nd:
            print([nd["index"], nd["parent"]], ",", sep="")
