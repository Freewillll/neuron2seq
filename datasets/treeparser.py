import numpy as np
from treelib import Tree
from collections import defaultdict
from swc_handler import parse_swc, write_swc, get_child_dict, get_index_dict


def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
            x > imgshape[2] - 1 or \
            y > imgshape[1] - 1 or \
            z > imgshape[0] - 1:
        return False
    return True


def trim_out_of_box(tree_orig, imgshape, keep_candidate_points=True):
    """
    Trim the out-of-box leaves
    """
    # execute trimming
    child_dict = {}
    for leaf in tree_orig:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]

    pos_dict = {}
    for i, leaf in enumerate(tree_orig):
        pos_dict[leaf[0]] = leaf

    tree = []
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        ib = is_in_box(x, y, z, imgshape)
        leaf = (idx, type_, x, y, z, r, p, ib)
        if ib:
            tree.append(leaf)
        elif keep_candidate_points:
            if p in pos_dict and is_in_box(*pos_dict[p][2:5], imgshape):
                tree.append(leaf)
            elif idx in child_dict:
                for ch_leaf in child_dict[idx]:
                    if is_in_box(*pos_dict[ch_leaf][2:5], imgshape):
                        tree.append(leaf)
                        break
    return tree


def swc_to_seq(tree, imgshape, max_level=1, p_idx=-2):
    pos_dict = {}
    poses_list = []
    labels_list = []
    level_dict = defaultdict(list)
    roots = []
    for i, leaf in enumerate(tree):
        pos_dict[leaf[0]] = leaf

    for i, leaf in enumerate(tree):
        if leaf[p_idx] not in pos_dict:
            roots.append(leaf[0])

    def dfs(idx, level, child_dict, tree):
        # 0 root, 1 branching point, 2 tip node, 3 boundary point, 4 other node
        leaf = pos_dict[idx]
        x, y, z, r = leaf[2:6]
        tag = 4
        if idx not in child_dict:
            *_, ib = leaf
            if idx in roots:
                tag = 0
            elif ib == 1:
                tag = 2
                level += 1
            elif ib == 0:
                tag = 3
                level += 1

            if tag == 0 or tag == 2 or tag == 3:
                level_dict[level].append(idx)
            if idx in roots:
                tree.create_node(tag=tag, identifier=idx, data=(z, y, x))
            else:
                tree.create_node(tag=tag, identifier=idx, parent=leaf[p_idx], data=(z, y, x))
            return
        else:
            cnum = len(child_dict[idx])
            if idx in roots:
                tag = 0
            elif cnum == 1:
                tag = 4
            elif cnum >= 2:
                tag = 1
                level += 1

            if tag == 0 or tag == 1:
                level_dict[level].append(idx)
            if idx in roots:
                tree.create_node(tag=tag, identifier=idx, data=(z, y, x))
            else:
                tree.create_node(tag=tag, identifier=idx, parent=leaf[p_idx], data=(z, y, x))
            for cidx in child_dict[idx]:
                dfs(cidx, level, child_dict, tree)

    Trees = []
    child_dict = get_child_dict(tree, p_idx_in_leaf=p_idx)

    for idx in roots:
        tree = Tree()
        poses = []
        labels = []
        dfs(idx, 0, child_dict, tree)
        sorted(level_dict)
        for key in level_dict:
            if key <= max_level:
                for idx in level_dict[key]:
                    pos = []
                    node = tree.get_node(idx)
                    if node.tag == 3:
                        par = node._predecessor[node._initial_tree_id]
                        par_node = tree.get_node(par)
                        pos = par_node.data
                        label = node.tag
                    elif node.tag == 0:
                        pos = node.data
                        label = node.tag
                        idx, _, x, y, z, *_, par, ib = pos_dict[idx]
                        if ib == 0:
                            pos = tree.children(idx)[0].data
                    else:
                        pos = node.data
                        label = node.tag
                        
                    z, y, x = pos
                    poses.append([z, y, x])
                    labels.append(label)

        poses_list.append(poses)
        labels_list.append(labels)
        Trees.append(tree)
        level_dict.clear()

    max_len = 0
    max_idx = 0
    for idx, lab in enumerate(labels_list):
        if len(lab) >= max_len:
            max_idx = idx
            max_len = len(lab)
        
    return poses_list[max_idx], labels_list[max_idx]


if __name__ == '__main__':
    swcfile = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/17302_18816.00_39212.03_2416.26.swc'
    tree = parse_swc(swcfile)
    sz = 50
    sy = 50
    sx = 50
    new_tree = []
    for leaf in tree:
        idx, type_, x, y, z, r, p = leaf
        x = x - sx
        y = y - sy
        z = z - sz
        new_tree.append((idx, type_, x, y, z, r, p))
    tree = trim_out_of_box(new_tree, imgshape=[32, 64, 64])
    print(len(tree))
    poses, labels = swc_to_seq(tree, imgshape=[32,64,64])
    # print(seq_list)
    print(poses, labels)


