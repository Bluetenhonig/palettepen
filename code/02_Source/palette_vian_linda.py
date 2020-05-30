"""
Author: Gaudenz Halter
Copyright: University of Zurich

This code is part of VIAN - A visual annotation system, and licenced under the LGPL.

"""
import h5py
from scipy.cluster.hierarchy import *
from fastcluster import linkage
import numpy as np
import cv2
from typing import List


class PaletteAsset():
    """
    A container class used by VIAN. Use to_hdf5() to generate data, as stored by VIAN in the hdf5 manager.
    """
    MERGE_DIST = 0
    MERGE_DEPTH = 1
    B = 2
    G = 3
    R = 4
    COUNT = 5

    def __init__(self, tree, merge_dists):
        self.tree = tree
        self.merge_dists = merge_dists

    def to_hdf5(self):
        d = np.zeros(shape=(COLOR_PALETTES_MAX_LENGTH, 6))
        count = COLOR_PALETTES_MAX_LENGTH
        if len(self.tree[0]) < COLOR_PALETTES_MAX_LENGTH:
            count = len(self.tree[0])
        d[:len(self.merge_dists), 0] = self.merge_dists
        d[:count, 1] = self.tree[0][:count]
        d[:count, 2:5] = self.tree[1][:count]
        d[:count, 5] = self.tree[2][:count]
        return d


class PaletteExtractorModel:
    """
    Seeds extraction model.
    """
    def __init__(self, img, n_pixels = 100, num_levels = 4):
        self.model = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], n_pixels, num_levels=num_levels)

    def forward(self, img, n_pixels = 100):

        self.model.iterate(img, 4)
        return self.model.getLabels()

    def labels_to_avg_color_mask(self, lab, labels):
        indices = np.unique(labels)
        for idx in indices:
            pixels = np.where(labels == idx)
            lab[pixels] = np.mean(lab[pixels], axis = 0)
        return lab

    def labels_to_palette(self, lab, labels):
        indices = np.unique(labels)
        bins = []
        for idx in indices:
            pixels = np.where(labels == idx)
            avg_col = np.mean(lab[pixels], axis = 0)
            n_pixels = pixels[0].shape[0]
            bins.append([avg_col, n_pixels])

        bins = np.array(sorted(bins, key=lambda x:x[1], reverse=True))

        n_palette = 10
        preview = np.zeros(shape=(100,1500,3))
        total = np.sum(bins[0:n_palette, 1])
        last  = 0
        for b in range(n_palette):
            preview[:, last : last + (int(bins[b][1] * 1500 / total))] = bins[b, 0]
            last += int(bins[b][1] * 1500 / total)

        return preview.astype(np.uint8)

    def hist_to_palette(self, hist, n_col = 10):
        hist_lin = hist.reshape(hist.shape[1] * hist.shape[1] * hist.shape[2])
        shist = np.sort(hist_lin, axis=0)[-n_col:]
        bins = []
        for s in range(shist.shape[0]):
            indices = np.where(hist == shist[s])
            col = np.array([indices[0][0] * (256 / hist.shape[0]) + (256 / hist.shape[0] / 2),
                    indices[1][0] * (256 / hist.shape[0]) + (256 / hist.shape[0] / 2),
                    indices[2][0] * (256 / hist.shape[0]) + (256 / hist.shape[0] / 2)], dtype=np.uint8)
            bins.append([col, shist[s]])

        bins = np.array(bins)
        n_palette = n_col
        preview = np.zeros(shape=(100, 1500, 3))
        total = np.sum(bins[:, 1])
        last = 0
        for b in range(n_palette):
            preview[:, last: last + (int(bins[b][1] * 1500 / total))] = bins[b, 0]
            last += int(bins[b][1] * 1500 / total)

        return preview.astype(np.uint8)


def to_cluster_tree(Z, labels:List, colors, n_merge_steps = 1000, n_merge_per_lvl = 10):
    all_lbl = labels.copy()
    all_col = colors.copy()
    all_n = [1] * len(all_col)

    # print("Recreating Tree")
    for i in range(Z.shape[0]):
        a = int(Z[i][0])
        b = int(Z[i][1])
        all_lbl.append(len(all_lbl))
        all_col.append(np.divide((all_col[a] * all_n[a]) + (all_col[b] * all_n[b]), all_n[a] + all_n[b]))
        all_n.append(all_n[a] + all_n[b])

    result_lbl = [[len(all_lbl) - 1]]
    current_nodes = [len(all_lbl) - 1]
    i = 0

    merge_dists = []
    while(len(current_nodes) <= n_merge_steps and i < len(all_lbl)):
        try:
            curr_lbl = len(all_lbl) - 1 - i
            entry = Z[Z.shape[0] - 1 - i]
            a = int(entry[0])
            b = int(entry[1])
            idx = current_nodes.index(curr_lbl)
            current_nodes.remove(curr_lbl)
            current_nodes.insert(idx, a)
            current_nodes.insert(idx + 1, b)
            result_lbl.append(current_nodes.copy())
            merge_dists.append(entry[2])
            i += 1
        except Exception as e:
            print(e)
            break

    cols = np.zeros(shape=(0, 3), dtype=np.uint8)
    ns = np.zeros(shape=(0), dtype=np.uint16)
    layers = np.zeros(shape=(0), dtype=np.uint16)

    all_col = np.array(all_col, dtype=np.uint8)
    all_n = np.array(all_n, dtype=np.uint16)

    i = 0
    for r in result_lbl:
        if i > 10 and i % n_merge_per_lvl != 0:
            i += 1
            continue
        cols = np.concatenate((cols, all_col[r]))
        ns = np.concatenate((ns, all_n[r]))
        layers = np.concatenate((layers, np.array([i] * all_col[r].shape[0])))
        i += 1

    result = [layers, cols, ns]
    return result, merge_dists



def color_palette(frame_bgr, mask = None, mask_index = None, n_merge_steps = 100, image_size = 400.0, seeds_model = None,
                  n_pixels = 400, n_merge_per_lvl = 10, mask_inverse = False, normalization_lower_bound = 100.0,
                  seeds_input_width = 600, use_lab = True, show_seed=False, seed_labels = None) -> PaletteAsset:
    """
    Computes a hierarchical color palette as generated by VIAN, does not keep the original tree.

    :param frame_bgr: A frame in bgr uint8, currently float32 is not allowed since OpenCV may crash on it
    :param mask: An optional mask of labels
    :param mask_index: The label which the palette should be computed on
    :param mask_inverse: If true, all but the given mask_index will be computed.
    :param n_merge_steps: Number of merge steps to return (approximately), this is restricted by the
    :param image_size: image size to compute on
    :param seeds_model: the seeds model can optionally be given as argument to avoid initialization after each image
    :param n_pixels: number of super pixels to compute (approximately)
    :param n_merge_per_lvl: After the first 10 merges, every nth depth to store in the result
    :param normalization_lower_bound: Minimal number of pixels to keep a cluster
    :param seeds_input_width: input for the seeds model
    :param use_lab: if false, RGB will used for average computation instead of lab
    :param show_seed: if true, the seeds output will be shown in opencv, make sure to put cv2.waitKey() to see the result
    :return: PaletteAsset
    """
    # I'd prefere using LAB32 but it seems unstable for seed, leading to weird crashes in the C code.
    # We stick to LAB_U8
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    if seeds_input_width < frame.shape[0]:
        rx = seeds_input_width / frame.shape[0]
        frame = cv2.resize(frame, None, None, rx, rx, cv2.INTER_CUBIC)

    if seed_labels is None:
        if seeds_model is None:
            seeds_model = PaletteExtractorModel(frame, n_pixels=n_pixels, num_levels=4)
        labels = seeds_model.forward(frame, 200).astype(np.uint8)
    else:
        labels = seed_labels

    if show_seed:
        cv2.imshow("SEED", cv2.cvtColor(seeds_model.labels_to_avg_color_mask(frame, labels), cv2.COLOR_LAB2BGR))

    # Resizing all to same dimension
    fx = image_size / frame.shape[0]
    frame = cv2.resize(frame, None, None, fx, fx, cv2.INTER_CUBIC)
    labels = cv2.resize(labels, None, None, fx, fx, cv2.INTER_NEAREST)
    frame_bgr = cv2.resize(frame_bgr, None, None, fx, fx, cv2.INTER_CUBIC)

    # if a pixel mask is provided, only extract the pixels of the mask
    if mask is not None:
        mask = cv2.resize(mask, (labels.shape[1], labels.shape[0]), None, cv2.INTER_NEAREST)

        if mask_inverse:
            labels[np.where(mask == mask_index)] = 255
        else:
            labels[np.where(mask != mask_index)] = 255

        bins = np.unique(labels)
        bins = np.delete(bins, np.where(bins==255))
    else:
        bins = np.unique(labels)

    data = []
    hist = np.histogram(labels, bins = bins)

    # Make sure the normalization factor is not too low
    normalization_f = np.amin(hist[0])
    if normalization_f < normalization_lower_bound:
        normalization_f = normalization_lower_bound
    labels_list = []
    colors_list = []

    all_cols = []
    all_labels = []

    # print("Normalization Factor: ", normalization_f)
    for i, bin in enumerate(hist[0]):
        if bin < normalization_f:
            continue
        lbl = hist[1][i]
        if use_lab:
            avg_color = np.round(cv2.cvtColor(
                np.array([[np.mean(frame[np.where(labels == lbl)], axis=0)]], dtype=np.uint8),
                cv2.COLOR_LAB2BGR)[0, 0]).astype(np.uint8)
        else:
            avg_color = np.round(np.mean(frame_bgr[np.where(labels == lbl)], axis = 0)).astype(np.uint8)

        labels_list.append(lbl)
        colors_list.append(avg_color)

        data.extend([avg_color] * int(np.round(bin / normalization_f))*2)
        all_cols.extend([avg_color] * int(np.round(bin / normalization_f)) * 2)
        all_labels.extend([lbl] * int(np.round(bin / normalization_f)) * 2)

    data = np.array(data)

    Z = linkage(data, 'ward')

    tree, merge_dists = to_cluster_tree(Z, all_labels, all_cols, n_merge_steps, n_merge_per_lvl)
    return PaletteAsset(tree, merge_dists)


if __name__ == '__main__':

    # Here comes a list of images to process #
    paths = []
    images = [cv2.imread(p) for p in paths]

    # The maximal number of clusters to store.
    COLOR_PALETTES_MAX_LENGTH = 1024

    # VIAN stores the palettes in a hdf5 file:
    hdf5_file = h5py.File("palettes.hdf5", "w")
    hdf5_file.create_dataset("palettes", shape=(len(images), COLOR_PALETTES_MAX_LENGTH, 6), dtype=np.float16)
    hdf5_counter = 0

    # Compute the palette for all images
    for img_bgr in images:
        cv2.imshow("Input", img_bgr)
        cv2.waitKey(5)

        for use_lab in (True, False):
            # Compute a palette
            p = color_palette(img_bgr, use_lab=use_lab, show_seed=True)
            palette = p.to_hdf5()

            # We only store the lab palettes
            if use_lab:
                hdf5_file['palettes'][hdf5_counter] = palette
                hdf5_counter += 1

            # The output of PaletteAsset().to_hdf5() is a numpy array structured as follows:
            # It has a six columns, each of length COLOR_PALETTES_MAX_LENGTH. As a helper, they are in the PaletteAsset as CONSTANTS:
            # Column 0: Merge Distance of this node (PaletteAsset.MERGE_DIST)
            # Column 1: Merge Depth, e.g. 3th means, that there has been three previous splits of the tree (PaletteAsset.MERGE_DEPTH)
            # Column 2: avg B - Channel of this Node (PaletteAsset.B)
            # Column 3: avg G - Channel of this Node (PaletteAsset.G)
            # Column 4: avg R - Channel of this Node (PaletteAsset.R)
            # Column 5: Number of pixels in this node (PaletteAsset.COUNT)

            # The dimensions of each palette
            p_width = 1000
            p_height = 20

            palette_example = None

            # We iterate over all merge depths stored in the palette, and create a image showing the palette
            for d in np.unique(palette[:, p.MERGE_DEPTH]):
                # Get all nodes at the specific depth
                indices = np.where(palette[:, p.MERGE_DEPTH] == d)

                # Get the total amount of pixels to normalize the bins later
                total_pixels = np.sum(palette[indices, p.COUNT])

                # a palette at a specific merge depth
                row = np.zeros(shape=(p_height, p_width, 3), dtype=np.uint8)
                x = 0

                # For each color cluster in the specific merge depth, we want to create a patch, with a size which
                # corresponds to the number of pixels in the cluster.
                for i in indices[0].tolist():
                    width = int(np.round(palette[i, p.COUNT] / total_pixels * p_width))
                    row[:, x:x+width] = palette[i, p.B : p.R + 1] # + 1 to also get the R channel
                    x += width

                # Append to the result
                if palette_example is None:
                    palette_example = row
                else:
                    palette_example = np.vstack((palette_example, row))

            title = "Palette in LAB"
            if not use_lab:
                title = "Palette in RGB"
            cv2.imshow(title, palette_example)
        cv2.waitKey()

    cv2.waitKey()
    hdf5_file.close()