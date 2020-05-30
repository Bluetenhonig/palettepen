"""
Author: Gaudenz Halter, Linda Samsinger
Copyright: University of Zurich

This code is part of VIAN - A visual annotation system, and licenced under the LGPL.

"""

# Goal: For a given image, create a folder with output files: 
# - original image (as .jpg/.png)
# - palette of the image (as hdf5)
# - lab palette of the image (as .jpg/.png)
# - rgb palette of the image (as .jpg/.png)

# load modules
import h5py
from scipy.cluster.hierarchy import *
from fastcluster import linkage
import numpy as np
import cv2
from typing import List
import pandas as pd 

# define 2 classes 
class PaletteAsset():
    """
    A container class used by VIAN. Use to_hdf5() to generate data, as stored by VIAN in the hdf5 manager.
    """
    # 6 columns to put into table of color palette 
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
        d = np.zeros(shape=(CP_MAX_LENGTH, 6)) # table of Color palette: 1024 rows (every row is a bin or a color or a cluster), 6 columns
        d.fill(-1)
        count = CP_MAX_LENGTH
        if len(self.tree[0]) < CP_MAX_LENGTH:
            count = len(self.tree[0])
        d[:len(self.merge_dists), 0] = self.merge_dists
        d[:count, 1] = self.tree[0][:count]
        d[:count, 2:5] = self.tree[1][:count]
        d[:count, 5] = self.tree[2][:count]
        return d

# The output of PaletteAsset().to_hdf5() is a numpy array structured as follows:
# It has six columns, each of length CP_MAX_LENGTH. As a helper, they are in the PaletteAsset as CONSTANTS:
# Column 0: Merge Distance of this node (PaletteAsset.MERGE_DIST)
# Column 1: Merge Depth, e.g. 3th means, that there has been three previous splits of the tree (PaletteAsset.MERGE_DEPTH)
# Column 2: avg B - Channel of this Node (PaletteAsset.B)
# Column 3: avg G - Channel of this Node (PaletteAsset.G)
# Column 4: avg R - Channel of this Node (PaletteAsset.R)
# Column 5: Number of pixels in this node (PaletteAsset.COUNT)


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

#%%
        
# define 2 (helper) functions 
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
    # convert BGR to LAB_U8
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

#%%
  
if __name__ == '__main__':
        
   
    # load module
    import os
    import shutil
    
    # to specify 
    IMAGE_FILE = ['0_#BD8CB5_4b2578e3ee58ba107806ee430ead95.jpg']
    
    # Image
    IMAGE_PATH = r'D:\thesis\code\ColorCode12\pinterest\vintage-style' # load image 
    MANYIMAGES = True
    EXTENSION = '.jpg'
    #IMAGE_FILE = ["frame125.jpg"]
    # IMAGE_FILE = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
    # load files from directory 
#    IMAGE_FILE = []
#    for r, d, f in os.walk(IMAGE_PATH): # r=root, d=directories, f = files
#        for file in f:
#            if EXTENSION in file:
#                IMAGE_FILE.append(file)  
    
    # Color Palette
    CP_PATH =  IMAGE_PATH #save image-extracted CP
    SAVE2FOLDER = False  # saves output to folder for each image 
    CP_FOLDER = f"{IMAGE_FILE[:-4]}_palette"       
    CP_MAX_LENGTH = 1024 # The maximal number of clusters (=non-leaf node = bin in the tree)  to store, cumulative count of bins across all levels representative of the depth 
    CP_WIDTH = 1000  # set palette dimensions
    CP_HEIGHT = 20   # set palette dimensions
#    DEPTH = 100  # set palette depth [0,100] or 'depth' for all depths [optioinal] where 0 is top image, 100 is lowest level in image, only for 1024 the lowest level will be 100, otherwise it's a random number
    

    ################## Settings ############################
    # set path 
    os.chdir(IMAGE_PATH) 
        
    # read image/s : specify if one or more images 
    if not MANYIMAGES: 
        images = [cv2.imread(IMAGE_FILE)] # single image
    else: 
        images = [cv2.imread(img) for img in IMAGE_FILE]  # many images 
    
 
    ################## Compute Color Palette ############################
    # create/open/save up a hdf5- palette/s file for all images (100 palettes in the hdf5-file)
    with h5py.File(f"{IMAGE_FILE}_palettes.hdf5", "w") as hdf5_file:
        hdf5_file.create_dataset("palettes", shape=(len(images), CP_MAX_LENGTH, 6), dtype=np.float16)
        hdf5_counter = 0 
        # compute palette for (all) image/s   
        for m, img_bgr in enumerate(images):    
            # color space conversion: BGR to LAB and/or RGB 
            for use_lab in (True, False):
                # compute palette p
                p = color_palette(img_bgr, use_lab=use_lab, show_seed=False)            
                # save palette to hdf5
                palette = p.to_hdf5()    
                # only store lab palettes
                if use_lab:
                    hdf5_file['palettes'][hdf5_counter] = palette # or m as hdf5_counter
                    hdf5_counter += 1
                
                # initialize palette image and palette colors
                palette_image = None  
                palette_bins = []
                palette_bin_widths = []
                palette_cum_bin_widths = []
                palette_ratio_widths = []
                palette_colors = [] # BGR colors - all colors in palette_image
                depths = []
                # palette_image = height x width (=depth x total_pixels) (= rows x bins) (= x nodes)
                # iterate over all merge depths of the palette
                for j, DEPTH in enumerate(np.unique(palette[:, p.MERGE_DEPTH])): #len = 20 rows, [1,2,3,4 ..., 90, 100] - rows 11-19 not there because no need to save or visualize the bins, however the bins are calculated                
                    # set depth: DEPTH = depth, comment out depth iteration
                    if DEPTH == -1: 
                        continue 
                    indices = np.where(palette[:, p.MERGE_DEPTH] == DEPTH)
                    depths.append(j)
                    # total_pixels = width of palette image or bins
                    total_pixels = np.sum(palette[indices, p.COUNT])                    
                    t = palette[np.where(np.amax(palette[:, p.MERGE_DEPTH]))]
                    # a palette at a specific merge depth
                    # initialize row, row_colors and cumul width x 
                    row = np.zeros(shape=(CP_HEIGHT, CP_WIDTH, 3), dtype=np.uint8)           
                    bins = []
                    bin_widths = []
                    cum_bin_widths = []
                    ratio_widths = []
                    row_colors = []
                    cum_bin_width = 0
                    # For each color cluster in the specific merge depth, we want to create a patch, with a size which
                    # corresponds to the number of pixels in the cluster.
                    for l, i in enumerate(indices[0].tolist()): # list len = 101 bins, [514, 515, ..., 614] from dedicated hpf5 file space, however in the hdf5 file's space is reserved until 1024, but palette is smaller
                        width = int(np.round(palette[i, p.COUNT] / total_pixels * CP_WIDTH)) # [1000,448, 552, 448, ..., 33,43 ]
                        width_stat = palette[i, p.COUNT] / total_pixels
                        row[:, cum_bin_width:cum_bin_width+width] = palette[i, p.B : p.R + 1] # + 1 to also get the R channel
                        colors = palette[i, p.B : p.R + 1].tolist() # [b,g,r]
                        bins.append(l)
                        bin_widths.append(width)
                        cum_bin_widths.append(cum_bin_width)
                        ratio_widths.append(np.round(width_stat*100, 2))
                        if colors != [0.0, 0.0, 0.0]:
                            row_colors.append(colors)
                        cum_bin_width += width
                        #print('bin:',l, ', bin_width:',width,', cum_bin_width:' ,cum_bin_width, ',%width:',round(width/1003*100, 2) , '%', ',bgr-cols:', colors)                                        
                    # create image of palette 
                    if palette_image is None:
                        palette_image = row
                    else:
                        palette_image = np.vstack((palette_image, row))                       
                    if palette_bins is None:
                        palette_bins = bins
                    else:
                        palette_bins.append(bins)
                    if palette_bin_widths is None:
                        palette_bin_widths = bin_widths
                    else:
                        palette_bin_widths.append(bin_widths)                    
                    if palette_cum_bin_widths is None:
                        palette_cum_bin_widths = cum_bin_widths
                    else:
                        palette_cum_bin_widths.append(cum_bin_widths)                 
                    if palette_ratio_widths is None:
                        palette_ratio_widths = ratio_widths
                    else:
                        palette_ratio_widths.append(ratio_widths)              
                    if palette_colors is None:
                        palette_colors = row_colors
                    else:
                        palette_colors.append(row_colors)
                
                #get palette image colors for each depth (=row) start at top 0, end 100
                foo = []
                for d in depths: 
                    string = f'row {d}'
                    foo.append(string)
                    
                palette_info = {}
                for i, key in enumerate(foo): 
                    palette_info[key] = {}  
                    palette_info[key]['bin'] = palette_bins[i]
                    palette_info[key]['bin_width'] = palette_bin_widths[i] 
                    palette_info[key]['cum_bin_width'] = palette_cum_bin_widths[i] 
                    palette_info[key]['ratio_width'] = palette_ratio_widths[i] 
                    palette_info[key]['bgr_colors'] = palette_colors[i]                         
                        
                ################## Save Color Palette ############################                                  
                # save palette image 
                # name it
                img_name = f"{IMAGE_FILE[m][:-4]}_lab_palette.jpg"
                if not use_lab:
                    img_name = f"{IMAGE_FILE[m][:-4]}_rgb_palette.jpg"
                # show it                
                cv2.imshow(img_name,palette_image) # bgr-image, palette_image.shape=(20,1000,3) 
                cv2.waitKey(5)
                # save it
                #cv2.imwrite(img_name, palette_image)
                df = pd.DataFrame(data=palette_info)               
                df.to_csv(f"{IMAGE_FILE[m][:-4]}_bgr_palette.csv", sep=',', index=True)
                
                
                if SAVE2FOLDER: 
                    # path for saving output of each image into its own folder
                    os.chdir(CP_PATH)
                    folder_dir = os.path.join(CP_PATH, CP_FOLDER)
                    try: # make a folder for saving 
                        os.mkdir(CP_FOLDER)
                    except: # if folder already there, overwrite folder
                        shutil.rmtree(folder_dir)
                        os.mkdir(CP_FOLDER)        
                    os.chdir(folder_dir) 
                    
                    # write original image into saving folder 
                    cv2.imwrite(IMAGE_FILE, images[0])
            
  
    


                    
    
