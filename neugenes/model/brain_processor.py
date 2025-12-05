import pandas as pd
import os
import cv2
from neugenes.model.brain_processor import config
from model.data_preprocessing import spaces_in_filenames, nd2_to_tiff
from model.signal_postprocessing import (to_csv,
                                   update_structure_weights,
                                   calibrate_expression)
from model.utils import (acronymn_to_id, 
                   atlas_registration,
                   generate_alignment_matrix,
                   process_image,
                   process_mask,
                   cells_in_ROI,
                   download_overlay)
# os.chdir(config.root_directory)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json

def process(base_dir,structure_acronymns,save_mask = False):
    # preprocess dataset 
    base_dir = os.path.join(os.path.dirname(config.root_directory),base_dir)
    print(base_dir)
    spaces_in_filenames(base_dir)
    R  = nd2_to_tiff(base_dir)
    data_path = os.path.join(base_dir,R)
    print("data path", data_path)
    structures = acronymn_to_id(structure_acronymns)
    print(structures)
    
    # brain registration
    registration = atlas_registration(data_path,ensemble=True,index_order=False,index_spacing=False,section_thickness = None)

    # iterative base 
    result = []
    im_n = 1
    num = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpeg', '.tif', '.jpg', '.png'))]   # count
    
    # output paths
    save = os.path.join(data_path,f'output')


    os.makedirs(save, exist_ok=True)

    all_hull_results = []

    for filename in os.listdir(data_path):
        path = os.path.join(data_path,filename)

        if os.path.isfile(path) and path.lower().endswith(('.jpeg', '.tif', '.jpg', '.png')):
            print(f'processing image {im_n}/{len(num)} --> {filename}')
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cell_yx = process_image(path)
            alignment = generate_alignment_matrix(registration,filename)
            
            # iteratively process structures 
            image_result = {'Filename': filename}
            if save_mask:
                all_masks = []
            for i in range(len(structures)):
                id,ac = structures[i],structure_acronymns[i]
                count = 0
                _, mask_yx,_ = process_mask(id, alignment,image)
                local_yx, count = cells_in_ROI(mask_yx,cell_yx,threshold=9)
                image_result[f'mask_{id}'] = count
                if save_mask:
                    all_masks.append(mask_yx)
                # download_overlay(save,filename,mask_yx,local_yx,ac,image)
            
            # print(all_masks)
            if save_mask:
                dpi = 100  # Set DPI
                figsize = (image.shape[1] / dpi, image.shape[0] / dpi)  # Convert pixels to inches
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # Ensure exact pixel match
                
                ax.imshow(image)
                structure_hulls = {}
                for i in range(len(structures)):
                    mask_yx = all_masks[i]
                    ac = structure_acronymns[i]
                    hulls_for_structure = []
                    # ax.scatter(mask_yx[:, 1], mask_yx[:, 0], s=3, edgecolor='red')
                    if mask_yx is not None and len(mask_yx) > 2:
                        mask_xy = mask_yx[:, [1, 0]]
                        clustering = DBSCAN(eps=10, min_samples=5).fit(mask_xy)
                        labels = clustering.labels_
                        for label in set(labels):
                            if label == -1: continue  # Skip noise
                            cluster_pts = mask_xy[labels == label]
                            if len(cluster_pts) >= 3:
                                cluster_pts_cv = cluster_pts.astype(np.int32).reshape(-1, 1, 2)
                                hull = cv2.convexHull(cluster_pts_cv)

                                hull_xy = hull.reshape(-1, 2) # Convert to [x, y]
                                hull_xy = np.vstack([hull_xy, hull_xy[0]])  # close loop

                                # Convert to list of [x, y] for JSON
                                hulls_for_structure.append(hull_xy.tolist())
                                ax.plot(hull_xy[:, 0], hull_xy[:, 1], linewidth=4, color='red')
                        hull = cv2.convexHull(mask_xy.astype(np.int32))
                        ax.scatter(mask_yx[:, 1], mask_yx[:, 0], s=3, edgecolor='red')
                    if hulls_for_structure:
                        structure_hulls[ac] = hulls_for_structure

                # Remove axes and padding to keep original look
                ax.set_xticks([])  
                ax.set_yticks([])  
                ax.set_frame_on(False)  

                # Save the image with the exact original pixel dimensions
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
                
                # plt.figure(figsize=(20,20))
                # plt.axis('off')
                # plt.imshow(image,cmap='gray',extent=[0, image.shape[1], image.shape[0], 0])
                # for mask_yx in all_masks:
                #     plt.scatter(mask_yx[:, 1], mask_yx[:, 0], s=3, edgecolor='red')
                
                save = os.path.join(data_path, os.path.splitext(filename)[0] + '_mask.png')
                plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0)  # Save to a new temp file
                plt.close() 
                print(f"Image loaded: {cv2.imread(path).shape}")
                print(f"Image after processing: {image.shape}")

                if structure_hulls:
                    all_hull_results.append({
                        "Filename": filename,
                        "Dimensions": [image.shape[1], image.shape[0]],  # width, height
                        "Structures": structure_hulls
                    })
            result.append(image_result)
            im_n+=1
    
    to_csv(result,os.path.join(config.output_directory,f'{data_path}/result_raw.csv'),acronym_map = config.acronym_map)
    structure_weights = update_structure_weights(result)
    calibrated_result = calibrate_expression(result,structure_weights)
    to_csv(calibrated_result,os.path.join(config.output_directory,f'{data_path}/result_norm.csv'),acronym_map = config.acronym_map)
    if save_mask and all_hull_results:
        hull_json_path = os.path.join(config.output_directory, f'{data_path}/convex_hulls.json')
        with open(hull_json_path, 'w') as f:
            json.dump(all_hull_results, f, indent=2)
        print(f"Saved all convex hulls to: {hull_json_path}")


