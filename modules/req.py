"""
Defining all the necessary functions here and calling from the main.py
"""

# importing all necessary packages
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# importing all necessary modules
from modules import config
from modules import cvfunc
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import DBSCAN
from PIL import Image 

"""
Custom loss function for track segmentation
It calculates two types of losses:
1. weighted_binary_crossentropy
2. focal_loss
3. combines both and gives out a combined loss
"""

"""
weighted_binary_crossentropy
"""
def weighted_binary_crossentropy(y_true, y_pred):
    w0 = 0.2
    w1 = 0.8
    bce = w1 * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
          w0 * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(bce)

"""
focal loss
"""
def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    focal_loss_value = (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) + \
                       alpha * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return -tf.reduce_mean(focal_loss_value)

"""
Combining both weighted binary crossentropy and focal loss
"""
def combined_loss(y_true, y_pred, alpha=0.8, gamma=2.0, w0=0.2, w1=0.8):
    w_bce = weighted_binary_crossentropy(y_true, y_pred)
    f_loss = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    # Combine the losses as per your requirement, for example, an average.
    combined = (w_bce + f_loss) / 2
    return combined

"""
Preprocessing function to preprocess the frame before passing it into Unet seg model
"""
def preprocess_image(frame, target_size=(576, 896)):
    img = Image.fromarray(frame)
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0 
    return img_array

"""
Postprocessing function to postprocess the frame after receiving it from Unet seg model
"""
def postprocess_mask(pred_mask,orginal_size):
    pred_mask = pred_mask[0] 
    pred_mask = pred_mask[:, :, 0] 
    pred_mask = Image.fromarray(pred_mask)
    pred_mask = pred_mask.resize(orginal_size, Image.NEAREST)
    pred_mask = tf.cast(pred_mask, tf.float32) 
    threshold = 0.4
    pred_mask = tf.where(pred_mask > threshold, 1, 0) 
    return np.array(pred_mask)


"""
function to check overlap of bboxes
"""
def is_overlapping(existing_tiles, new_tile):
    for ex_tile in existing_tiles:
        if not (ex_tile[2] < new_tile[0] or new_tile[2] < ex_tile[0] or 
                ex_tile[3] < new_tile[1] or new_tile[3] < ex_tile[1]):
            return True
    return False


"""
This function is to check the length of the cluster of raillines
"""
def cluster_length(cluster_points):
    # Assuming cluster_points is a list of (x,y) tuples
    return np.sum(np.sqrt(np.sum(np.diff(cluster_points, axis=0)**2, axis=1)))

"""
This function uses the dbscan algorithm for clustering the rail track
It then creates tiles of fixed sizes to be passed into autoencoder model
The autoencoder model predicts whether the tile is an anamoly or not based on a threshold value
"""
def dbscans(image, frame, anamoly_model, EPS, MIN_SAMPLES, max_tile_size):
    
    # making sure the image is in right format for thresholing
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary thresholding the input image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # finding contours of the binary thresholded image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # stacking the points as x,y points
    points = np.vstack([contour.reshape(-1, 2) for contour in contours])

    # sklearn's DBSCAN algorithm to cluster the rail tracks
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(points)
    labels = dbscan.labels_
    
    # Get cluster points and lengths
    cluster_lengths = []
    clusters = {}
    for label in set(labels):
        if label == -1:  # Ignore noise
            continue
        cluster_points = points[labels == label]
        length = cluster_length(cluster_points)
        cluster_lengths.append(length)
        clusters[label] = cluster_points

    
    # Identify the length of the longest cluster
    max_length = max(cluster_lengths)

    
    # Filter clusters based on length and remove noise 
    filtered_clusters = {label: cluster_points for label, cluster_points in clusters.items()
                         if cluster_length(cluster_points) >= max_length / 2}

    # Draw the clustered points with same dimensionsed image as the orginal frame
    clustered_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color_map = plt.get_cmap("tab10")

    original_color_image = frame
    bbox_image = original_color_image.copy()
 
    max_y = np.max(points[:, 1])
    existing_tiles = []

    for label, cluster_points in filtered_clusters.items():
        color = color_map(label)[:3]
        color = tuple(int(c * 255) for c in color[::-1])
        for (x, y) in cluster_points:
            cv2.circle(clustered_image, (x, y), radius=1, color=color, thickness=-1)
        cluster_points = cluster_points[np.argsort(-cluster_points[:, 1])]

        for i, (x, y) in enumerate(cluster_points):
            tile_size = int(max_tile_size * (y / max_y))  
            tile_size = max(1, tile_size)  
            half_tile = tile_size // 2
            x1, y1, x2, y2 = x - half_tile, y - half_tile, x + half_tile, y + half_tile
            
            if not is_overlapping(existing_tiles, (x1, y1, x2, y2)):
                #cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                existing_tiles.append((x1, y1, x2, y2))
                tile = original_color_image[y1:y2, x1:x2]

                # Ensure the tile is not empty before saving
                if tile.size > 0:
                    # Save the tile image
                    tile_img = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                    tile_img = tile_img.resize((16, 16), Image.Resampling.LANCZOS)
                    img_array = np.array(tile_img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    reconstructed_img = anamoly_model.predict(img_array)
                    mse = np.mean(np.square(img_array - reconstructed_img), axis=(1, 2, 3))
                    rmse = np.sqrt(mse)
                    # is_anomaly = rmse > 0.06263970211148262
                    is_anomaly = rmse > 0.07
                    if is_anomaly:
                        print(f"anamoly detected for: {(x1, y1), (x2, y2)}")
                        # Draw bounding box for anomalies
                        cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return clustered_image, bbox_image

"""
main function for processing each frame in a video

"""

def process_video(input_path, output_path, binary_path, clustered_path, track_model, anamoly_model):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # getting the width, height, fps of the extracted frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps)
    frame_count = 0
    saved_frame_count = 0

    # main loop which opens video and processes all the frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % interval == 0:
                # Convert frame to format suitable for model input
                processed_frame = preprocess_image(frame, target_size=(576, 896))
                pred_mask = track_model.predict(processed_frame)
                output_mask = postprocess_mask(pred_mask, orginal_size=(frame_width, frame_height))

                # Convert mask to 3 channels
                output_mask_3ch = np.repeat(output_mask[:, :, np.newaxis], 3, axis=2)
                output_mask_3ch = (output_mask_3ch * 255).astype(np.uint8)

                # Overlay the mask on the original frame
                combined_frame = cv2.addWeighted(frame, 1, output_mask_3ch, 0.5, 0)

                output_mask = np.expand_dims(output_mask, axis=-1) * 255
                w_reduce = cvfunc.reduce_width(output_mask)

                clustered_image, bbox_image = dbscans(w_reduce, frame, anamoly_model, config.EPS, config.MIN_SAMPLES, config.MAX_TILE_SIZE)
                cv2.imwrite(f"./Input_Output/clustered_mask/frame_{saved_frame_count:05d}.png", clustered_image)
                cv2.imwrite(f"./Input_Output/input_2_encoder/frame_{saved_frame_count:05d}.png", bbox_image)

                # Save the combined frame
                output_filename = os.path.join(output_path, f'frame_{saved_frame_count:05d}.png')
                # save the binary mask
                binary_output_filename = os.path.join(binary_path, f'frame_{saved_frame_count:05d}.png')
                clustered_image_filename = os.path.join(clustered_path, f'frame_{saved_frame_count:05d}.png')
                cv2.imwrite(output_filename, combined_frame)
                cv2.imwrite(binary_output_filename, output_mask)
                cv2.imwrite(clustered_image_filename, clustered_image)
                saved_frame_count += 1
            frame_count += 1

        else:
            break

    # Release everything when job is finished
    cap.release()


"""
main function to process all images in the given folder
"""

def process_images(input_folder, output_path, binary_path, clustered_path, track_model, anomaly_model):
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    
    saved_frame_count = 0

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
       
        # Check if the image was loaded successfully
        if frame is None:
            print(f"Error opening image file: {image_file}")
            continue
        
        # Process the image by sending it to the preprocess and postprocess function
        processed_frame = preprocess_image(frame, target_size=(576, 896))
        pred_mask = track_model.predict(processed_frame)
        output_mask = postprocess_mask(pred_mask, (frame.shape[1], frame.shape[0]))

       # Convert mask to 3 channels
        output_mask_3ch = np.repeat(output_mask[:, :, np.newaxis], 3, axis=2)
        output_mask_3ch = (output_mask_3ch * 255).astype(np.uint8)

        # Overlay the mask on the original frame
        combined_frame = cv2.addWeighted(frame, 1, output_mask_3ch, 0.5, 0)

        output_mask = np.expand_dims(output_mask, axis=-1) * 255
        w_reduce = cvfunc.reduce_width(output_mask)
        clustered_image, bbox_image = dbscans(w_reduce, frame, anomaly_model, config.EPS, config.MIN_SAMPLES, config.MAX_TILE_SIZE)
        cv2.imwrite(f"./Input_Output/input_2_encoder/frame_{saved_frame_count:05d}.png", bbox_image)
        cv2.imwrite(f"./Input_Output/clustered_mask/frame_{saved_frame_count:05d}.png", clustered_image)
        cv2.imwrite(f"./Input_Output/reduced_width/frame_{saved_frame_count:05d}.png", w_reduce)

        # Save the combined frame
        output_filename = os.path.join(output_path, f'frame_{saved_frame_count:05d}.png')
        # save the binary mask
        binary_output_filename = os.path.join(binary_path, f'frame_{saved_frame_count:05d}.png')
        clustered_image_filename = os.path.join(clustered_path, f'frame_{saved_frame_count:05d}.png')
        cv2.imwrite(output_filename, combined_frame)
        cv2.imwrite(binary_output_filename, output_mask)
        #cv2.imwrite(clustered_image_filename, clustered_image)
        saved_frame_count += 1