"""
All the path confgurations and constants to be defined here
"""

# defining all the necessary paths

input_video_path = './Input_Output/Input/4k2.avi'
input_images = './Input_Output/Input/images'
output_path = './Input_Output/Output'
binary_output_path = './Input_Output/binary_mask'
clustered_output_path = './Input_Output/clustered_mask'
track_model = './models/weights/track_model/best_model.h5'
anomaly_detector = './models/weights/anamoly_detector/autoencoder_model.h5'

EPS = 3
MIN_SAMPLES = 10
MAX_TILE_SIZE = 28