import numpy as np
import zarr
import cv2
from functools import partial
import copy

from reactive_diffusion_policy.model.common.pca_embedding import PCAEmbedding
from reactive_diffusion_policy.common.tactile_marker_utils import display_motion

test_zarr_path = 'data/marker_motion/mctac_v1.zarr'
TRANSFORM_MATRIX_PATH = 'data/PCA_Transform_McTAC_v1/pca_transform_matrix.npy'
MEAN_MATRIX_PATH = 'data/PCA_Transform_McTAC_v1/pca_mean_matrix.npy'

if __name__ == '__main__':
    pca_embedding = PCAEmbedding(
        n_components=15, normalize=False, mode='Eval', store=False,
        transformation_matrix_path=TRANSFORM_MATRIX_PATH,
        mean_matrix_path=MEAN_MATRIX_PATH
    )
    zarr_data = zarr.open(test_zarr_path, mode='r')
    # Read rgb image and initial marker from zarr path
    background_image = zarr_data['color_image'][0][:]
    initial_markers = zarr_data['initial_marker'][0][:]
    original_shape = initial_markers.shape
    h, w = background_image.shape[:2]

    coefficients = np.ones(15)

    def on_trackbar(val, dim):
        global coefficients
        coefficients[dim] = (val - 100) / 10.0  # the actual range of the coefficients is (-10, 10)

    cv2.namedWindow('Tick Bar')

    for i in range(15):
        cv2.createTrackbar(f'Coeff {i+1}', 'Tick Bar', 100, 200, partial(on_trackbar, dim=i))

    while True:
        background_frame = copy.deepcopy(background_image)
        background_marker = copy.deepcopy(initial_markers[:, :2] * np.array([w, h]))

        demo_motion = coefficients # (15,)

        demo_motion_reconstructed = (demo_motion @ pca_embedding.W.T + pca_embedding.mean).reshape((
            original_shape[0], -1)) * np.array([w, h])

        adjusted_marker_img = display_motion(background_frame, background_marker, demo_motion_reconstructed, 0.5, 2)
        cv2.imshow('frame with adjusted marker motion' , adjusted_marker_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
