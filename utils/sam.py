import numpy as np
import matplotlib.pyplot as plt


# Generate object masks using SAM alongside their original point from Molmo
def generate_masks_sam(predictor, image, input_points):
   
    predictor.set_image(image)

    masks = []
    mask_points = []
    for point in input_points:
        input_point = np.array([point])
        input_label = np.array([1])  # 1 = "foreground"
        masks_, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        masks.append(masks_[0]) 
        mask_points.append(point)

    return masks, mask_points

def display_mask(masks):
    for i, mask in enumerate(masks):
        plt.figure()
        plt.title(f"Masque de l'objet {i+1}")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.show()