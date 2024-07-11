import numpy as np
import cv2
import os

def save_img(file_path, img):
    cv2.imwrite(file_path, img)

def draw_line_with_angle(width, height, angle, line_length, thickness=5, img=None, center_x=None, center_y=None):
    img = np.zeros((height, width)) if img is None else img
    
    if center_x is None:
        center_x = width // 2
    if center_y is None:
        center_y = height // 2
        
    half_length = line_length // 2
    
    if angle == 0:
        start_point = (center_x, center_y - half_length)
        end_point = (center_x, center_y + half_length)
    elif angle == 90:
        start_point = (center_x - half_length, center_y)
        end_point = (center_x + half_length, center_y)
    elif angle == 45:
        start_point = (center_x - half_length, center_y + half_length)
        end_point = (center_x + half_length, center_y - half_length)
    elif angle == 135:
        start_point = (center_x - half_length, center_y - half_length)
        end_point = (center_x + half_length, center_y + half_length)
    
    cv2.line(img, start_point, end_point, 255, thickness)
    
    return img

def apply_gaussian_filter(img, kernel_size=(9, 9), sigma=5):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def create_dataset(output_dir, filtered_dir, width=256, height=256, line_length=100, thickness=17, positions=4, variations=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
    
    angles = [0, 45, 90, 135]
    position_offsets = np.linspace(-width//4, width//4, positions)
    
    corners = [(width // 4, height // 4), (3 * width // 4, height // 4), (width // 4, 3 * height // 4), (3 * width // 4, 3 * height // 4)]
    
    for angle in angles:
        for var in range(variations):
            if angle in [45, 135]:
                for i, (cx, cy) in enumerate(corners):
                    img = draw_line_with_angle(width, height, angle, line_length, thickness, center_x=cx, center_y=cy)
                    img_name = f"angle_{angle}_corner_{i}_var_{var}.png"
                    img_path = os.path.join(output_dir, img_name)
                    
                    save_img(img_path, img)

                    # Apply Gaussian filter and save the result
                    filtered_img = apply_gaussian_filter(img)
                    filtered_img_name = f"filtered_angle_{angle}_corner_{i}_var_{var}.png"
                    filtered_img_path = os.path.join(filtered_dir, filtered_img_name)
                    
                    save_img(filtered_img_path, filtered_img)
            else:
                for pos in range(positions):
                    img = draw_line_with_angle(width, height, angle, line_length, thickness)
                    position_offset = position_offsets[pos]
                    
                    if angle == 0:
                        M = np.float32([[1, 0, position_offset], [0, 1, 0]])
                    elif angle == 90:
                        M = np.float32([[1, 0, 0], [0, 1, position_offset]])
                    
                    img = cv2.warpAffine(img, M, (width, height))
                    
                    img_name = f"angle_{angle}_pos_{pos}_var_{var}.png"
                    img_path = os.path.join(output_dir, img_name)
                    
                    save_img(img_path, img)

                    # Apply Gaussian filter and save the result
                    filtered_img = apply_gaussian_filter(img)
                    filtered_img_name = f"filtered_angle_{angle}_pos_{pos}_var_{var}.png"
                    filtered_img_path = os.path.join(filtered_dir, filtered_img_name)
                    
                    save_img(filtered_img_path, filtered_img)

if __name__ == '__main__':
    output_dir = 'images'
    filtered_dir = 'filtered_images'
    create_dataset(output_dir, filtered_dir)
