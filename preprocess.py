from PIL import Image
import os
import argparse

def crop_and_rescale_image(image_path, output_path, size=(512, 512)):
    """
    Crops the image from the center and rescales it to the specified size.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        size (tuple): Desired output size (width, height).
    """
    with Image.open(image_path) as img:
        # Calculate center crop dimensions
        width, height = img.size
        new_width, new_height = size
        left = (width - min(width, height)) // 2
        top = (height - min(width, height)) // 2
        right = left + min(width, height)
        bottom = top + min(width, height)

        # Crop and resize
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize(size, Image.BILINEAR)

        # Save the processed image
        img_resized.save(output_path)

def process_images_in_folder(input_folder, output_folder, size=(512, 512)):
    """
    Processes all images in the input folder by cropping and rescaling them.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save processed images.
        size (tuple): Desired output size (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path):
            try:
                crop_and_rescale_image(input_path, output_path, size)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and rescale images in a folder.")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing input images.")
    parser.add_argument("--output_folder", type=str, help="Path to the folder to save processed images.")
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), help="Desired output size (width height).")

    args = parser.parse_args()

    process_images_in_folder(args.input_folder, args.output_folder, tuple(args.size))