import argparse
import os
import cv2

from scan import DocScanner
from ocr import ImageExtractor

def scan_ocr(img_path, is_interactive_mode):
    from processors import GrayScaler, TruncThresholder
    
    scanner = DocScanner(is_interactive_mode)
    extractor = ImageExtractor()

    need_output_process = True
    processors = [
        # Resizer(height = 2000, output_process = need_output_process),
        # Enhancer(output_process = need_output_process),
        GrayScaler(output_process = need_output_process),
        # BlurDenoiser(strength = 9, output_process = need_output_process),
        # FastDenoiser(strength = 9, output_process = need_output_process),
        TruncThresholder(output_process = need_output_process),
        # Dilater(output_process = need_output_process),
        # Eroder(output_process = need_output_process),
        # Opener2(output_process = need_output_process),
    ]


    scanned_image = scanner.scan(img_path)
    processed_image = scanned_image
    for processor in processors:
        processed_image = processor(processed_image)

    extracted_data = extractor(processed_image)

    print('extracted_data', extracted_data)
    
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Flag for manually verifying and/or setting document corners")

    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    interactive_mode = args["i"]

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        scan_ocr(im_file_path, interactive_mode)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        for im in im_files:
            scan_ocr(im_dir + '/' + im, interactive_mode)