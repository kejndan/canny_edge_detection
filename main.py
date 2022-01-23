from core import canny_edge_detection
import argparse
from utils import read_image, save_image, show_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for blur image')
    parser.add_argument('--sigma', type=int, default=1, help='sigma for blur image')
    parser.add_argument('--lower_thr', type=float, default=0.06, help="% of max value after NMS. It's lower threshold for hysteresis")
    parser.add_argument('--upper_thr', type=float, default=0.18, help="% of max value after NMS. It's upper threshold for hysteresis")
    args = parser.parse_args()

    img = read_image('images/emma.jpeg')
    res_img = canny_edge_detection(img, args.upper_thr, args.lower_thr, args.kernel_size, args.sigma)
    show_image(res_img)
    save_image(res_img, 'result/result_img.jpeg')
