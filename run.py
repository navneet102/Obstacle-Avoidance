import os
import glob
import torch
import utils
import cv2
import argparse
import time
import socket
import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

first_execution = True

def send_region_to_esp8266(region_number):
    host = '192.168.149.169'  # Replace with the actual IP address of the ESP8266
    port = 80  # Port number where the ESP8266 server is listening

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        message = f"{region_number}\n"
        s.sendall(message.encode())
        # Wait for a response to ensure data is received correctly
        response = s.recv(1024)
        print(f"Response from ESP8266: {response.decode()}")

def compute_openness(depth_map, num_regions=4):
    # Split the depth map into vertical regions
    height, width = depth_map.shape
    region_width = width // num_regions

    avg_depths = []
    for i in range(num_regions):
        region = depth_map[:, i * region_width:(i + 1) * region_width]
        avg_depths.append(np.mean(region))

    return avg_depths

def decide_turn(avg_depths):
    # Return the region with the maximum average depth
    return np.argmin(avg_depths) + 1

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def create_side_by_side(image, depth, grayscale, num_regions=4):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is not None:
        combined_image = np.concatenate((image, right_side), axis=1)
    else:
        combined_image = right_side

    # Draw region lines and numbers
    height, width, _ = combined_image.shape
    region_width = width // num_regions

    for i in range(num_regions):
        x = i * region_width
        cv2.line(combined_image, (x, 0), (x, height), (255, 255, 255), 1)
        cv2.putText(combined_image, str(i + 1), (x + region_width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return combined_image

def run(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    print("Initialize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)
    else:
        print("No input path specified. Grabbing images from camera.")

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):
            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

            original_image_rgb = utils.read_image(image_name)
            image = transform({"image": original_image_rgb})["image"]

            with torch.no_grad():
                prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                                     optimize, False)

            if output_path is not None:
                filename = os.path.join(output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type)
                if not side:
                    utils.write_depth(filename, prediction, grayscale, bits=2)
                else:
                    original_image_bgr = np.flip(original_image_rgb, 2)
                    content = create_side_by_side(original_image_bgr*255, prediction, grayscale)
                    cv2.imwrite(filename + ".png", content)
                utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))

    else:
        with torch.no_grad():
            fps = 1
            video = VideoStream(3).start()
            time_start = time.time()
            frame_index = 0
            print_interval = 1  # seconds
            last_print_time = time.time()
            
            while True:
                frame = video.read()
                if frame is not None:
                    original_image_rgb = np.flip(frame, 2)
                    image = transform({"image": original_image_rgb/255})["image"]

                    prediction = process(device, model, model_type, image, (net_w, net_h),
                                         original_image_rgb.shape[1::-1], optimize, True)

                    if time.time() - last_print_time >= print_interval:
                        avg_depths = compute_openness(prediction)
                        turn_direction = decide_turn(avg_depths)
                        print(f"Turn direction: Region {turn_direction}")
                        send_region_to_esp8266(turn_direction)  # Send the region number to the ESP8266
                        last_print_time = time.time()

                    original_image_bgr = np.flip(original_image_rgb, 2) if side else None
                    content = create_side_by_side(original_image_bgr, prediction, grayscale)
                    cv2.imshow('MiDaS Depth Estimation - Press Escape to close window', content/255)
                    cv2.imshow('Original Image', frame)

                    if output_path is not None:
                        filename = os.path.join(output_path, 'Camera' + '-' + model_type + '_' + str(frame_index))
                        cv2.imwrite(filename + ".png", content)

                    alpha = 0.1
                    if time.time()-time_start > 0:
                        fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)
                        time_start = time.time()
                    print(f"\rFPS: {round(fps,2)}", end="")

                    if cv2.waitKey(1) == 27:
                        break

                    frame_index += 1
        print()

    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use grayscale colormap'
                        )

    args = parser.parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)
