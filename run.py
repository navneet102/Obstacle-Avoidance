import socket
import os
import glob
import torch
import utils
import cv2
import argparse
import time
import numpy as np
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

# Replace with the IP address of your ESP8266 and the port number (80 in this case)
esp_ip = "ESP8266_IP_ADDRESS"
esp_port = 80

def create_client_socket():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((esp_ip, esp_port))
    return client_socket

first_execution = True

def compute_openness(depth_map, num_regions=4):
    height, width = depth_map.shape
    region_width = width // num_regions
    avg_depths = []
    for i in range(num_regions):
        region = depth_map[:, i * region_width:(i + 1) * region_width]
        avg_depths.append(np.mean(region))
    return avg_depths

def decide_turn(avg_depths):
    return np.argmin(avg_depths) + 1

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    global first_execution
    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False
        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)
        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("Optimization to half-floats activated. Use with caution, because models like Swin require float precision to work properly and may yield non-finite depth values to some extent for half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"Input resized to {width}x{height} before entering the encoder")
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
        return np.hstack((image, right_side))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.stack((grayscale_image,)*3, axis=-1)
    for i in range(1, num_regions):
        cv2.line(grayscale_image, (i * grayscale_image.shape[1] // num_regions, 0),
                 (i * grayscale_image.shape[1] // num_regions, grayscale_image.shape[0]), (0, 0, 0), 2)
    return np.hstack((grayscale_image, right_side))

def get_prediction(device, model, model_type, img, input_size, target_size, optimize):
    img_input = utils.preprocess(img, input_size, model_type)
    depth_map = process(device, model, model_type, img_input, input_size, target_size, optimize, use_camera=True)
    return depth_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", default="dpt_hybrid")
    parser.add_argument("-t", "--target_size", default=[384, 384], nargs="+", type=int)
    parser.add_argument("-c", "--side", action="store_true")
    parser.add_argument("-v", "--video", action="store_true")
    parser.add_argument("-i", "--input_path", default="input")
    parser.add_argument("-o", "--output_path", default="output")
    parser.add_argument("-r", "--record_video", action="store_true")
    parser.add_argument("-g", "--grayscale", action="store_true")
    parser.add_argument("-d", "--depth_path")
    parser.add_argument("-u", "--utils_path")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    if args.utils_path:
        utils_path = args.utils_path
    else:
        utils_path = os.path.join("..", "utils")
    sys.path.append(utils_path)

    if args.depth_path:
        depth_path = args.depth_path
    else:
        depth_path = os.path.join("..", "input", "depth")
    sys.path.append(depth_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    model_path = default_models[model_type]
    model, transform, net_w, net_h = load_model(device, model_type, model_path)

    if args.video:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        while True:
            frame = vs.read()
            depth_map = get_prediction(device, model, model_type, frame, (net_w, net_h), args.target_size, args.optimize)
            avg_depths = compute_openness(depth_map)
            turn_direction = decide_turn(avg_depths)
            
            client_socket = create_client_socket()
            client_socket.sendall(str(turn_direction).encode())
            client_socket.close()
            
            print(f"Sending turn direction: {turn_direction}")
            if args.side:
                side_by_side = create_side_by_side(frame, depth_map, args.grayscale)
                cv2.imshow("Depth Map", side_by_side)
            else:
                cv2.imshow("Depth Map", depth_map)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        vs.stop()
    else:
        input_path = args.input_path
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)
        images = glob.glob(os.path.join(input_path, "*"))
        for img_name in images:
            img = utils.read_image(img_name)
            depth_map = get_prediction(device, model, model_type, img, (net_w, net_h), args.target_size, args.optimize)
            avg_depths = compute_openness(depth_map)
            turn_direction = decide_turn(avg_depths)
            
            client_socket = create_client_socket()
            client_socket.sendall(str(turn_direction).encode())
            client_socket.close()
            
            print(f"Sending turn direction: {turn_direction}")
            if args.side:
                side_by_side = create_side_by_side(img, depth_map, args.grayscale)
                cv2.imwrite(os.path.join(output_path, os.path.basename(img_name)), side_by_side)
            else:
                utils.write_depth(output_path, os.path.basename(img_name), depth_map, bits=2)
