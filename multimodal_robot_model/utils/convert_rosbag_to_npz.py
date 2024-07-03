import os
import cv2
import glob
import rospy
import rosbag
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
args = parser.parse_args()

topics = [
    "/joint_states",
    "/camera/color/image_raw/compressed",
]
step_duration = rospy.Duration.from_sec(1.0 / float(args.freq))

bag_path_list = glob.glob(os.path.join(args.in_dir, "**/*.bag"), recursive=True)
bag_path_list.sort()
for bag_path in bag_path_list:
    print(f"[convert_rosbag_to_npz] Load a rosbag file: {bag_path}")
    bag = rosbag.Bag(bag_path)

    start_time = rospy.Time.from_sec(bag.get_start_time())

    joint_list = []
    joint_target_time = start_time
    image_list = []
    image_target_time = start_time

    for topic, msg, time in bag.read_messages(topics):
        if topic == "/joint_states":
            while time >= joint_target_time:
                joint_list.append(msg.position[0:6]) # TODO: hardcode
                joint_target_time += step_duration
        elif topic == "/camera/color/image_raw/compressed":
            while time >= image_target_time:
                np_img = np.frombuffer(msg.data, np.uint8)
                np_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                image_list.append(np_img.astype(np.uint8))
                image_target_time += step_duration

    bag.close()

    # Convert list to array
    joints = np.array(joint_list, dtype=np.float32)
    images = np.array(image_list, dtype=np.uint8)

    # Get shorter length
    seq_len = min(len(joints), len(images))

    # Trim
    joints = joints[:seq_len]
    images = images[:seq_len]
    times = np.arange(seq_len) / float(args.freq)

    # Save
    if args.out_dir is None:
        npz_path = bag_path.split(".bag")[0] + ".npz"
    else:
        filename = os.path.splitext(os.path.basename(bag_path))[0]
        npz_path = os.path.join(args.out_dir, filename + ".npz")
    print(f"[convert_rosbag_to_npz] Save a npz file: {npz_path}")
    np.savez(
        npz_path,
        time=times,
        joint=joints,
        front_image=images,
        wrench=np.zeros((seq_len, 6)), # TODO: dummy
        side_image=np.zeros_like(images), # TODO: dummy
    )
