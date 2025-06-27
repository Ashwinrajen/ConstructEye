import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU to use [default: GPU 0]")
parser.add_argument("--batch_size", type=int, default=1, help="Batch Size during training [default: 1]")
parser.add_argument("--num_point", type=int, default=4096, help="Point number [default: 4096]")
parser.add_argument("--model_path", required=True, help="model checkpoint file path")
parser.add_argument("--dump_dir", required=True, help="dump folder path")
parser.add_argument("--input_npy_file", required=True, help="Path to the input point cloud .npy file")
parser.add_argument("--no_clutter", action="store_true", help="If true, donot count the clutter class")
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
ROOM_PATH_LIST = [FLAGS.input_npy_file]  # Directly accept a single .npy file

NUM_CLASSES = 13


def evaluate():
    is_training = False

    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)

    ops = {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "pred": pred,
        "pred_softmax": pred_softmax,
        "loss": loss,
    }

    total_correct = 0
    total_seen = 0
    for room_path in ROOM_PATH_LIST:
        out_data_label_filename = os.path.basename(room_path)[:-4] + "_pred.txt"
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        print(room_path, out_data_label_filename)
        a, b = eval_one_epoch(sess, ops, room_path, out_data_label_filename)
        total_correct += a
        total_seen += b


def room2blocks_plus_normalized(data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug):
    """room2block, with input filename and RGB preprocessing.
    for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:, 0:6]
    data[:, 3:6] /= 255.0
    label = data_label[:, -1].astype(np.uint8)
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    data_batch, label_batch = room2blocks(
        data, label, num_point, block_size, stride, random_sample, sample_num, sample_aug
    )
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= minx + block_size / 2
        data_batch[b, :, 1] -= miny + block_size / 2
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch


def room2blocks_wrapper_normalized(
    data_label_filename, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1
):
    if data_label_filename[-3:] == "txt":
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == "npy":
        data_label = np.load(data_label_filename)
    else:
        print("Unknown file type! exiting.")
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug)


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):
    """Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert stride <= block_size

    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
            continue

        block_data = data[cond, :]
        block_label = label[cond]

        # randomly subsample data
        block_data_sampled, block_label_sampled = sample_data_label(block_data, block_label, num_point)
        block_data_list.append(np.expand_dims(block_data_sampled, 0))
        block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def sample_data(data, num_sample):
    """data is in N x ...
    we want to keep num_samplexC of them.
    if N > num_sample, we will randomly keep num_sample of them.
    if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if N == num_sample:
        return data, range(N)
    elif N > num_sample:
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def eval_one_epoch(sess, ops, room_path, out_data_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout_data_label = open(out_data_label_filename, "w")

    current_data, current_label = room2blocks_wrapper_normalized(room_path, NUM_POINT)

    current_data = current_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(current_label)
    # Get room dimension..
    data_label = np.load(room_path)
    data = data_label[:, 0:6]
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        feed_dict = {
            ops["pointclouds_pl"]: current_data[start_idx:end_idx, :, :],
            ops["labels_pl"]: current_label[start_idx:end_idx],
            ops["is_training_pl"]: is_training,
        }
        loss_val, pred_val = sess.run([ops["loss"], ops["pred_softmax"]], feed_dict=feed_dict)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:, :, 0:12], 2)  # BxN
        else:
            pred_label = np.argmax(pred_val, 2)  # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx + b, :, :]
            l = current_label[start_idx + b, :]
            pts[:, 6] *= max_room_x
            pts[:, 7] *= max_room_y
            pts[:, 8] *= max_room_z
            pts[:, 3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                fout_data_label.write(
                    "%f %f %f %d %d %d %f %d\n"
                    % (
                        pts[i, 6],
                        pts[i, 7],
                        pts[i, 8],
                        pts[i, 3],
                        pts[i, 4],
                        pts[i, 5],
                        pred_val[b, i, pred[i]],
                        pred[i],
                    )
                )
        correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
        total_correct += correct
        total_seen += cur_batch_size * NUM_POINT
        loss_sum += loss_val * BATCH_SIZE
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += pred_label[i - start_idx, j] == l

    fout_data_label.close()
    return total_correct, total_seen


if __name__ == "__main__":
    with tf.Graph().as_default():
        evaluate()
