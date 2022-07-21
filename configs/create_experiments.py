import math
import os
import signal
import subprocess
import sys
import itertools
from pprint import pprint

import configargparse


def CreateIndexListSparseUniform(total_size, target_size):
    print("Creating Index List for", total_size, "images with target:", target_size)
    target_size = min(target_size, total_size);
    step = total_size / target_size;
    print("step", step)
    train_indices = []
    eval_indices = []
    for i in range(target_size):
        x = round(i * step)
        y = round((i + 0.5) * step)
        assert (y < total_size)
        train_indices.append(x)
        eval_indices.append(y)
    return train_indices, eval_indices
    print(train_indices)
    print(eval_indices)


def CreateIndexListLimitedAngle(total_size, target_angle):
    print("Creating Index List for", total_size, "images with target angle:", target_angle)

    reduced_input, unused = CreateIndexListSparseUniform(total_size, min(361, total_size))
    reduced_size = len(reduced_input)

    target_angle = target_angle / 360 * 2 * math.pi
    train_indices = []
    eval_indices = []
    for j in range(reduced_size):
        i = reduced_input[j]
        ang = i / (total_size - 1) * 2 * math.pi
        if ang <= target_angle + 1e-6:
            train_indices.append(i)

    train2, eval2 = CreateIndexListSparseUniform(reduced_size, 60)
    for j in eval2:
        i = reduced_input[j]
        ang = i / (total_size - 1) * 2 * math.pi
        if not i in train_indices and ang > target_angle + 1e-6:
            eval_indices.append(i)

    return train_indices, eval_indices
    print(train_indices)
    print(eval_indices)


def SaveIndexList(out_dir, indices):
    os.makedirs(out_dir, exist_ok=True)
    train_indices, eval_indices = indices
    grp = [(train_indices, "train"), (eval_indices, "eval")]

    for indices, name in grp:
        print(indices, name)

        with open(out_dir + "/" + name + ".txt", 'w') as f:
            for i in indices:
                f.write(str(i) + "\n")


def CreateSettings(scene_dir):
    with open(scene_dir + "/poses.txt") as f:
        lines = f.read().splitlines()
    num_images = len(lines)

    SaveIndexList(scene_dir + "/exp_uniform_1", CreateIndexListSparseUniform(num_images, 1))
    SaveIndexList(scene_dir + "/exp_uniform_11", CreateIndexListSparseUniform(num_images, 11))
    SaveIndexList(scene_dir + "/exp_uniform_25", CreateIndexListSparseUniform(num_images, 25))
    SaveIndexList(scene_dir + "/exp_uniform_50", CreateIndexListSparseUniform(num_images, 50))
    SaveIndexList(scene_dir + "/exp_uniform_73", CreateIndexListSparseUniform(num_images, 73))
    SaveIndexList(scene_dir + "/exp_uniform_100", CreateIndexListSparseUniform(num_images, 100))

    SaveIndexList(scene_dir + "/exp_limited_angle_80", CreateIndexListLimitedAngle(num_images, 80))
    SaveIndexList(scene_dir + "/exp_limited_angle_120", CreateIndexListLimitedAngle(num_images, 120))


if __name__ == "__main__":
    # scene_names = ["Chest", "Fan", "Fruit", "pepper", "Plastic_flower", "RopeBall", "star", "Textile_flower", "Ball_synthetic", "Flower_synthetic"]
    # scene_names = ["pomegranate", "toy_car", "monument", "marine_decoration"]
    # scene_names = ["orange_synthetic"]
    # for n in scene_names:
    #     CreateSettings("/home/dari/Projects/HyperAcorn/scenes/" + n)
    # exit(0)

    p = configargparse.ArgumentParser()
    p.add_argument('--exe', required=True, type=str)
    p.add_argument('--base_config', required=True, type=str)
    p.add_argument('--debug', default=False, type=bool)
    p.add_argument('--num_clients', default=1, type=int)
    p.add_argument('--client_id', default=0, type=int)

    opt = p.parse_args()
    # for k, v in opt.__dict__.items():
    #     print(k, v)

    main(opt)
