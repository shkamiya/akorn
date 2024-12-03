# provided by @loeweX
import os
from typing import List

import numpy as np
from PIL import Image
from einops import rearrange
import argparse



from tfloaders import (
    clevr_with_masks,
    multi_dsprites,
    tetrominoes,
)

def resize_input(x):
    # Center-crop with boundaries [29, 221] (height) and [64, 256] (width).
    x = x[29:221, 64:256]
    # Resize cropped image to 128x128 resolution.
    return np.array(
        Image.fromarray(x).resize((128, 128), resample=Image.Resampling.NEAREST)
    )

def resize_input_tetrominoes(x):
    # Center-crop 
    x = x[1:33, 1:33]
    return x


def get_hparams(dataset_name):

    pdir = './'

    if dataset_name == "multi_dsprites":
        variant = "colored_on_grayscale"  # binarized, colored_on_colored
        input_path = f"{pdir}/multi_dsprites/multi_dsprites_{variant}.tfrecords"
        output_path = f"{pdir}/multi_dsprites/"

        dataset = multi_dsprites.dataset(input_path, variant)
        train_size = 60000
        dataset_name = variant

    elif dataset_name == "tetrominoes":
        input_path = f"{pdir}/tetrominoes/tetrominoes_train.tfrecords"
        output_path = f"{pdir}/tetrominoes/"

        dataset = tetrominoes.dataset(input_path)
        train_size = 60000

    elif dataset_name in ["clevr_with_masks"]:
        input_path = f"{pdir}/clevr_with_masks/clevr_with_masks_train.tfrecords"
        output_path = f"{pdir}/clevr_with_masks/"

        dataset = clevr_with_masks.dataset(input_path)
        train_size = 70000

    val_size = 10000  # 5000
    test_size = 320
    eval_size = 64
    return (
        input_path,
        output_path,
        dataset,
        train_size,
        val_size,
        test_size,
        eval_size,
        dataset_name,
    )
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert TFRecord to NPZ")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["multi_dsprites", "tetrominoes", "clevr_with_masks"],
        help="Name of the dataset to convert",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    clevr6 = True

    (
        input_path,
        output_path,
        dataset,
        train_size,
        val_size,
        test_size,
        eval_size,
        dataset_name,
    ) = get_hparams(dataset_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    batched_dataset = dataset.batch(1)
    iterator = iter(batched_dataset)

    counter = 0
    images: List[np.array] = []
    labels: List[np.array] = []


    while True:
        try:
            data = next(iterator)
        except StopIteration:
            break

        input_image = np.squeeze(data["image"].numpy())

        pixelwise_label = np.zeros(
            (1, input_image.shape[0], input_image.shape[1]), dtype=np.uint8
        )
        for idx in range(data["mask"].shape[1]):
            pixelwise_label[
                np.where(data["mask"].numpy()[:, idx, :, :, 0] == 255)
            ] = idx

        pixelwise_label = np.squeeze(pixelwise_label)

        if dataset_name in ["clevr_with_masks"]:
            input_image = resize_input(input_image)
            pixelwise_label = resize_input(pixelwise_label)

            if clevr6 and np.max(pixelwise_label) > 6:
                # CLEVR6: only use images with maximally 6 objects
                continue

        if dataset_name in ["tetrominoes"]:
            input_image = resize_input_tetrominoes(input_image)
            pixelwise_label = resize_input_tetrominoes(pixelwise_label)

        input_image = rearrange(input_image, "h w c -> c h w")
        input_image = (input_image / 255)

        # pixelwise_label = rearrange(pixelwise_label, "w h -> h w")

        images.append(input_image)
        labels.append(pixelwise_label)

        counter += 1

        if counter % 1000 == 0:
            print(counter)

        if counter % (train_size + val_size + test_size + eval_size) == 0:
            break

    print("Save files")

    np.savez_compressed(
        os.path.join(output_path, f"{dataset_name}_eval"),
        images=np.squeeze(np.array(images[:eval_size])),
        labels=np.squeeze(np.array(labels[:eval_size])),
    )

    start_idx = eval_size
    np.savez_compressed(
        os.path.join(output_path, f"{dataset_name}_test"),
        images=np.squeeze(np.array(images[start_idx : start_idx + test_size])),
        labels=np.squeeze(np.array(labels[start_idx : start_idx + test_size])),
    )

    start_idx += test_size
    np.savez_compressed(
        os.path.join(output_path, f"{dataset_name}_val"),
        images=np.squeeze(np.array(images[start_idx : start_idx + val_size])),
        labels=np.squeeze(np.array(labels[start_idx : start_idx + val_size])),
    )

    start_idx += val_size
    np.savez_compressed(
        os.path.join(output_path, f"{dataset_name}_train"),
        images=np.squeeze(np.array(images[start_idx:])),
        labels=np.squeeze(np.array(labels[start_idx:])),
    )

    print(f"Train dataset size: {len(images[start_idx:])}")
                                                                
                                                                                                        
