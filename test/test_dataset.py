from dataset import FlowDataset

import argparse

def test_dataset(data_folder):
    dataset = FlowDataset(data_folder)

    for data_point in dataset:
        name, image = data_point 
        print(name)
        print(image.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    args = parser.parse_args()

    test_dataset(args.data_folder)