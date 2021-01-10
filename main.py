import numpy as np 
import cv2
import argparse

from stereoVO.configs import yaml_parser
from stereoVO.datasets import KittiDataset
from stereoVO.utils import rmse_error
from stereoVO.model import StereoVO

def parse_argument():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/params.yaml')
    return parser.parse_args()

def main():

    args = parse_argument()

    # Load the config file
    params = yaml_parser(args.config_path)

    # Get data params using the dataloader
    dataset = KittiDataset(params.dataset.path)
    num_frame = len(dataset)
    cameraMatrix = dataset.intrinsic
    projectionMatrixL = dataset.PL
    projectionMatrixR = dataset.PR

    # Iniitialise the StereoVO model
    model = StereoVO(cameraMatrix, projectionMatrixL, projectionMatrixR, params)

    # Iterate over the frame and update the rotation and translation vectors

    for index in range(2):
        left_frame, right_frame, _ = dataset[index]

        pred_location, pred_orientation = model(left_frame, right_frame, index)

        print(index)
        print("----------------------------------------")
        print(pred_location)
        print(pred_orientation)


if __name__ == "__main__":
    main()