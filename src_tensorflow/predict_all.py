#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import nibabel as nib

from tf_utils import UNet, misc
from models import varnet, solver_all
from processing import data_loader, qsm

import argparse
import os
from pathlib import Path

def main(ckp_path, ckp_name, source_path, mask_path, out_file, b_vec):
    base_path = misc.get_base_path()
    #ckp_path = base_path + ckp_path
    params = misc.load_json(ckp_path + "params.json")

    if source_path is None:
        print("Loading 2019 simulated data...")
        datasets = data_loader.get_QSM_2019_simulated(base_path + "dataset/QSM_2019_Simulated/")
    else:
        print(f"Loading phase data from {source_path}, mask data from {mask_path} ...")
        datasets, meta = data_loader.load_testing_volume({"source": source_path, "mask": mask_path})
        datasets[list(datasets.keys())[0]][0]["freq"] = datasets[list(datasets.keys())[0]][0]["source"]
        datasets[list(datasets.keys())[0]][0]["chi"] = datasets[list(datasets.keys())[0]][0]["source"]

    # BF
    bf_network = UNet(1, params["n_layers"], params["starting_filters"], 3, params["kernel_initializer"], params["batch_norm"],
                      0., misc.get_act_function(params["act_func"]), params["conv_per_layer"], False, False, None)
    bf_network.load_weights(ckp_path + "zdir_calc-HRbf-rmse-weights")
    bf_network.summary((64, 64, 64, 1))

    # VN
    vn = varnet.VarNet(params)
    vn.load_weights(ckp_path + "zdir_calc-HR-vn-rmse-weights")

    slv = solver_all.Solver(params)
    kernels = {}
    for key in datasets:

        # Kernel
        voxel_size = meta['header'].get_zooms()
        op = qsm.QSM(voxel_size=voxel_size, b_vec=b_vec)
        kernels[key] = op.get_dipole_kernel_fourier(datasets[key][0]["source"].shape[1:-1])

        # Run
        x_init = tf.Variable(tf.zeros_like(datasets[key][0]["source"]), name='x', trainable=True, dtype=tf.float32)
        bf_logits, vn_logits, _ = slv.test_step(bf_model=bf_network, model=vn, x=x_init, source=datasets[key][0]["source"], freq=datasets[key][0]["freq"], chi=datasets[key][0]["chi"], mask=datasets[key][0]["mask"], kernel=kernels[key], weight_vn=params["weight_vn"])
        
        if source_path is None:
            nib.save(nib.Nifti1Image(bf_logits.numpy()[0, :, :, :, 0], np.eye(4)), ckp_path + str(ckp_name) + "BF-ALL-RMSE-" + key + ".nii.gz")
            nib.save(nib.Nifti1Image(vn_logits.numpy()[0, :, :, :, 0], np.eye(4)), ckp_path + str(ckp_name) + "-ALL-RMSE-" + key + ".nii.gz")
        else:
            if "pad_added" in meta:
                bf_logits = misc.remove_padding(bf_logits, meta["orig_shape"], meta["pad_added"])
                vn_logits = misc.remove_padding(vn_logits, meta["orig_shape"], meta["pad_added"])
            print(f'Save results as {out_file}')
            nib.save(nib.Nifti1Image(vn_logits[0, :, :, :, 0], meta["affine"], meta["header"]), out_file)
            nib.save(nib.Nifti1Image(bf_logits[0, :, :, :, 0], meta["affine"], meta["header"]), out_file.replace(".nii.gz", "") + "_BF.nii.gz")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="NeXtQSM: Deep Learning QSM Algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'phase',
        help='Unwrapped and normalized phase image'
    )
    parser.add_argument(
        'mask',
        help='Brain mask'
    )
    parser.add_argument(
        'out_file',
        help='QSM output end with .nii.gz'
    )
    parser.add_argument(
        '--b_vec', nargs=3, type=float,
        default=(0, 0, 1),
        help='B vector, set to axial coordinate by default'
    )
    
    args = parser.parse_args()
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ckp_folder = os.path.join(this_dir, "..", "checkpoints")
    ckp_name = "zdir_calc-HR"
    
    
    main(str(ckp_folder + "/"), str(ckp_name), str(Path(args.phase)), str(Path(args.mask)), str(Path(args.out_file)), args.b_vec)
