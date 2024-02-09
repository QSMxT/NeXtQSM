#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import osfclient
import tarfile

import argparse
import os
import sys

from pathlib import Path

def main(ckp_path, ckp_name, source_path, mask_path, out_file, b_vec):
    import tensorflow as tf
    from nextqsm.tf_utils import UNet, misc
    from nextqsm.models import varnet, solver_all
    from nextqsm.processing import data_loader, qsm

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


def download_weights():
    current_file_path = os.path.abspath(__file__)
    package_root = os.path.dirname(current_file_path)
    checkpoints_dir = os.path.join(package_root, 'checkpoints')
    checkpoint_file_path = os.path.join(checkpoints_dir, 'checkpoint')

    weights_exist = True
    if os.path.exists(checkpoint_file_path):
        with open(checkpoint_file_path, 'r') as f:
            content = f.readlines()
        
        for line in content:
            checkpoint_name = line.split('"')[1]
            weights_exist &= any(fname.startswith(checkpoint_name) for fname in os.listdir(checkpoints_dir) if os.path.isfile(os.path.join(checkpoints_dir, fname)))
    
    if not weights_exist:
        print('Downloading NeXtQSM weights...')
        osf = osfclient.OSF()
        osf_project = osf.project("zqfdc")
        osf_file = list(osf_project.storage().files)[0]
        tar_path = 'nextqsm-weights.tar'
        with open(tar_path, 'wb') as fpr:
            osf_file.write_to(fpr)

        print("Extracting NeXtQSM weights...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=checkpoints_dir)
        os.remove(tar_path)

        print("Weights downloaded and extracted successfully.")
    else:
        print(f"NeXtQSM weights found in {checkpoints_dir}.")
        

def cli_main():
    if '--download_weights' in sys.argv:
        parser = argparse.ArgumentParser(description="Download weights for NeXtQSM.")
        parser.add_argument('--download_weights', action='store_true', help='Only download the weights and exit')
        args = parser.parse_args()
        
        if args.download_weights:
            download_weights()
            exit()
    
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
    ckp_folder = os.path.join(this_dir, "checkpoints")
    ckp_name = "zdir_calc-HR"
    
    
    main(str(ckp_folder + "/"), str(ckp_name), str(Path(args.phase)), str(Path(args.mask)), str(Path(args.out_file)), args.b_vec)
