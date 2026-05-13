#!/usr/bin/env python3
"""Convert NeXtQSM TensorFlow models to ONNX format."""

import os
import tensorflow as tf
import tf2onnx
import numpy as np

# Add the package to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nextqsm.tf_utils import UNet, misc
from nextqsm.models import varnet

def convert_models(checkpoint_dir, output_dir, input_shape=(64, 64, 64)):
    """
    Convert BF and VN models to ONNX format.

    Args:
        checkpoint_dir: Path to checkpoints folder
        output_dir: Where to save ONNX files
        input_shape: Input volume shape (will be dynamic in ONNX)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load params
    params = misc.load_json(os.path.join(checkpoint_dir, "params.json"))
    print(f"Loaded params: {params}")

    # === Convert BF (Background Field) Network ===
    print("\n=== Converting BF Network ===")
    bf_network = UNet(
        n_classes=1,
        n_layers=params["n_layers"],
        starting_filters=params["starting_filters"],
        k_size=3,
        init=params["kernel_initializer"],
        batch_norm=params["batch_norm"],
        dropout=0.,
        activation=misc.get_act_function(params["act_func"]),
        conv_per_layer=params["conv_per_layer"],
        max_pool=False,
        upsampling=False,
        kernel_regularizer=None
    )

    # Build model with dummy input
    dummy_input = tf.zeros((1, *input_shape, 1), dtype=tf.float32)
    _ = bf_network(dummy_input, training=False)

    # Load weights (prefer H5 format, fall back to legacy checkpoint)
    bf_h5 = os.path.join(checkpoint_dir, "zdir_calc-HRbf-rmse-weights.weights.h5")
    bf_legacy = os.path.join(checkpoint_dir, "zdir_calc-HRbf-rmse-weights")
    bf_network.load_weights(bf_h5 if os.path.exists(bf_h5) else bf_legacy)
    print(f"Loaded BF weights from {bf_h5 if os.path.exists(bf_h5) else bf_legacy}")
    bf_network.summary((*input_shape, 1))

    # Convert to ONNX
    input_signature = [tf.TensorSpec([1, None, None, None, 1], tf.float32, name="input")]
    bf_onnx_path = os.path.join(output_dir, "nextqsm_bf.onnx")

    model_proto, _ = tf2onnx.convert.from_keras(
        bf_network,
        input_signature=input_signature,
        opset=15,
        output_path=bf_onnx_path
    )
    print(f"Saved BF ONNX model to {bf_onnx_path}")

    # === Convert VN (VarNet) Network ===
    print("\n=== Converting VN Network ===")
    vn_network = varnet.VarNet(params)

    # Build model
    _ = vn_network(dummy_input, training=False)

    # Load weights (prefer H5 format, fall back to legacy checkpoint)
    vn_h5 = os.path.join(checkpoint_dir, "zdir_calc-HR-vn-nets.weights.h5")
    vn_legacy = os.path.join(checkpoint_dir, "zdir_calc-HR-vn-rmse-weights")
    if os.path.exists(vn_h5):
        vn_network.nets.load_weights(vn_h5)
        lambdas_npy = np.load(os.path.join(checkpoint_dir, "zdir_calc-HR-vn-lambdas.npy"))
        for i, l in enumerate(lambdas_npy):
            vn_network.lambdas[i].assign(l)
        print(f"Loaded VN weights from {vn_h5}")
    else:
        vn_network.load_weights(vn_legacy)
        lambdas_npy = np.array([l.numpy() for l in vn_network.lambdas])
        print(f"Loaded VN weights from {vn_legacy}")

    # Convert to ONNX
    vn_onnx_path = os.path.join(output_dir, "nextqsm_vn.onnx")

    model_proto, _ = tf2onnx.convert.from_keras(
        vn_network,
        input_signature=input_signature,
        opset=15,
        output_path=vn_onnx_path
    )
    print(f"Saved VN ONNX model to {vn_onnx_path}")

    # === Export VarNet lambdas and params ===
    import json
    lambdas_path = os.path.join(output_dir, "nextqsm_lambdas.json")
    with open(lambdas_path, 'w') as f:
        json.dump({"lambdas": lambdas_npy.tolist()}, f)
    print(f"Saved lambdas to {lambdas_path}")

    params_path = os.path.join(output_dir, "nextqsm_params.json")
    with open(params_path, 'w') as f:
        json.dump({"vn_n_steps": params["vn_n_steps"], "vn_dt_loss": params["vn_dt_loss"]}, f)
    print(f"Saved params to {params_path}")

    print("\n=== Conversion Complete ===")
    print(f"BF model: {bf_onnx_path}")
    print(f"VN model: {vn_onnx_path}")
    print(f"Lambdas:  {lambdas_path}")
    print(f"Params:   {params_path}")

    # Print file sizes
    bf_size = os.path.getsize(bf_onnx_path) / (1024 * 1024)
    vn_size = os.path.getsize(vn_onnx_path) / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  BF: {bf_size:.2f} MB")
    print(f"  VN: {vn_size:.2f} MB")

    return bf_onnx_path, vn_onnx_path


def verify_onnx(onnx_path, input_shape=(64, 64, 64)):
    """Verify ONNX model runs correctly."""
    import onnxruntime as ort

    print(f"\nVerifying {onnx_path}...")
    session = ort.InferenceSession(onnx_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference with random data
    test_input = np.random.randn(1, *input_shape, 1).astype(np.float32)
    result = session.run([output_name], {input_name: test_input})

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {result[0].shape}")
    print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
    print("  Verification passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert NeXtQSM to ONNX")
    parser.add_argument("--checkpoint-dir", default="nextqsm/checkpoints",
                        help="Path to checkpoints directory")
    parser.add_argument("--output-dir", default="onnx_models",
                        help="Output directory for ONNX files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify converted models with onnxruntime")

    args = parser.parse_args()

    bf_path, vn_path = convert_models(args.checkpoint_dir, args.output_dir)

    if args.verify:
        verify_onnx(bf_path)
        verify_onnx(vn_path)
