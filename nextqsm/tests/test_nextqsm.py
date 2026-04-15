import pytest
import numpy as np
import nibabel as nib
import os
import tempfile


@pytest.fixture
def phantom_field(tmp_path):
    """Generate a simple susceptibility phantom and forward field using qsm-forward."""
    import qsm_forward

    # Create a small susceptibility phantom (64^3 so no padding needed by nextqsm)
    chi = qsm_forward.generate_susceptibility_phantom(
        resolution=[64, 64, 64],
        background=0,
        large_cylinder_val=0.005,
        small_cylinder_radii=[4, 4],
        small_cylinder_vals=[0.1, 0.2],
    )

    # Generate a brain-like mask (non-zero susceptibility region + large cylinder)
    mask = (chi != 0).astype(np.float32)

    # Generate the local field map (in ppm) via forward convolution
    field = qsm_forward.generate_field(chi, mask=mask, voxel_size=[1, 1, 1], B0_dir=[0, 0, 1])
    field = field.astype(np.float32)

    # Save as NIfTI files
    affine = np.eye(4)
    field_path = str(tmp_path / "field.nii.gz")
    mask_path = str(tmp_path / "mask.nii.gz")
    chi_path = str(tmp_path / "chi.nii.gz")

    nib.save(nib.Nifti1Image(field, affine), field_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    nib.save(nib.Nifti1Image(chi.astype(np.float32), affine), chi_path)

    return {
        "field_path": field_path,
        "mask_path": mask_path,
        "chi_path": chi_path,
        "chi": chi,
        "field": field,
        "mask": mask,
        "tmp_path": tmp_path,
    }


class TestImports:
    """Test that all nextqsm modules can be imported."""

    def test_import_nextqsm(self):
        import nextqsm

    def test_import_predict_all(self):
        from nextqsm import predict_all

    def test_import_data_loader(self):
        from nextqsm.processing import data_loader

    def test_import_qsm(self):
        from nextqsm.processing import qsm


class TestDataLoader:
    """Test the data loading and preprocessing pipeline."""

    def test_load_testing_volume(self, phantom_field):
        from nextqsm.processing import data_loader

        paths = {
            "source": phantom_field["field_path"],
            "mask": phantom_field["mask_path"],
        }
        datasets, meta = data_loader.load_testing_volume(paths)

        # Check metadata
        assert "affine" in meta
        assert "header" in meta
        assert "orig_shape" in meta
        assert meta["orig_shape"] == (64, 64, 64)

        # Check dataset structure
        assert len(datasets) == 1
        key = list(datasets.keys())[0]
        assert "source" in datasets[key][0]
        assert "mask" in datasets[key][0]

    def test_load_volume_shape(self, phantom_field):
        from nextqsm.processing import data_loader

        paths = {
            "source": phantom_field["field_path"],
            "mask": phantom_field["mask_path"],
        }
        datasets, meta = data_loader.load_testing_volume(paths)

        key = list(datasets.keys())[0]
        source = datasets[key][0]["source"]
        mask = datasets[key][0]["mask"]

        # Should be 5D: (1, x, y, z, 1)
        assert len(source.shape) == 5
        assert source.shape[0] == 1
        assert source.shape[-1] == 1
        assert len(mask.shape) == 5

    def test_load_volume_no_mask(self, phantom_field):
        """When mask is None, data_loader should create an all-ones mask."""
        from nextqsm.processing import data_loader
        import tensorflow as tf

        paths = {
            "source": phantom_field["field_path"],
            "mask": None,
        }
        datasets, meta = data_loader.load_testing_volume(paths)

        key = list(datasets.keys())[0]
        mask = datasets[key][0]["mask"]
        assert tf.reduce_all(mask == 1.0)

    def test_load_volume_applies_mask(self, phantom_field):
        """Source data should be multiplied by the mask."""
        from nextqsm.processing import data_loader

        paths = {
            "source": phantom_field["field_path"],
            "mask": phantom_field["mask_path"],
        }
        datasets, _ = data_loader.load_testing_volume(paths)

        key = list(datasets.keys())[0]
        source = datasets[key][0]["source"]
        mask = datasets[key][0]["mask"]

        # Where mask is 0, source should be 0
        zero_mask = mask == 0
        assert np.allclose(source[zero_mask], 0.0)


class TestQSMOperations:
    """Test the QSM dipole kernel and forward operations."""

    def test_dipole_kernel_shape(self):
        from nextqsm.processing.qsm import QSM

        op = QSM(voxel_size=(1, 1, 1), b_vec=(0, 0, 1))
        kernel = op.get_dipole_kernel_fourier((64, 64, 64))

        assert kernel.shape == (64, 64, 64)
        assert kernel.dtype == np.float32

    def test_dipole_kernel_symmetry(self):
        from nextqsm.processing.qsm import QSM

        op = QSM(voxel_size=(1, 1, 1), b_vec=(0, 0, 1))
        kernel = op.get_dipole_kernel_fourier((64, 64, 64))

        # Kernel should have zero mean (approximately 1/3 - 1/3 = 0 at DC)
        assert abs(np.mean(kernel)) < 0.1

    def test_forward_operation(self):
        import tensorflow as tf
        from nextqsm.processing.qsm import QSM

        op = QSM(voxel_size=(1, 1, 1), b_vec=(0, 0, 1))
        kernel = op.get_dipole_kernel_fourier((64, 64, 64))

        # Create a simple 5D input
        y = tf.zeros((1, 64, 64, 64, 1), dtype=tf.float32)
        result = op.forward_operation_fourier(y, kernel)

        assert result.shape == (1, 64, 64, 64, 1)
        assert np.allclose(result.numpy(), 0.0)


class TestPaddingUtils:
    """Test padding and unpadding utilities."""

    def test_get_how_much_to_pad_exact(self):
        from nextqsm.tf_utils.misc import get_how_much_to_pad

        pad, is_same = get_how_much_to_pad((64, 64, 64), 64)
        assert pad == [64, 64, 64]
        assert is_same is True

    def test_get_how_much_to_pad_needs_padding(self):
        from nextqsm.tf_utils.misc import get_how_much_to_pad

        pad, is_same = get_how_much_to_pad((50, 50, 50), 64)
        assert pad == [64, 64, 64]
        assert is_same is False

    def test_add_remove_padding_roundtrip(self):
        from nextqsm.tf_utils.misc import add_padding, remove_padding, get_how_much_to_pad

        orig_shape = (50, 50, 50)
        data = np.random.rand(1, *orig_shape, 1).astype(np.float32)

        pad_size, _ = get_how_much_to_pad(orig_shape, 64)
        padded, orig, pad_added = add_padding(data, pad_size)

        assert padded.shape == (1, 64, 64, 64, 1)

        recovered = remove_padding(padded, orig, pad_added)
        assert recovered.shape[1:4] == orig_shape
        np.testing.assert_allclose(recovered[0, :, :, :, 0], data[0, :, :, :, 0], atol=1e-7)


class TestEndToEnd:
    """End-to-end test: generate phantom, run nextqsm, verify output."""

    @pytest.mark.slow
    def test_nextqsm_produces_output(self, phantom_field):
        from nextqsm.predict_all import main
        import os

        out_file = str(phantom_field["tmp_path"] / "qsm_output.nii.gz")

        this_dir = os.path.dirname(os.path.abspath(__file__))
        ckp_folder = os.path.join(os.path.dirname(this_dir), "checkpoints")
        ckp_path = ckp_folder + "/"

        # Skip if weights are not downloaded
        if not os.path.exists(os.path.join(ckp_folder, "checkpoint")):
            pytest.skip("NeXtQSM weights not downloaded (run: nextqsm --download_weights)")

        main(
            ckp_path=ckp_path,
            ckp_name="zdir_calc-HR",
            source_path=phantom_field["field_path"],
            mask_path=phantom_field["mask_path"],
            out_file=out_file,
            b_vec=(0, 0, 1),
        )

        # Check output exists
        assert os.path.exists(out_file)

        # Load and validate output
        qsm_img = nib.load(out_file)
        qsm_data = qsm_img.get_fdata()

        # Output should have same spatial dims as input
        assert qsm_data.shape == (64, 64, 64)

        # Output should not be all zeros (model should produce non-trivial output)
        assert not np.allclose(qsm_data, 0.0)

        # Output should be finite
        assert np.all(np.isfinite(qsm_data))

    @pytest.mark.slow
    def test_nextqsm_output_correlates_with_ground_truth(self, phantom_field):
        """QSM output should have some positive correlation with the ground truth chi."""
        from nextqsm.predict_all import main
        import os

        out_file = str(phantom_field["tmp_path"] / "qsm_corr.nii.gz")

        this_dir = os.path.dirname(os.path.abspath(__file__))
        ckp_folder = os.path.join(os.path.dirname(this_dir), "checkpoints")
        ckp_path = ckp_folder + "/"

        if not os.path.exists(os.path.join(ckp_folder, "checkpoint")):
            pytest.skip("NeXtQSM weights not downloaded (run: nextqsm --download_weights)")

        main(
            ckp_path=ckp_path,
            ckp_name="zdir_calc-HR",
            source_path=phantom_field["field_path"],
            mask_path=phantom_field["mask_path"],
            out_file=out_file,
            b_vec=(0, 0, 1),
        )

        qsm_data = nib.load(out_file).get_fdata()
        chi = phantom_field["chi"]
        mask = phantom_field["mask"]

        # Correlation within the mask should be positive
        qsm_masked = qsm_data[mask > 0]
        chi_masked = chi[mask > 0]
        correlation = np.corrcoef(qsm_masked, chi_masked)[0, 1]
        assert correlation > 0, f"Expected positive correlation with ground truth, got {correlation}"
