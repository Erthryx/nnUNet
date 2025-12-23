from typing import Union, Tuple, List
from copy import deepcopy
import numpy as np

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm


class Fixed4StagePlanner(ExperimentPlanner):
    """
    Forces a 4-stage U-Net with fixed feature widths and number of blocks.
    Everything else (patch size, spacing, batch size, VRAM estimation)
    remains nnUNet-default.
    """

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:

        # -----------------------------
        # Fixed architecture definition
        # -----------------------------
        FIXED_NUM_STAGES = 4
        FIXED_FEATURES_PER_STAGE = (16, 32, 64, 128)
        FIXED_BLOCKS_PER_STAGE = (1, 2, 2, 2)
        FIXED_BLOCKS_DECODER = (2, 2, 2)

        assert len(FIXED_FEATURES_PER_STAGE) == FIXED_NUM_STAGES
        assert len(FIXED_BLOCKS_PER_STAGE) == FIXED_NUM_STAGES
        assert len(FIXED_BLOCKS_DECODER) == FIXED_NUM_STAGES - 1

        # --------------------------------
        # Determine basic dataset settings
        # --------------------------------
        num_input_channels = len(
            self.dataset_json['channel_names']
            if 'channel_names' in self.dataset_json
            else self.dataset_json['modality']
        )

        num_classes = len(self.dataset_json['labels'])
        unet_conv_op = convert_dim_to_conv_op(len(spacing))
        norm_op = get_matching_instancenorm(unet_conv_op)

        # ----------------------------------------------------
        # Initial patch size (same logic as default planner)
        # ----------------------------------------------------
        tmp = 1 / np.array(spacing)

        if len(spacing) == 3:
            initial_patch_size = [
                round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)
            ]
        elif len(spacing) == 2:
            initial_patch_size = [
                round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)
            ]
        else:
            raise RuntimeError("Unsupported dimensionality")

        initial_patch_size = np.array([
            min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])
        ])

        # ----------------------------------------------------
        # Infer pooling topology (we will truncate to 4 stages)
        # ----------------------------------------------------
        (
            _,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            patch_size,
            shape_must_be_divisible_by
        ) = get_pool_and_conv_props(
            spacing,
            initial_patch_size,
            self.UNet_featuremap_min_edge_length,
            999999
        )

        if len(pool_op_kernel_sizes) < FIXED_NUM_STAGES:
            raise RuntimeError(
                f"Dataset only allows {len(pool_op_kernel_sizes)} stages, "
                f"but Fixed4StagePlanner requires 4."
            )

        # -------------------------
        # FORCE 4 STAGES HERE
        # -------------------------
        pool_op_kernel_sizes = pool_op_kernel_sizes[:FIXED_NUM_STAGES]
        conv_kernel_sizes = conv_kernel_sizes[:FIXED_NUM_STAGES]

        # Ensure patch size divisibility for 4 stages
        required_divisibility = np.prod(pool_op_kernel_sizes, axis=0)
        patch_size = [
            int(np.floor(p / d) * d)
            for p, d in zip(patch_size, required_divisibility)
        ]

        # -------------------------
        # Architecture definition
        # -------------------------
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': FIXED_NUM_STAGES,
                'features_per_stage': FIXED_FEATURES_PER_STAGE,
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'n_conv_per_stage': FIXED_BLOCKS_PER_STAGE,
                'n_conv_per_stage_decoder': FIXED_BLOCKS_DECODER,
                'conv_bias': True,
                'norm_op': norm_op.__module__ + '.' + norm_op.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }

        # -------------------------
        # VRAM estimation (unchanged)
        # -------------------------
        estimate = self.static_estimate_VRAM_usage(
            patch_size,
            num_input_channels,
            num_classes,
            architecture_kwargs['network_class_name'],
            architecture_kwargs['arch_kwargs'],
            architecture_kwargs['_kw_requires_import']
        )

        reference = (
            self.UNet_reference_val_2d if len(spacing) == 2
            else self.UNet_reference_val_3d
        ) * (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        ref_bs = (
            self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2
            else self.UNet_reference_val_corresp_bs_3d
        )

        batch_size = round((reference / estimate) * ref_bs)
        batch_size = max(batch_size, self.UNet_min_batch_size)

        # -------------------------
        # Resampling & normalization
        # -------------------------
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = \
            self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = \
            self.determine_segmentation_softmax_export_fn()
        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        # -------------------------
        # Final plan dictionary
        # -------------------------
        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }

        return plan
