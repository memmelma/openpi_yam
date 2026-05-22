"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.yam_policy as yam_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If set, load sidecar reward arrays under
    # ``<lerobot_home or $HF_LEROBOT_HOME>/<repo_id>/meta/rewards/<reward_name>/episode_*.npy``
    # and emit a per-sample scalar ``weight`` for reward-weighted BC. ``None``
    # leaves the data pipeline (and loss) unchanged.
    reward_name: str | None = None

    # If set, LeRobot datasets (and reward sidecars when ``reward_name`` is set) load from
    # ``<lerobot_home>/<repo_id>/`` instead of ``$HF_LEROBOT_HOME/<repo_id>/``. Use this when
    # ``HF_LEROBOT_HOME`` cannot be changed (it is read once at LeRobot import time).
    lerobot_home: str | None = None

    # ----- AWR / weighted-BC reward shaping (see openpi.training.rewards) -----
    # AWR temperature: weight = exp(advantage / reward_beta).  Lower beta = sharper.
    reward_beta: float = 2.0
    # Weighting scheme applied to the stored reward/value sidecar:
    #   "default"   – use stored values directly as weights (no transform)
    #   "awr"       – exp(value / reward_beta)  [AWR]
    #   "chunk"     – V(s_{t+N}) - V(s_t); stored data must be V values, N = action_horizon
    #   "awr_chunk" – exp((V(s_{t+N}) - V(s_t)) / reward_beta)
    weight_scheme: str = "awr"
    # If set, drop the bottom ``weight_quantile * 100`` percent of frames
    # globally (across the whole repo) before training.  ``0.8`` keeps the top
    # 20% of frames.  Combined with ``weight_cutoff`` via ``max(...)``.
    weight_quantile: float | None = None
    # Absolute weight cutoff: frames with weight below this value are dropped.
    weight_cutoff: float | None = None
    # If True, replace every sample's ``prompt`` with the
    # ``reference_instruction`` recorded in the reward annotation's
    # ``config.json``.  Useful when datasets were collected with different
    # task strings but you want a single instruction during reward-weighted BC.
    override_prompt_from_reward: bool = False

    # ----- CFGRL / π*0.6 advantage-conditioned policy extraction ------------
    # When True, use CFGRL instead of AWR weighting: a binary "Advantage:
    # positive/negative" indicator is appended to the prompt based on the
    # chunk advantage V(s_{t+H})-V(s_t) from the reward sidecar.  Requires
    # ``reward_name`` to point at a *value function* directory (reward_name
    # must contain "reward").  ``weight_scheme`` is ignored when this is True
    # (internally forced to "chunk").
    cfgrl_enabled: bool = False
    # Fraction of frames labelled positive (top-k by advantage).
    # Paper uses 0.30 for pre-training, 0.40 for fine-tuning.
    cfgrl_positive_quantile: float = 0.30
    # Probability of omitting the indicator entirely (unconditional branch).
    # Enables classifier-free guidance at inference time.
    cfgrl_dropout_prob: float = 0.30
    # When True, always inject "Advantage: positive" regardless of advantage
    # value.  Use during SFT / pure-demonstration fine-tune phases.
    cfgrl_force_positive: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotYAMDataConfig(DataConfigFactory):
    """Config for YAM bimanual robot dataset.

    State/action: 14D joint positions [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
    Cameras: head, left_wrist, right_wrist
    Actions are absolute joint targets; converted to delta (except grippers) for training.
    """

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack: map LeRobot dataset keys to the keys expected by YAMInputs.
        repack_structure: dict[str, str] = {
            "observation/image_head": "observation.images.head",
            "observation/image_left_wrist": "observation.images.left_wrist",
            "observation/image_right_wrist": "observation.images.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if (self.base_config or DataConfig()).reward_name is not None:
            # Preserve the scalar injected by AddRewardWeight through repack.
            repack_structure["weight"] = "weight"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        data_transforms = _transforms.Group(
            inputs=[yam_policy.YAMInputs(model_type=model_config.model_type)],
            outputs=[yam_policy.YAMOutputs()],
        )

        # Convert absolute joint actions to deltas (grippers stay absolute).
        # 14D: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [

    # weighted BC training - tillicum
    TrainConfig(
        name="weighted_bc_tillicum_star_wars",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/star_wars_shelf_box_switch",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="default_advantage",
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=30_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # weighted BC training - local
    TrainConfig(
        name="weighted_bc_local_star_wars",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="local/raiden_star_wars_bookshelf",
            base_config=DataConfig(prompt_from_task=True, reward_name="rvlm_advantage"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),


    # AWR chunk-weighted BC - local (raw rewards as V, no filtering)
    TrainConfig(
        name="awr_chunk_swb_local",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                reward_beta=2.0,
                weight_scheme="awr_chunk",
                weight_quantile=None,
                weight_cutoff=None,
                override_prompt_from_reward=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # CFGRL advantage-conditioned BC - local smoke test (same V sidecar as awr_chunk_swb_local)
    TrainConfig(
        name="cfgrl_swb_local_smoke",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                override_prompt_from_reward=True,
                cfgrl_enabled=True,
                cfgrl_positive_quantile=0.30,
                cfgrl_dropout_prob=0.30,
                cfgrl_force_positive=False,
                weight_quantile=None,
                weight_cutoff=None,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=500,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # success only BC training - local
    TrainConfig(
        name="bc_swb_local",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                weight_quantile=None,
                weight_cutoff=0.5,
                weight_scheme="default",
                reward_name="success_reward",
                override_prompt_from_reward=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),


    # success only BC training - tillicum
    TrainConfig(
        name="bc_swb_bimanual",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_bimanual_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                weight_quantile=None,
                weight_cutoff=0.5,
                weight_scheme="default",
                reward_name="success_reward",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # success only BC training - tillicum
    TrainConfig(
        name="bc_swb",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                weight_quantile=None,
                weight_cutoff=0.5,
                weight_scheme="default",
                reward_name="success_reward",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # success only AWR training - tillicum
    TrainConfig(
        name="awr_rvlm_delta_advantage_swb",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_g099_delta_advantage",
                reward_beta=2.0,
                weight_quantile=0.7,
                weight_cutoff=None,
                weight_scheme="awr",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # success only AWR training - tillicum
    TrainConfig(
        name="awr_rvlm_delta_advantage_swb_quantile_06",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_g099_delta_advantage",
                reward_beta=2.0,
                weight_quantile=0.6,
                weight_cutoff=None,
                weight_scheme="awr",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # AWR chunk-weighted BC - tillicum (raw rewards as V, no filtering)
    TrainConfig(
        name="awr_chunk_swb_tillicum",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                reward_beta=2.0,
                weight_scheme="awr_chunk",
                weight_quantile=None,
                weight_cutoff=None,
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # success only AWR training - tillicum
    TrainConfig(
        name="awr_rvlm_delta_advantage_swb_quantile_05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_g099_delta_advantage",
                reward_beta=2.0,
                weight_quantile=0.5,
                weight_cutoff=None,
                weight_scheme="awr",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # CFGRL advantage-conditioned BC - tillicum
    TrainConfig(
        name="cfgrl_swb_tillicum",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/swb_joint_05_20",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_star_wars_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                override_prompt_from_reward=True,
                cfgrl_enabled=True,
                cfgrl_positive_quantile=0.30,
                cfgrl_dropout_prob=0.30,
                cfgrl_force_positive=False,
                weight_quantile=None,
                weight_cutoff=None,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # AWR chunk-weighted BC - red_book_05_21
    TrainConfig(
        name="awr_chunk_red_book",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/red_book_05_21",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_red_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                reward_beta=2.0,
                weight_scheme="awr_chunk",
                weight_quantile=None,
                weight_cutoff=None,
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # CFGRL advantage-conditioned BC - red_book_05_21
    TrainConfig(
        name="cfgrl_red_book",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/red_book_05_21",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="move_the_red_book_from_the_book_shelf_to_the_gray_box_rvlm_reward",
                override_prompt_from_reward=True,
                cfgrl_enabled=True,
                cfgrl_positive_quantile=0.30,
                cfgrl_dropout_prob=0.30,
                cfgrl_force_positive=False,
                weight_quantile=None,
                weight_cutoff=None,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # plain BC - red_book_05_21_bc (success-filtered)
    TrainConfig(
        name="bc_red_book",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/red_book_05_21_bc",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="success_reward",
                weight_quantile=None,
                weight_cutoff=0.5,
                weight_scheme="default",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # AWR chunk-weighted BC - pick_and_choose_05_21
    TrainConfig(
        name="awr_chunk_pick_and_choose",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/pick_and_choose_05_21",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="put_the_daniel_kahnemann_book_on_the_bookshelf_rvlm_reward",
                reward_beta=2.0,
                weight_scheme="awr_chunk",
                weight_quantile=None,
                weight_cutoff=None,
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # CFGRL advantage-conditioned BC - pick_and_choose_05_21
    TrainConfig(
        name="cfgrl_pick_and_choose",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/pick_and_choose_05_21",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="put_the_daniel_kahnemann_book_on_the_bookshelf_rvlm_reward",
                override_prompt_from_reward=True,
                cfgrl_enabled=True,
                cfgrl_positive_quantile=0.30,
                cfgrl_dropout_prob=0.30,
                cfgrl_force_positive=False,
                weight_quantile=None,
                weight_cutoff=None,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # plain BC - pick_and_choose_05_21_bc (success-filtered)
    TrainConfig(
        name="bc_pick_and_choose",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            repo_id="memmelma/pick_and_choose_05_21_bc",
            base_config=DataConfig(
                prompt_from_task=True,
                reward_name="success_reward",
                weight_quantile=None,
                weight_cutoff=0.5,
                weight_scheme="default",
                override_prompt_from_reward=True,
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),

    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
