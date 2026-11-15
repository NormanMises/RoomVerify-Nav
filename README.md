# RoomVerify-Nav

**Room-Aware Frontier Reranking and Proposal–Verification for Zero-Shot Object Navigation**

RoomVerify-Nav is a training-free framework for finding objects in unseen indoor environments. It combines spatial room memory with frontier exploration and verifies detector proposals before committing to a navigation goal.

- **Room-aware exploration:** room distributions inferred from scene objects are accumulated in a spatial memory. Target–room compatibility reranks candidate frontiers with a bounded weighting factor.
- **Proposal–verification:** a vision-language model checks candidate detections on a box-annotated view. Accepted and rejected instance memories avoid repeated verification and maintain a persistent navigation goal.

Authors: Hongfei Yan, Xiangyi Zhu, Yanchao Lin, and Guorui Sheng.

## Installation

Use Linux, an NVIDIA GPU, CUDA 11.8-compatible drivers, Conda, Git, and tmux. The development environment uses Python 3.9, PyTorch 2.0.0, torchvision 0.15.1, and Habitat-Sim 0.2.4. GroundingDINO builds a CUDA extension and requires a compatible CUDA toolkit/compiler.

```bash
git clone https://github.com/NormanMises/RoomVerify-Nav.git
cd RoomVerify-Nav

conda create -n roomverify-nav python=3.9 -y
conda activate roomverify-nav
conda install -c conda-forge -c aihabitat habitat-sim=0.2.4 headless -y

pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install habitat-lab==0.2.420230405 habitat-baselines==0.2.420230405

git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install --no-build-isolation -e ./GroundingDINO
git clone https://github.com/WongKinYiu/yolov7.git

export PYTHONPATH="$PWD"
```

The Python import package remains `vlfm`; the project distribution is named `roomverify-nav`. Run the following commands from the repository root with the environment active.

## Datasets and model weights

Dataset layout and Habitat configuration follow the standard ObjectNav setup. Obtain HM3D or MP3D scene assets through their official access procedures, and download the matching ObjectNav episodes using the [Habitat dataset instructions](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). Keep the scene paths referenced by each episode file valid under `data/scene_datasets/`.

```text
data/
├── datasets/objectnav/
│   ├── hm3d/v1/val/val.json.gz
│   ├── hm3d/v1/val/content/*.json.gz
│   ├── mp3d/v1/val/val.json.gz
│   └── mp3d/v1/val/content/*.json.gz
└── scene_datasets/
    └── ...
```

Download these checkpoints into `data/`:

| File | Source |
| --- | --- |
| `mobile_sam.pt` | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| `groundingdino_swint_ogc.pth` | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `yolov7-e6e.pt` | [YOLOv7](https://github.com/WongKinYiu/yolov7) |
| `pointnav_weights.pth` | [PointNav checkpoint](https://github.com/bdaiinstitute/vlfm/blob/main/data/pointnav_weights.pth) |

Datasets and model weights are downloaded separately and remain subject to their respective terms. BLIP-2 and tokenizer assets may also download on first launch.

Generate Habitat's evaluation placeholder checkpoint:

```bash
mkdir -p data
python -m vlfm.utils.generate_dummy_policy
```

## Model API configuration

```bash
cp .env.example .env
```

The latest recorded experiments (October 18, 2026) used the following API settings:

| Setting | Value |
| --- | --- |
| Text model | `qwen3.8-flash` |
| Vision model | `qwen3.8-flash` |
| Temperature | `0.0` |
| Thinking | Enabled |
| Target-room prior reasoning effort | `medium` |
| Verification reasoning effort | `low` |
| Room inference reasoning effort | `low` |

These values are provided in `.env.example`. Replace the placeholder API key and endpoint with your own credentials and provider deployment URL. The experiments used an Alibaba Cloud Model Studio OpenAI-compatible endpoint. The endpoint must support chat completions, JSON responses, image inputs, and the `enable_thinking`/`thinking` request fields sent by `vlfm/utils/query_llm.py`. Reasoning options must also be supported if configured. API credentials stay in `.env` and must not be committed.

## Run

Start the four local model servers:

```bash
./scripts/launch_vlm_servers.sh
tmux attach -t vlm-servers-12181
```

Wait for all four servers to finish loading. Detach with `Ctrl-b`, then `d`. Default ports are 12181 (GroundingDINO), 12182 (BLIP-2 ITM), 12183 (SAM), and 12184 (YOLOv7).

Run a five-episode HM3D evaluation with room-aware reranking and proposal verification enabled:

```bash
PROMPTING_FRONTIER_RERANK=1 \
FIRST_DETECTION_VERIFY_ENABLED=1 \
TEST_NUM_EPISODES=5 \
./scripts/test_vlfm.sh
```

Feature switches default to off in the script; enable them explicitly as above. Verification defaults to the `toilet,tv,chair` categories. Set `FIRST_DETECTION_VERIFY_CATEGORIES=''` to verify all target categories.

Use `TEST_NUM_EPISODES=-1` for a full evaluation, `VLFM_DATASET=mp3d` for MP3D, or `TEST_EPISODE_IDS=33 TEST_NUM_EPISODES=1` to select an episode. `scripts/test_vlfm.sh` lists runtime switches; `config/experiments/vlfm_objectnav_hm3d.yaml` defines the policy and detection thresholds. LLM settings belong in `.env`.

Evaluation outputs are written under `video/`; `outputs/` contains Hydra run outputs. Video export can be enabled in the experiment YAML via `habitat_baselines.eval.video_option: ["disk"]`.

## Code layout

| Path | Purpose |
| --- | --- |
| `vlfm/policy/itm_policy.py` | Room-aware frontier selection; `ITMPolicyV2` is the default policy |
| `vlfm/policy/base_objectnav_policy.py` | Detection verification, instance memory, and goal management |
| `vlfm/mapping/room_memory_map.py` | Spatial room memory |
| `vlfm/mapping/room_tracker.py` | Room-state tracking |
| `vlfm/utils/target_guidance.py` | Target–room priors |
| `vlfm/vlm/` | Local perception servers and clients |
| `config/` | Habitat and policy configuration |

## License

The project code is distributed under the [MIT License](LICENSE), with existing copyright and license notices preserved. External dependencies and pretrained models retain their own licenses; see [third-party notes](docs/third_party.md).

## Acknowledgments

We thank the authors of [VLFM](https://github.com/bdaiinstitute/vlfm) for their open-source implementation, which provides the foundation for this project.
