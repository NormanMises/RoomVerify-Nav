# Third-party components

The root MIT license covers this project's code subject to preserved notices. It does not replace the licenses or access terms of external components, model checkpoints, or datasets.

- [VLFM](https://github.com/rai-opensource/vlfm): upstream implementation, MIT. Its copyright notice and license text are retained in `LICENSE` and source files.
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) and [Habitat-Sim](https://github.com/facebookresearch/habitat-sim): simulator and evaluation dependencies. Check their license files and the notices on any reused source.
- [YOLOv7](https://github.com/WongKinYiu/yolov7): external detection implementation, GPL-3.0. Its source and weights are not bundled here. Installation and any redistribution must account for its license; a separate download does not by itself resolve all licensing obligations.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): external detection implementation, Apache-2.0. Check checkpoint terms separately.
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and [LAVIS](https://github.com/salesforce/LAVIS): externally installed model implementations. Consult their licenses and the terms for the specific pretrained assets.
- [frontier_exploration](https://github.com/naokiyokoyama/frontier_exploration) and [depth_camera_filtering](https://github.com/naokiyokoyama/depth_camera_filtering): external mapping dependencies. Consult their repository licenses.
- HM3D and MP3D: datasets are obtained independently through official access procedures and are not included in this repository.

The README links to upstream PointNav weights rather than redistributing checkpoint files. Other dependencies declared in `pyproject.toml` retain their respective licenses.
