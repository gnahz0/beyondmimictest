from collections import defaultdict
from typing import List


from vuer_sim_example.wrappers.camera_wrapper import DepthWrapper
from vuer_sim_example.wrappers.utils.depth_util import invisibility
import numpy as np
from pathlib import Path

labels = [
    "wall",
    "building;edifice",
    "sky",
    "floor;flooring",
    "tree",
    "ceiling",
    "road;route",
    "bed",
    "windowpane;window",
    "grass",
    "cabinet",
    "sidewalk;pavement",
    "person;individual;someone;somebody;mortal;soul",
    "earth;ground",
    "door;double;door",
    "table",
    "mountain;mount",
    "plant;flora;plant;life",
    "curtain;drape;drapery;mantle;pall",
    "chair",
    "car;auto;automobile;machine;motorcar",
    "water",
    "painting;picture",
    "sofa;couch;lounge",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug;carpet;carpeting",
    "field",
    "armchair",
    "seat",
    "fence;fencing",
    "desk",
    "rock;stone",
    "wardrobe;closet;press",
    "lamp",
    "bathtub;bathing;tub;bath;tub",
    "railing;rail",
    "cushion",
    "base;pedestal;stand",
    "box",
    "column;pillar",
    "signboard;sign",
    "chest;of;drawers;chest;bureau;dresser",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace;hearth;open;fireplace",
    "refrigerator;icebox",
    "grandstand;covered;stand",
    "path",
    "stairs;steps",
    "runway",
    "case;display;case;showcase;vitrine",
    "pool;table;billiard;table;snooker;table",
    "pillow",
    "screen;door;screen",
    "stairway;staircase",
    "river",
    "bridge;span",
    "bookcase",
    "blind;screen",
    "coffee;table;cocktail;table",
    "toilet;can;commode;crapper;pot;potty;stool;throne",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove;kitchen;stove;range;kitchen;range;cooking;stove",
    "palm;palm;tree",
    "kitchen;island",
    "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
    "swivel;chair",
    "boat",
    "bar",
    "arcade;machine",
    "hovel;hut;hutch;shack;shanty",
    "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle",
    "towel",
    "light;light;source",
    "truck;motortruck",
    "tower",
    "chandelier;pendant;pendent",
    "awning;sunshade;sunblind",
    "streetlight;street;lamp",
    "booth;cubicle;stall;kiosk",
    "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
    "airplane;aeroplane;plane",
    "dirt;track",
    "apparel;wearing;apparel;dress;clothes",
    "pole",
    "land;ground;soil",
    "bannister;banister;balustrade;balusters;handrail",
    "escalator;moving;staircase;moving;stairway",
    "ottoman;pouf;pouffe;puff;hassock",
    "bottle",
    "buffet;counter;sideboard",
    "poster;posting;placard;notice;bill;card",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter",
    "canopy",
    "washer;automatic;washer;washing;machine",
    "plaything;toy",
    "swimming;pool;swimming;bath;natatorium",
    "stool",
    "barrel;cask",
    "basket;handbasket",
    "waterfall;falls",
    "tent;collapsible;shelter",
    "bag",
    "minibike;motorbike",
    "cradle",
    "oven",
    "ball",
    "food;solid;food",
    "step;stair",
    "tank;storage;tank",
    "trade;name;brand;name;brand;marque",
    "microwave;microwave;oven",
    "pot;flowerpot",
    "animal;animate;being;beast;brute;creature;fauna",
    "bicycle;bike;wheel;cycle",
    "lake",
    "dishwasher;dish;washer;dishwashing;machine",
    "screen;silver;screen;projection;screen",
    "blanket;cover",
    "sculpture",
    "hood;exhaust;hood",
    "sconce",
    "vase",
    "traffic;light;traffic;signal;stoplight",
    "tray",
    "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin",
    "fan",
    "pier;wharf;wharfage;dock",
    "crt;screen",
    "plate",
    "monitor;monitoring;device",
    "bulletin;board;notice;board",
    "shower",
    "radiator",
    "glass;drinking;glass",
    "clock",
    "flag",
]
colors = np.array(
    [
        (120, 120, 120),
        (180, 120, 120),
        (6, 230, 230),
        (80, 50, 50),
        (4, 200, 3),
        (120, 120, 80),
        (140, 140, 140),
        (204, 5, 255),
        (230, 230, 230),
        (4, 250, 7),
        (224, 5, 255),
        (235, 255, 7),
        (150, 5, 61),
        (120, 120, 70),
        (8, 255, 51),
        (255, 6, 82),
        (143, 255, 140),
        (204, 255, 4),
        (255, 51, 7),
        (204, 70, 3),
        (0, 102, 200),
        (61, 230, 250),
        (255, 6, 51),
        (11, 102, 255),
        (255, 7, 71),
        (255, 9, 224),
        (9, 7, 230),
        (220, 220, 220),
        (255, 9, 92),
        (112, 9, 255),
        (8, 255, 214),
        (7, 255, 224),
        (255, 184, 6),
        (10, 255, 71),
        (255, 41, 10),
        (7, 255, 255),
        (224, 255, 8),
        (102, 8, 255),
        (255, 61, 6),
        (255, 194, 7),
        (255, 122, 8),
        (0, 255, 20),
        (255, 8, 41),
        (255, 5, 153),
        (6, 51, 255),
        (235, 12, 255),
        (160, 150, 20),
        (0, 163, 255),
        (140, 140, 140),
        (250, 10, 15),  # corrected '0250' to 250
        (20, 255, 0),
        (31, 255, 0),
        (255, 31, 0),
        (255, 224, 0),
        (153, 255, 0),
        (0, 0, 255),
        (255, 71, 0),
        (0, 235, 255),
        (0, 173, 255),
        (31, 0, 255),
        (11, 200, 200),
        (255, 82, 0),
        (0, 255, 245),
        (0, 61, 255),
        (0, 255, 112),
        (0, 255, 133),
        (255, 0, 0),
        (255, 163, 0),
        (255, 102, 0),
        (194, 255, 0),
        (0, 143, 255),
        (51, 255, 0),
        (0, 82, 255),
        (0, 255, 41),
        (0, 255, 173),
        (10, 0, 255),
        (173, 255, 0),
        (0, 255, 153),
        (255, 92, 0),
        (255, 0, 255),
        (255, 0, 245),
        (255, 0, 102),
        (255, 173, 0),
        (255, 0, 20),
        (255, 184, 184),
        (0, 31, 255),
        (0, 255, 61),
        (0, 71, 255),
        (255, 0, 204),
        (0, 255, 194),
        (0, 255, 82),
        (0, 10, 255),
        (0, 112, 255),
        (51, 0, 255),
        (0, 194, 255),
        (0, 122, 255),
        (0, 255, 163),
        (255, 153, 0),
        (0, 255, 10),
        (255, 112, 0),
        (143, 255, 0),
        (82, 0, 255),
        (163, 255, 0),
        (255, 235, 0),
        (8, 184, 170),
        (133, 0, 255),
        (0, 255, 92),
        (184, 0, 255),
        (255, 0, 31),
        (0, 184, 255),
        (0, 214, 255),
        (255, 0, 112),
        (92, 255, 0),
        (0, 224, 255),
        (112, 224, 255),
        (70, 184, 160),
        (163, 0, 255),
        (153, 0, 255),
        (71, 255, 0),
        (255, 0, 163),
        (255, 204, 0),
        (255, 0, 143),
        (0, 255, 235),
        (133, 255, 0),
        (255, 0, 235),
        (245, 0, 255),
        (255, 0, 122),
        (255, 245, 0),
        (10, 190, 212),
        (214, 255, 0),
        (0, 204, 255),
        (20, 0, 255),
        (255, 255, 0),
        (0, 153, 255),
        (0, 41, 255),
        (0, 255, 204),
        (41, 0, 255),
        (41, 255, 0),
        (173, 0, 255),
        (0, 245, 255),
        (71, 0, 255),
        (122, 0, 255),
        (0, 255, 184),
        (0, 92, 255),
        (184, 255, 0),
        (0, 133, 255),
        (255, 214, 0),
        (25, 194, 194),
        (102, 255, 0),
        (92, 0, 255),
    ],
    dtype=np.uint8,
)


class SegmentationWrapper(DepthWrapper):
    """Need to consider getting a floating point depth render wrapper, too."""

    def __init__(self, env, *, invisible_prefix: List = ["gripper", "ur5"], prefix_to_class_ids: dict[str, int] = None, **kwargs):
        super().__init__(env, **kwargs)
        model = self.unwrapped.env.physics.model
        self.invisible_objects = set()
        self.visible_objects = []
        for bodyid in range(model.nbody):
            body_name = model.body(bodyid).name
            if any(body_name.startswith(prefix) for prefix in invisible_prefix):
                for geomid in range(model.body(bodyid).geomadr[0], model.body(bodyid).geomadr[0] + model.body(bodyid).geomnum[0]):
                    self.invisible_objects.add(geomid)
        for geomid in range(model.ngeom):
            if geomid not in self.invisible_objects:
                self.visible_objects.append(geomid)
        self.invisible_objects = list(self.invisible_objects)
        self.geom_to_class_id = {}
        self.prefix_to_geom_id = defaultdict(list)
        if prefix_to_class_ids is not None:
            for prefix, class_id in prefix_to_class_ids.items():
                for bodyid in range(model.nbody):
                    body_name = model.body(bodyid).name
                    if body_name.startswith(prefix):
                        for geomid in range(model.body(bodyid).geomadr[0], model.body(bodyid).geomadr[0] + model.body(bodyid).geomnum[0]):
                            self.geom_to_class_id[geomid] = class_id
                            self.prefix_to_geom_id[prefix].append(geomid)
        self.image_key = f"{self.image_key}/raw_segmentation"

    def _compute_additional_obs(self, obs=None):
        physics = self.unwrapped.env.physics
        with invisibility(physics, self.invisible_objects):
            segmentation = self.render(
                segmentation=True,
                # mode='segmentation',
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
        segmentation = segmentation[..., 0]
        return {self.image_key: segmentation}


class SegmentationRGBWrapper(SegmentationWrapper):
    def __init__(self, env, *, invisible_prefix: List = ["gripper", "ur5"], prefix_to_class_ids: dict[str, int] = None, **kwargs):
        super().__init__(env, invisible_prefix=invisible_prefix, prefix_to_class_ids=prefix_to_class_ids, **kwargs)
        self.segmentation_key = self.image_key
        self.image_key = Path(self.image_key).parent
        self.image_key = f"{self.image_key}/rgb"

    def _compute_additional_obs(self, obs=None):
        segmentation_image_uint8 = obs[self.segmentation_key].copy()
        for geom_id in self.geom_to_class_id:
            segmentation_image_uint8[segmentation_image_uint8 == geom_id] = self.geom_to_class_id[geom_id]
        segmentation_image_uint8[segmentation_image_uint8 == 255] = len(colors) - 1
        segmentation_image_uint8 = colors[segmentation_image_uint8 - 1]
        return {self.image_key: segmentation_image_uint8}


class OverlayWrapper(SegmentationWrapper):
    def __init__(self, env, *, invisible_prefix: List = ["gripper", "ur5"], prefix_to_class_ids: dict[str, int] = None, **kwargs):
        super().__init__(env, invisible_prefix=invisible_prefix, prefix_to_class_ids=prefix_to_class_ids, **kwargs)
        self.image_key = Path(self.image_key).parent
        self.image_key = f"{self.image_key}/overlay"
        self.mask_key = f"{self.image_key}/mask"

    def _compute_additional_obs(self, obs=None):
        overlay = self.render(
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )
        segmentation = self.render(
            segmentation=True,
            # mode="segmentation",
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )
        segmentation = segmentation[..., 0]
        mask = np.logical_not(np.isin(segmentation, self.invisible_objects))
        overlay[mask] = (255, 255, 255)
        return {self.image_key: overlay, self.mask_key: mask}


class MidasDepthWrapper(SegmentationWrapper):
    def __init__(self, env, *, invisible_prefix: List = ["gripper", "ur5"], prefix_to_class_ids: dict[str, int] = None, **kwargs):
        super().__init__(env, invisible_prefix=invisible_prefix, prefix_to_class_ids=prefix_to_class_ids, **kwargs)
        self.image_key = Path(self.image_key).parent
        self.image_key = f"{self.image_key}/midas_depth"
        self.near = 0.02
        self.far = 1.9

    def to_midas_depth(self, depth_render):
        near_mask = depth_render < self.far
        midas_depth = 1 / depth_render
        if near_mask.sum() > 0:
            low = midas_depth[near_mask].min()
        else:
            low = midas_depth.min()
        low -= 1
        midas_depth = midas_depth.clip(low, None)
        midas_depth = (midas_depth - low) / (midas_depth.max() - low + 1e-8)
        midas_depth = (midas_depth * 255).astype(np.float16)
        return midas_depth

    def to_normalized_depth(self, depth_render):
        d = np.clip(depth_render, self.near, self.far)

        d = (d - self.near) / (self.far - self.near)
        d = (d * 255).astype(np.uint8)

        return d

    def recover_metric_depth(self, n_depth):
        d = n_depth / 255
        d = d * (self.far - self.near) + self.near
        valid_mask = np.logical_and(d > self.near, d < self.far)
        return d, valid_mask

    def _compute_additional_obs(self, obs=None):
        physics = self.unwrapped.env.physics
        with invisibility(physics, self.invisible_objects):
            depth = self.render(
                "depth",
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
        full_depth = self.render(
            "depth",
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )

        additional_obs = {
            self.image_key: self.to_midas_depth(depth),
            f"{self.image_key}_full": self.to_normalized_depth(full_depth),
        }

        return additional_obs


class MaskedMidasDepthWrapper(DepthWrapper):
    def __init__(self, env, *, invisible_prefix: List = ["gripper", "ur5"], prefix_to_class_ids: dict[str, int] = None, **kwargs):
        super().__init__(env, invisible_prefix=invisible_prefix, prefix_to_class_ids=prefix_to_class_ids, **kwargs)
        self.image_key = Path(self.image_key).parent
        self.image_key = f"{self.image_key}/masked_midas_depth"
        self.near = 0.0
        self.far = 1.9

    def _compute_additional_obs(self, obs=None):
        mask_keys = [
            k
            for k in obs.keys()
            if not (k.endswith("overlay") or k.endswith("segmentation") or k.endswith("depth"))
            and k.startswith(str(Path(self.image_key).parent))
        ]
        masks = [obs[k].astype(np.bool) for k in mask_keys]
        masks = np.array(masks)
        masks = np.logical_or.reduce(masks, axis=0)
        physics = self.unwrapped.env.physics
        with invisibility(physics, self.invisible_objects):
            depth = self.render(
                "depth",
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
            depth = depth.clip(self.near, self.far)
        midas_depth = 1 / depth
        if np.sum(masks) > 0:
            midas_depth = (midas_depth - midas_depth[masks].min()) / (midas_depth[masks].max() - midas_depth[masks].min() + 1e-8)
        midas_depth[np.logical_not(masks)] = 0
        midas_depth = (midas_depth * 255).astype(np.uint8)
        return {self.image_key: midas_depth}


class MaskWrapper(SegmentationWrapper):
    def __init__(self, env, *, object_prefix, **kwargs):
        super().__init__(env, **kwargs)
        model = self.unwrapped.env.physics.model
        self.object_prefix = object_prefix
        self.geom_ids = []
        for bodyid in range(model.nbody):
            body_name = model.body(bodyid).name
            if body_name.startswith(object_prefix):
                for geomid in range(model.body(bodyid).geomadr[0], model.body(bodyid).geomadr[0] + model.body(bodyid).geomnum[0]):
                    self.geom_ids.append(geomid)
        self.segmentation_key = self.image_key
        self.image_key = Path(self.image_key).parent
        self.image_key = f"{self.image_key}/{object_prefix}"

    def _compute_additional_obs(self, obs):
        segmentation = obs[self.segmentation_key]
        object_mask = np.isin(segmentation, self.geom_ids)
        return {self.image_key: np.packbits(object_mask)}
