import random
from pathlib import Path

from vuer_sim_example.schemas.schema import Body
from vuer_sim_example.schemas.utils.file import Save
from vuer_sim_example.tasks import add_env
from vuer_sim_example.tasks._floating_robotiq import FloatingRobotiq2f85
from vuer_sim_example.tasks._tile_floor import TileFloor
from vuer_sim_example.tasks.entrypoint import make_env
from vuer_sim_example.vendors.robohive.robohive_object import RobohiveObj


# Generate random values for r, g, and b
r, g, b = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
x, y = random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)


def make_schema():
    # return scene._xml | Prettify()
    pass


def register():
    add_env(
        env_id="PickBlock-v1",
        entrypoint=make_env,
        kwargs=dict(
            xml_path="pick_block.mjcf.xml",
            workdir=Path(__file__).parent,
            mode="multiview",
        ),
    )


if __name__ == "__main__":
    make_schema() | Save(__file__.replace(".py", ".mjcf.xml"))
