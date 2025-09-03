from dotvar import auto_load  # noqa

from vuer_sim_example.tasks import make

env = make("Pick_block-v1")

for camera_id in [0, 1, 2, 3, 4]:
    image = env.render("rgb", camera_id=camera_id)
    print("hey hey!", image.shape)
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.title(f"Camera {camera_id}")
    plt.tight_layout()
    plt.show()

obs = env.reset()
act = env.action_space.sample()

obs, _, _, info = env.step(act)

env.close()
print(info.keys())
