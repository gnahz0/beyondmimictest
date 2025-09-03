from dotvar import auto_load  # noqa

from vuer_sim_example import tasks

for env_name in [
    "Pick_block-v1",
    "Pick_place-v1",
    "Flip_mug-v1",
]:
    env = tasks.make(env_name)
    act = env.action_space.sample()
    print(act.shape)

    obs = env.reset()

    obs, _, _, info = env.step(act)

    import matplotlib.pyplot as plt

    for camera_id in [0, 1, 2, 3, 4]:
        image = env.render("depth", camera_id=camera_id)
        plt.imshow(image)
        plt.show()

        print("hey hey!", image.shape)

# state = obs["state"]
# image = obs["render"]
#
#
# plt.imshow(image)
# plt.show()
# print(image.dtype)
#
# print(state.shape, image.shape)
