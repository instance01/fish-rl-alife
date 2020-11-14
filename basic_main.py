from env.aquarium import Aquarium
from env.animal_controller import DefaultSharkController as DShark


EPISODES = 100


if __name__ == '__main__':
    env = Aquarium(
        observable_sharks=3,
        observable_fishes=3,
        size=30,
        max_fish=10,
        torus=False,
        fish_collision=True,
        lock_screen=False,
        max_steps=1000
    )

    env.select_fish_types(
        Random_Fish=0,
        Turn_Away_Fish=5,
        Boid_Fish=0
    )

    env.select_shark_types(Shark_Agents=1)

    for training_episode in range(1, EPISODES + 1):
        joint_shark_observation = env.reset()
        while not env.is_finished:
            shark_joint_action = {}
            for shark_name, observation in joint_shark_observation.items():
                action = DShark.get_action(
                    **env.prepare_observation_for_controller(observation)
                )
                shark_joint_action[shark_name] = action
            new_state = env.step(shark_joint_action)
            joint_shark_observation = new_state[0]
            env.render()
