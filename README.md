
MARL on predator-prey aquarium environment --- can sharks learn to cooperate? Or is the tragedy of the commons unavoidable?

Using PPO.

## Project Structure

|File/Folder|Purpose|
|----|-------|
|contrib/|Helper bash files for the compute pool.|
|env/|Contains the aquarium environment with an API in accordance to OpenAI gym.|
|guide/|Contains a small guide to the environment.|
|models/|Trained models are saved here.|
|paper/|Configuration (json) handling|
|pickles/|Data created by evaluators.|
|plots/|Plots for the collected data.|
|profiles/|cProfile files used when I sped up the pipeline by using Cython.|
|runs/|Contains runs (for watching in Tensorboard).|
|build.sh|Builds the Cython project.|
|config.pyx|Handles simulations.json. Can be executed directly to get all keys in simulations.json.|
|custom\_logger.pyx|Logging KPIs to Tensorboard.|
|main.py|Entry point.|
|main\_ddpg.py|Out of date!|
|network\_models.py|MLP (norm/batchnorm) neural network models for baselines PPO.|
|pipeline.py|End to end training and evaluation of experiments from simulations.json. The meat of the project.|
|run\_profile.py|For running cProfile.|
|show\_profile.py|For showing cProfile results.|
|shark\_baselines.py|Contains code to run deterministic shark algorithm.|
|simulations.json|All experiments and configurations.|

Note. This is a Cython project, which is why there are `.pyx` files everywhere.
They still work with normal Python when executed directly (e.g. `python3 config.pyx`), because I didn't go deep into cythonization due to time constraints.
It was simply a basically free 40% performance boost.

## Installation and building

1. Install all packages in `requirements.txt`: `pip3 install -r requirements.txt`
2. Run `build.sh`. This requires `g++`!

## Running

1. Create a new experiment in simulations.json or re-use one. For instance, you could pick `ma9_i5_p150_r10_s05_sp200_two_net_vd35_f`.
2. Run: `python3 main.py ma9_i5_p150_r10_s05_sp200_two_net_vd35_f single`

Models are saved in `models/` and runs (for checking them in Tensorboard, e.g. `tensorboard --logdir runs`) are saved in `runs/`.

To load and watch a model, run for example: `python3 main.py ma8_obs load models/ma8_obs-chamb.cip.ifi.lmu.de-20.12.29-20:02:12-83775101-model-F`
You can change the cfg\_id to run the model in (ma8\_obs) and you can change the model to run.

## Results

Accompanying blog post can be found [here](https://blog.xa0.de/post/Emergent-Behaviour-in-Multi-Agent-Reinforcement-Learning%20---%20Independent-PPO/).

TODO: Add a few pictures here.

## Citing

To cite this repository in publications:

```
@misc{RRP+21,
  author = {Ratke, Daniel and Ritz, Fabian and Phan, Thomy and Belzner, Lenz and Linnhoff-Popien, Claudia},
  title = {fish-rl-alife},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/instance01/fish-rl-alife}},
}
```
