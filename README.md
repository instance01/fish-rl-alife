
MARL on Aquarium environment --- can sharks learn to cooperate? Or is the tragedy of the commons unavoidable?

Using PPO.

![PPO](plots/plot.png)

This is a Cython project, that's why there's `.pyx` files everywhere. They still work with normal Python when executed directly (e.g. `python3 config.pyx`), because I didn't go deep into cythonization due to time constraints. It was simply a basically free 40% performance boost.

## Debugging

python3 -m pdb -c continue main.py

## Results

Best so far: 13\_3 and ma3 (20-11-27\_07:56:47-99-inn.cip.ifi.lmu.de-ma3)

Of course ma3 had the sharks not die. So they cooperated because they can't die and cooperating increases reward.

![Sterberisiko vs Herding rate](plots/sterberisiko_vs_herding_rate_3_evals_per_model.png)

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
