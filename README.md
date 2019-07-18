# Pong AI
Python-based Pong game implementation with AI players

## Getting Started

### Prerequisites
This application requires these python libraries to run:

* Numpy
* Pygame
* Keras
* TensorFlow

To download these modules, run the command
```
pip install pygame numpy keras tensorflow
```
### Running
Once the required libraries are installed, simply run
```
python pong.py
```
The application is currently set up to begin with untrained weights on the network. If you would like to use the pretrained weights, uncomment the line
```
#self.model = self.network(filename)
```
found in Agent.py

## Authors
* **Michael Read**  - [MichaelInAction](https://github.com/MichaelInAction)

## Acknowledgements

* Thanks to [Maurock](https://github.com/maurock) for the Reinforcement Learning example [Snake](https://github.com/maurock/snake-ga) which I learned from
