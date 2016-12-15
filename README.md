# Residual Networks in TensorFlow

## Residual Network in TensorFlow
This entire code is implemented in pure TensorFlow.  It's a simpler version to run but it has the full capabilities of the original one. The original version was complicated without proper logging.

## Instructions
- Running Training
    - `python main.py`
- Running TensorBoard
    - `tensorboard --logdir=train_log`

## Credits
- The original was obtained from the [official repository](https://github.com/tensorflow/models/tree/master/resnet) by TensorFlow.
- The paper on [Residual Networks](https://arxiv.org/abs/1512.03385) on arXiv.org

## Dependencies
- To simplify the code, I read the CIFAR dataset using TensorLayer.
    - Simply run `sudo pip install tensorlayer` and you are good to go! 

## License
MIT

