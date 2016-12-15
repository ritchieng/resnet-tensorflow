# Residual Networks in TensorFlow

## Residual Network in TensorFlow
This entire code is implemented in pure TensorFlow.  It's a simpler version to run but it has the full capabilities of the original one. The original version was complicated without some essential features like stopping the training and logging.

## Simple Instructions
- Running Training
    - `python main.py`
        - If you want to modify any parameters, you can use for example `python main.py --n_epoch==10`
            - The default runs on CIFAR-10 dataset and this configuration is made for that.
            - `n_epoch`: number of epochs, default `10`
            - `n_batch`: batch size, default `64`
            - `n_img_row`: dimension of image (row), default `32`
            - `n_img_col`: dimension of image (col), default `32`
            - `n_img_channels`: number of channels, default `3`
            - `n_classes`: number of classes, default `10`
        
- Running TensorBoard
    - Training logs
        - `tensorboard --logdir=train_log`
    - Evaluation logs
        - `tensorboard --logdir=eval_log`
    - You can use any path you want. 
        - If you encountered a `permission denied` error, you can easily solve it by changing the directory to `tmp/train_log`.
        - I experienced this while running on Amazon AWS and it was solved with this fix.

## Credits
- The original model was obtained from the [official repository](https://github.com/tensorflow/models/tree/master/resnet) by TensorFlow.
    - You can access this original file with the file named `resnet_main.py`. 
- The paper on [Residual Networks](https://arxiv.org/abs/1512.03385) on arXiv.org.

## Dependencies
- To simplify the code, I read the CIFAR dataset using TensorLayer.
    - Simply run `sudo pip install tensorlayer` and you are good to go! 
- TensorFlow v0.12

## License
MIT

