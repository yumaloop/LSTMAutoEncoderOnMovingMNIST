docker run \
--name moving_mnist \
--runtime=nvidia  \
-it \
-d \
-p 8888:8888 \
-v /home/uchiumi/workspace/moving_mnist:/root/user/moving_mnist \
kaggle_base:latest \
/bin/bash


