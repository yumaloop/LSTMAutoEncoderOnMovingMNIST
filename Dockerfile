FROM continuumio/anaconda3

## add user & SSH
RUN apt update && apt install -y openssh-server curl zip unzip vim
RUN mkdir /var/run/sshd
ENV USER UserName
ENV SHELL /bin/bash
RUN apt install -y sudo
RUN useradd -m ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:myPasswd" | chpasswd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

## spark
RUN curl -O http://ftp.tsukuba.wide.ad.jp/software/apache/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz &&\
tar -zxvf spark-2.4.3-bin-hadoop2.7.tgz &&\
mv spark-2.4.3-bin-hadoop2.7 /usr/local &&\
ln -s /usr/local/spark-2.4.3-bin-hadoop2.7 /usr/local/spark 
RUN echo "export SPARK_HOME=/usr/local/spark"  >> /etc/environment
RUN rm spark-2.4.3-bin-hadoop2.7.tgz

## java,kotlin,gradle
ENV SDKMAN_DIR "/usr/local/sdkman" 
SHELL ["/bin/bash", "-c"]
RUN curl -s "https://get.sdkman.io" | bash &&\
source "/usr/local/sdkman/bin/sdkman-init.sh" \
;sdk install java;sdk install kotlin;sdk install gradle;sdk install scala;sdk flush temp;
#sdk install sbt;sdk install spark;

## android
# RUN wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip
# RUN mkdir -p /Android/sdk
# RUN unzip sdk-tools-linux-4333796.zip -d /Android/sdk
ENV ANDROID_HOME /Android/sdk

## path and env for the user
RUN echo "/bin/bash" >> /etc/profile.d/default.sh
RUN echo "alias python="/opt/conda/bin/python"" >> /home/${USER}/.bashrc
RUN echo "export SDKMAN_DIR="/usr/local/sdkman""  >> /etc/environment
RUN chmod 755 /usr/local/sdkman/bin/sdkman-init.sh
RUN source /usr/local/sdkman/bin/sdkman-init.sh;echo "PATH=${PATH}:/usr/local/spark/bin:$ANDROID_HOME/bin" >> /home/${USER}/.bashrc
RUN echo "bash -c "/usr/local/sdkman/bin/sdkman-init.sh"" >> /etc/profile.d/default.sh
RUN echo "if [ -f ~/.bashrc ]; then  . ~/.bashrc;  fi" >>/home/${USER}/.bash_profile
RUN chmod 755 /etc/profile.d/default.sh

## for root login
# RUN sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/g" /etc/ssh/sshd_config &&\
# echo "root:myRootPasswd" | chpasswd
# ENV NOTVISIBLE "in users profile"
# RUN source /usr/local/sdkman/bin/sdkman-init.sh;echo "PATH=/usr/local/spark/bin:${PATH}" >> ~/.bashrc &&\
# echo "if [ -f ~/.bashrc ]; then  . ~/.bashrc;  fi" >>~/.bash_profile

## RDkit
RUN conda install -c rdkit rdkit

## Deep Neural net
RUN pip install -U pip && \
pip install tensorflow-gpu==2.0.0-alpha0 && \
pip install python-louvain && \
conda install -y cudnn cudatoolkit numba
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

## Others
RUN pip install cairosvg && \
pip install fastprogress japanize-matplotlib && \
pip install xgboost &&\
pip install lightgbm &&\
pip install rgf-python &&\
pip install mlxtend &&\
pip install bayesian-optimization

## Factorization Machines
# apt install -y --reinstall build-essential && \
# apt-get install -y python-dev libopenblas-dev && \
# git clone --recursive https://github.com/ibayer/fastFM.git && \
# cd fastFM && \
# pip install -r ./requirements.txt && \
# PYTHON=python3 make TARGET=core2 && \  
# pip install . && \

## Reinforcement learning
RUN apt -y install git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev  python3-tk tk-dev python-tk libfreetype6-dev python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
RUN pip install gym && \
pip install gym[classic_control] &&\
pip install gym[box2d] && \
git clone https://github.com/openai/baselines.git &&\
cd baselines &&\
pip install -e . &&\
pip install pytest &&\
pip install gym-retro

## clean files
RUN apt clean
