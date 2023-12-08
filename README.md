# The main function of this script is to simultaneously recognize the faces of two passengers captured by the ZED camera, and to conduct facial emotion recognition based on the captured faces. During this process, Opencv's GPU support and Tensorflow's GPU acceleration are used to achieve better real-time analysis results with the pretrained model.

1. Install Python3 on Ubuntu 22.04
##sudo apt update
##sudo apt install build-essential software-properties-common -y
##sudo add-apt-repository ppa:deadsnakes/ppa
##sudo apt update
##sudo apt install python3.11 -y
##python3.11 --version
#sudo apt install python3-pip

2. Create a Virtual environment
#sudo apt install python3.10-venv
#python3 -m venv myenv
#source myenv/bin/activate

3. Install OpenCV with CPU support
#pip install opencv-python

4. Install NVIDIA-driver
#sudo add-apt-repository ppa:graphics-drivers/ppa
#sudo apt-get update
#sudo ubuntu-drivers autoinstall
    #a.Reboot your computer and enter the BIOS/UEFI settings. This is usually done by pressing a key like F2, F12, 
Delete, or Esc during boot-up, depending on your motherboard manufacturer.
    #b.Navigate to the Secure Boot option in the BIOS settings. This is usually found under the 'Boot', 
    'Security', or 'System Configuration' tab.
    #c.Disable Secure Boot and then save and exit the BIOS.
#nvidia-smi

5. Install OpenCV with CPU supported
#pip install opencv-python

6. Install TensorFlow with GPU supported
#pip install --upgrade pip
#pip install tensorflow[and-cuda]

7.Install MTCNN__Specialized face detection model
#pip install mtcnn

8.These websites below is you need to check out the version of Tensorflow, cuda, and cudnn
#https://www.tensorflow.org/install/source#gpu

9.Upgrade your Tensorflow version to 2.15.0
#pip install --upgrade tensorflow

Version	                Python-version	    Compiler        Build tools	      cuDNN	     CUDA
tensorflow-2.15.0	    3.9-3.11	        Clang 16.0.0	Bazel 6.1.0	      8.8	     12.2

10. check out version
a.check python version 
#python --version
Python 3.10.12
    
b.install clang-16.0.0
#wget https://apt.llvm.org/llvm.sh
#chmod +x llvm.sh
#sudo ./llvm.sh 16
#sudo apt-get install clang-16
#clang-16 --version
#sudo ln -s /usr/bin/clang-16 /usr/bin/clang
#clang --version

c. Install bazel 
#wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-installer-linux-x86_64.sh
#chmod +x bazel-6.1.0-installer-linux-x86_64.sh
#bazel --version

d.Install Cuda Toolkit 12.2 version, check below link
https://developer.nvidia.com/cuda-toolkit-archive
https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
#sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb
#sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
#sudo apt-get update
#sudo apt-get -y install cuda
#nvcc --version

e.Install cuDNN 8.8 version, check link below
https://developer.nvidia.com/rdp/cudnn-archive
Download cuDNN v8.8.0 (February 7th, 2023), for CUDA 12.0
Local Installer for Ubuntu22.04 x86_64 (Deb)

#sudo dpkg -i cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb
#sudo cp /var/cudnn-local-repo-ubuntu2204-8.8.0.121/cudnn-local-04B81517-keyring.gpg /usr/share/keyrings/
#sudo apt-get update
#sudo apt-get install libcudnn8
#sudo apt-get install libcudnn8-dev
#ldconfig -p | grep cudnn

d. edit your bashrc file
#vim ~/.bashrc
#export PATH="$HOME/bin:/usr/local/cuda-12.2/bin:$PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
#source ~/.bashrc
#source myenv/bin/activate

f.version check for all
#python --version
#clang --version
#Bazel --version
#ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so*
#nvcc --version

11. compile tensorflow manually, check link below
https://www.tensorflow.org/install/source

#git clone https://github.com/tensorflow/tensorflow.git
#cd tensorflow
#git checkout r2.15  # Check out the branch for TensorFlow 2.15
#./configure

12.install TensorRT, check link below
https://developer.nvidia.com/nvidia-tensorrt-8x-download
TensorRT 8.6 GA for Ubuntu 22.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 DEB local repo Package

13. find out current your cuddn version and where the file is located
dpkg -l | grep cudnn

14. Install new version of cudnn 8.9.6
#sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.6.50_1.0-1_amd64.deb
#sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.6.50/cudnn-local-1998375D-keyring.gpg /usr/share/keyrings/
#sudo apt-get update
#sudo apt-get install libcudnn8
#dpkg -l | grep cudnn

15. Install tensorflow website
https://www.tensorflow.org/install/source#build_and_install_the_pip_package

You need to download the file below from the link
https://huggingface.co/spaces/mxz/emtion/blob/c697775e0adc35a9cec32bd4d3484b5f5a263748/fer2013.csv

13. train the model
python3 emotionRecTrain.py --csv_file=data/fer2013.csv --export_path=model_out/

14. Configuring your OpenCV model with GPU supported 
#cd ~/Facial_Detection/opencv
#mkdir build
#cd build

      #cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/home/dl/Facial_Detection/myenv \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=/home/dl/Facial_Detection/myenv/bin/python \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.10 \
      -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/dl/Facial_Detection/myenv/lib/python3.10/site-packages/numpy/core/include \
      -D PYTHON3_PACKAGES_PATH=/home/dl/Facial_Detection/myenv/lib/python3.10/site-packages \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_CUDA=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDNN_INCLUDE_DIR=/home/dl/Facial_Detection/myenv/lib/python3.10/site-packages/nvidia/cudnn/include \
      -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.6 \
      -D WITH_CUDNN=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_FFMPEG=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/home/dl/Facial_Detection/opencv_contrib/modules \
      -D WITH_GSTREAMER=ON \
      -D BUILD_EXAMPLES=ON \
      -D WITH_GTK=ON \
      ..

#make -j$(nproc)
#sudo make install 

15. check out your gpu usage when running the script
#watch -n 1 nvidia-smi

