# 安裝系統

- JetPack下載：[https://developer.nvidia.com/embedded/jetpack](https://developer.nvidia.com/embedded/jetpack)
    - JetPack 4.4 內相關套件版本
        - TensorRT 7.1.3
        - cuDNN 8.0
        - CUDA 10.2
        - OpenCV 4.1.1
- 燒錄image檔案到SD卡（或從SD備份映像檔燒錄）
    - 使用balenaEtcher: [https://www.balena.io/etcher/](https://www.balena.io/etcher/)
    - 使用rufus: [https://rufus.ie/](https://rufus.ie/)
    - 已把jetson nano從SD備份(使用rufus)的映像檔(30GB)壓縮成zip檔(7GB) [ 要用7-Zip解壓縮軟體解壓縮，否則無法燒錄到SD卡裡 ]: [https://drive.google.com/drive/folders/1dSqKI6-TU0pfpJU3UAW7j0_wMPmCWj-9](https://drive.google.com/drive/folders/1dSqKI6-TU0pfpJU3UAW7j0_wMPmCWj-9)
- [補充] [Backup Raspberry Pi SD Card on macOS](https://medium.com/@ccarnino/backup-raspberry-pi-sd-card-on-macos-the-2019-simple-way-to-clone-1517af972ca5)
- 映像檔開機
- 連接網路

## 遠端連線

- Get current IP address: $`ifconfig`

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/57b03d08-a6eb-4f98-923e-6d52dbcb9c7f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/57b03d08-a6eb-4f98-923e-6d52dbcb9c7f/Untitled.png)

- 以SSH連線 (以上圖為例)
    - ssh ai-nano@192.168.0.112 (ip位址由ifconfig查詢)
- 傳輸檔案：使用SFTP連線（可用Filezilla或Terminal）
    - sftp ai-nano@192.168.0.112 (ip位址由ifconfig查詢)

## 設定SWAP為8G

- 查看SWAP狀態 $ `free -h`

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
ls -lh /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon -show
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'
```

## 移除 libreoffice

```bash
sudo apt-get purge libreoffice*
sudo apt-get clean
```

## 安裝pip —> [REF](https://blog.csdn.net/beckhans/article/details/89146881)

- 先更新apt-get $ `sudo apt-get update`
- ~~$`sudo apt-get install python3-pip`~~
- $`sudo apt-get install python3-pip python3-dev`
- $`python3 -m pip install --upgrade pip`
- $`sudo vim /usr/bin/pip3`

    ```python
    # REPLACE
    # from pip import mai
    # if __name__ == '__main__':
        # sys.exit(main())

    # BY
    from pip import __main__
    if __name__ == '__main__':
        sys.exit(__main__._main())
    ```

    —

- check $`pip3 -V`
    - pip 20.2.2 from /home/ai-nano/.local/lib/python3.6/site-packages/pip (python 3.6)

## 安裝Jtop監控工具

- $`sudo -H pip install jetson-stats`
- 執行Jtop監控工具 $`sudo jtop`

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d605a289-a1f1-453d-8650-5b080013db35/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d605a289-a1f1-453d-8650-5b080013db35/Untitled.png)

---

## 安裝  virtualenv 和 virtualenvwrapper

- $ `sudo pip3 install virtualenv virtualenvwrapper`
- $ `vi ~/.bashrc`

    ```bash
    # virtualenv and virtualenvwrapper
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source /usr/local/bin/virtualenvwrapper.sh
    ```

    $ `source ~/.bashrc`

### 使用virtualenv

- `mkvirtualenv`: 建立環境
    - Ex: `mkvirtualenv AI -p python3`
- `lsvirtualenv`: 列出所有環境
- `rmvirtualenv`: 移除環境
- `workon`: 進入環境
    - Ex: `workon AI`
- `deactivate`: 離開環境

[CSDN-个人空间](https://me.csdn.net/beckhans)

[玩转Jetson Nano（三）安装TensorFlow GPU_beckhans的博客-CSDN博客_玩转jetson nano](https://blog.csdn.net/beckhans/article/details/89146881)

[Nvidia Jetson Nano 使用心得 - HackMD](https://hackmd.io/@0p3Xnj8xQ66lEl0EHA_2RQ/Skoa_phvB)

## 檢查CUDA

- 透過Jtop可知JetPack4.4已預裝CUDA 10.2

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eed1c3f5-1aa4-439e-bfd4-5453f7a040f3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eed1c3f5-1aa4-439e-bfd4-5453f7a040f3/Untitled.png)

- 在.bashrc增加以下三行

    ```bash
    export CUDA_HOME=/usr/local/cuda-10.2/
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
    export PATH=${CUDA_HOME}bin:$PATH
    ```

- $ `source .bashrc`
- 測試 $ `nvcc -V`，顯示👇🏻

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_21:14:42_PDT_2019
    Cuda compilation tools, release 10.2, V10.2.89

## 安裝python常用套件

- Scipy:
    - $ `sudo apt-get install python3-scipy`
- Pandas:
    - $ `sudo apt-get install python3-pandas`
- Scikit-learn:
    - $ `sudo apt-get install python3-sklearn`
- Matplotlib:
    - $ `sudo apt-get install python3-matplotlib`
- Numpy:
    - $ `sudo apt-get install python3-numpy`
    - $ `sudo apt install python3-h5py`

## 安裝TensorFlow GPU

- 官方說明

[Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#uninstall)

[Release Notes For Jetson Platform :: NVIDIA Deep Learning Frameworks Documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html)

- 查看目前JetPack 4.4適用的Tensorflow版本

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b8cca23-e6ca-402e-98b4-00618901d065/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b8cca23-e6ca-402e-98b4-00618901d065/Untitled.png)

- 先安裝相依套件：
    - $ `sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran`
    - $ `sudo apt-get install python3-pip`
    - $ `sudo pip3 install -U pip testresources setuptools`
    - $ `sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11`

- 安裝Tensorflow版本

    ***1.15.2版(成功！！！🎉🎉🎉)***

    $ `sudo pip3 install --pre --extra-index-url [https://developer.download.nvidia.com/compute/redist/jp/v44](https://developer.download.nvidia.com/compute/redist/jp/v44) tensorflow==1.15.2`

- 安裝Keras
    - ~~$ `sudo pip3 install keras`~~(不使用！)
    - $ `sudo pip3 install keras==2.2.4` (成功！)

- 驗證安裝
    - $ `python3`
    - >>> `import tensorflow as tf`
    - >>> `tf.__version__`
    - >>> `import keras`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4450cd3e-398f-4bba-83b5-55e63124b50f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4450cd3e-398f-4bba-83b5-55e63124b50f/Untitled.png)

## **安裝 pycuda**

[jetson nano安装pycuda！！！_weixin_44501699的博客-CSDN博客](https://blog.csdn.net/weixin_44501699/article/details/106470671)

## **調整功耗模式**

- 鎖住功率使其不過載
    - `sudo jetson_clocks`
- 顯示當前模式
    - `sudo nvpmodel -q`
- 預設為高效能模式MAX N模式(10W) (這個功率需要接DC 5V 4A，不然會突然關機)
    - `sudo nvpmodel -m 0`
- #切換換到 5W 模式(Micro-USB供電)
    - `sudo nvpmodel -m 1`

## 使用Web cam

- 查看USB Webcam編號：video0代表編號為0
    - `ls -ltrh /dev/video*`
- Webcam - [sample codes](https://blog.cavedu.com/2020/02/06/jetson-nano-01-webcam/)

.

- 風扇使用說明 : 請下指令~~~~***🎉🎉🎉***

[【教學】NVIDIA® Jetson Nano™ 散熱風扇 使用教學](https://blog.cavedu.com/2019/10/04/nvidia-jetson-nano-fan/)

---

## 參考資料

[NVIDIA Jetson Nano 實際使用難不難？從入手到安裝系統、開機與遠端連線](https://blog.cavedu.com/2019/04/03/jetson-nano-installsystem/)

[NVIDIA Jetson Nano學習筆記（一）:初次使用](https://medium.com/@yanweiliu/nvidia-jetson-nano學習筆記-一-初次使用-4dce57a0b2b1)

[Nvidia Jetson Nano 初體驗：安裝與測試 - Building Maker Economy：自造達人社群/媒體/平台](https://makerpro.cc/2019/05/the-installation-and-test-of-nvida-jetson-nano/)

[How to configure your NVIDIA Jetson Nano for Computer Vision and Deep Learning - PyImageSearch](https://www.pyimagesearch.com/2020/03/25/how-to-configure-your-nvidia-jetson-nano-for-computer-vision-and-deep-learning/)

- [x]  Step #5: Install system-level dependencies
- [x]  Step #6: Update CMake
- [x]  Step #7: Install OpenCV system-level dependencies and other development dependencies
- [x]  Step #8: Set up Python virtual environments
- [x]  Step #9: Create your ‘py3cv4’ virtual environment
- [x]  Step #10: Install the Protobuf Compiler (1HR)
- [x]  Step #11: Install TensorFlow, Keras, NumPy, and SciPy
- [x]  Step #12: Install the TensorFlow Object Detection API
- [x]  Step #13: Install NVIDIA’s ‘tf_trt_models’
- [x]  Step #14: Install OpenCV 4.1.2
- [x]  Step #15: Install other useful libraries via pip
    - [x]  matplotlib
    - [x]  sklearn
    - [x]  pillow imutils scikit-image
    - [x]  dlib
    - [x]  flask
    - [ ]  jupyter
    - [x]  lxml progressbar2
- [ ]  Step #16: Testing and Validation
    - [x]  Testing TensorFlow and Keras
    - [ ]  Testing the TFOD API
    - [ ]  Testing OpenCV
    - [ ]  Testing webcam
- 安裝常用套件

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install libfreetype6-dev pkg-config -y
    sudo apt-get install zlib1g-dev zip libjpeg8-dev libhdf5-dev -y
    sudo apt-get install libssl-dev libffi-dev python3-dev -y
    sudo apt-get install libhdf5-serial-dev hdf5-tools -y
    sudo apt-get install libblas-dev liblapack-dev
    sudo apt-get install  libatlas-base-dev -y
    sudo apt-get install build-essential cmake libgtk-3-dev libboost-all-dev -y
    ```

    ```bash
    pip3 install matplotlib
    pip3 install scikit-build
    pip3 install imutils
    pip3 install pillow
    ```

    - [ ]  安裝Scipy

        ```bash
        wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz
        tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3
        cd scipy-1.3.3/
        python setup.py install
        ```

    - [ ]  安裝Scikit-learn
    - [ ]  安裝Keras
        - `pip3 install keras`
    - [ ]  安裝Tensorflow (建議安裝1.13避免與TensorRT衝突。~~TF2.0~~)
        - `pip3 install --extra-index-url [https://developer.download.nvidia.com/compute/redist/jp/v42](https://developer.download.nvidia.com/compute/redist/jp/v42) tensorflow-gpu==1.13.1+nv19.3`
    - [ ]  安裝Jupyter notebook
    - [ ]  開啟SWAP
    - [ ]  安裝Darknet
    - [ ]  安裝Jetson stats

### 確認套件版本

- Check python3 `python3 —version`
- Check opencv: enter the python shell `python3`

    ```python
    import cv2
    print(cv2.__version__)
    ```

---

---

### 跑Model

- [使用內建的sample model做分類測試](https://blog.cavedu.com/2019/04/30/nvidia-jetson-nano-example/)

- 安裝 TensorRT - [A Guide to using TensorRT on the Nvidia Jetson Nano](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/)

export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.2
