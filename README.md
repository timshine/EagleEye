# EagleEye Drone Surveillance System
## Central Base Station Specs for Running GPU Accelerated OpenCV
### Current Configuration
* AMD Ryzen 5 2600 Processor
* GeForce GTX 1660 oc 6G Graphics Card, 6GB 192-bit GDDR5
* 16 GB DDR4 DRAM 3000MHz
* 240 GB SATA 2.5 inch SSD
* 1 TB HDD SATA 6Gb/s

### Minimum Computer Requirements
* A computer with a Nvidia Graphics Card
  
## Configuring the Central Base Station
### Downloading Software
* Download Ubuntu 18.04 LTS Operating System onto PC
* Download [Python 3](https://www.python.org/downloads/) and [set the default configuration](https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu) to the most recent Python 3 that is downloaded 
* Download [VSCode](https://code.visualstudio.com/download) and download Python backage in extensions. Extensions is the last tab on the right side of the IDE. 
* Download [Nvidia Drivers] (https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu) and make sure that CUDA and the Nvidia Drivers are added to your bin. Follow these instructions exactly!

### Downloading Libraries
* All pip commands can be downloaded using the command line in Ubuntu (or Windows/Mac) using the command sudo pip install. An [example](https://askubuntu.com/questions/95037/what-is-the-best-way-to-install-python-packages) of using pip to install packages is shown in the link. 
* **IMPORTANT** - all pip packages must be downloaded to the most recent Python installation. Ensure that pip is download to Python 3 rather than Python 2. The current version of pip can be determined by using the command `pip --version` in the terminal. If if is pointing to the wrong version, one possible fix is shown [here](https://askubuntu.com/questions/412178/how-to-install-pip-for-python-3-in-ubuntu-12-04-lts).

* Build [OpenCV](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) from source.  This will give OpenCV with CUDA support. 
* Download [imutils](https://pypi.org/project/imutils/) package from pip. This is used to detecting edges and contours when using OpenCV
* Download [numpy](https://pypi.org/project/numpy/) package from pip
* Download [Flask](https://pypi.org/project/Flask/) package from pip
* Download [psutil](https://pypi.org/project/psutil/) package from pip

## Running the program with a GoPro
* Connect your computer to the Wifi of the GoPro. 
* Pull `GPUAccelerated` from this repository to a local folder on your computer. 
* Make sure you are in the virutal environment where OpenCV with CUDA support was set up. If the above tutorial was followed this should be done by typing `workon opencv_cuda` 
* Navigate to the `EagleEye` folder in a terminal and run `python3 yolo_object_detection.py`
* To access the webpage, go to a web browser, type in `localhost:5000` to the URL and hit enter. The output video after processing will be shown on the screen.
* To close the host, close out of the terminal to stop the Python program. The webpage at `localhost:5000` should not be outputting video after closing the terminal.
