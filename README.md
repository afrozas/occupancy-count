# Occupancy Count

Counting number of occupants in a room with image recoginition tools from Caffe2.

# Requirements
	
The project runs on python3 and uses pip3 as python package manager.

### Installing python3 and pip
To see which version of Python 3 you have installed, open a command prompt and run

```
$ python3 --version
```

If you are using Ubuntu 16.10 or newer, then you can easily install Python 3.6 with the following commands:

```
$ sudo apt-get update
$ sudo apt-get install python3.6
```

To install pip, download get-pip.py :

```
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
Then run the following:

```
$ python3 get-pip.py
```

### Project dependencies

Dependencies are compiled in **requirements.txt** file. To install them, run:

```
$ pip install -r requirements.txt
```

Installing `caffe` and `sci-stack libraries` :

Caffe2 requires Anaconda for installation. Refer to the official guide for steps: 

[Installing Caffe2 with Anaconda](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=prebuilt)

With Anaconda installed, run:
```
$ conda install -c caffe2 caffe2
```


In case of issues, refer to official documentation of [Caffe2](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=prebuilt) and [Jupyter](http://jupyter.org/install) for installations.


		