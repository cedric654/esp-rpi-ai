# Installation guide

## Section 1 - Setting up environment

### Step 1 : Update Raspberry Pi

First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:

```bash
sudo apt-get update
sudo apt-get dist-upgrade
```

### Step 2 : Create a new project folder

Second, create a new folder called the name you want by using the following command. Issue:

```bash
mkdir FOLDER_NAME
```

### Step 2 : Dowload GitHub repository

Next, clone this GitHub repository by usign the following command. The repository contains the scripts we'll use to run TensorFlow Lite, as well as a shell script that will make installing everything easier. Issue:

```bash
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
```

### Step 3 : Install virtualenv

Change directory by issuing:

```bash
cd FILE_NAME_YOU_CREATED_BEFORE
```

Install virtualenv by issuing:

```bash
sudo pip3 install virtualenv
```

Then, create the "esp-rpi-ai-env" virtual environment by issuing:

```bash
python3 -m venv esp-rpi-ai-env
```

Now, you need to issue this command to activate the environment every time you open a new terminal window. You can tell when the environment is active by checking if (esp-ai-env) appears before the path in your command prompt, as shown in the screenshot below.

```bash
source tflite1-env/bin/activate
```

### Step 4 : Install TensorFlow Lite dependencies, OpenCV and Flask

To make things easier, a shell script will automatically download and install all the packages and dependencies. Run it by issuing:

```bash
bash get_pi_requirements.sh
```

### Step 5 : Verify if model is correctly configured

Make sure the model is correctly linked to the main file by issuing :

```bash
nano main.py
```

## Section 2 - Run the Tensorflow Lite model and Web server

### Step 1 : Start server

It's time to see the TFLite object detection model in action! First, free up memory and processing power by closing any applications you aren't using. Also, make sure you have your webcam or Picamera plugged in.

If you follow all steps correctly it's supposed to work without any error

Now it's time, issue this command for make it works :

```bash
python main.py
```
