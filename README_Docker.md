# ============ BUILDING IMAGE ============

`brew install docker`
Via the UI: (1) login, (2) set sensible hardware resource constraints.

# Overview
- `Dockerfile` is used to locally build the container image.
- Whereas the `docker-compose.yml` file is used to run the container with specified settings
  as opposed to passing all of those arguments into the `docker run ...` command.


# Files for Build
Put all of these files in the same directory e.g. '/desktop/aiqc/docker':
    1. Your python packages e.g. 'requirements.txt' and 'requirements_dev.txt'
    2. Dockerfile
    3. docker-compose.yml


# Build the Image
To locally build the image, navigate to that directory and run:
`docker build --tag <your_dockerhub_account>/<your_image_name>:<your_tag> .` 
The `.` tells docker to look in local folder for files.


# Run the Service
You'll want to test the image before you push it as uploading can take a long time.
To run the service using the settings in docker-compose.yml:
`cd ~/desktop/AIQC/docker`
`docker-compose up`
http://127.0.0.1:8888/lab

If you want to run it manually to inspect the file system:
`docker run -it <image_id> /bin/bash` # bash shell at root directory.


# Push the Image to DockerHub
Create the remote repo: https://hub.docker.com/repository/create?namespace=hashrocketsyntax
`docker push <your_dockerhub_account>/<your_repo>:<your_tag>`


# ============ AWS ECR ============

# Initial configuration of AWS CLI w Docker

IAM user = "user_aiqc"
IAM group = "group_aiqc"

Download credentials CSV generated when creating the IAM user (saved to 'GoogleDrive/AIQC').

`brew update`
`brew doctor`
`brew install awscli`

$ aws configure
Default region name: us-east-1
Default output format: table


Hook AWS ECR up to DockerHub
https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-authenticate-registry 

"When passing the Amazon ECR authorization token to the docker login command, use the value AWS for the username and specify the Amazon ECR registry URI"


`aws ecr get-login-password --region us-east-1 --output text | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com`

"Login Succeeded"

You can run `aws ecr get-login-password --region us-east-1 --output text` standalone to make sure it's working. It outputs a big multi-line token that looks that is not the regular AWS key/secret. It wanted "us-east-1" not "us-1-east".

```
aws ecr create-repository \
    --repository-name aiqc \
    --image-scanning-configuration scanOnPush=true \
    --region us-east-1
```

# Tagging and pushing a new image to AWS ECR: 

Just switch in the ECR repo's URL instead of the regular DockerHub repo. 

`docker tag hashrocketsyntax/aiqc:dev_dsktp_v3.0.2_py3.7.12-slim-bulls <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/aiqc:dev_dsktp_v3.0.2_py3.7.12-slim-bulls`

`<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/aiqc:dev_dsktp_v3.0.2_py3.7.12-slim-bulls`



# ========= WHEN RUNNING CONTAINER =========

# Expected folders on your desktop:
- Assumes that you have git cloned AIQC repo to `~/Desktop/AIQC`
- And have made these folders `mkdir ~/Desktop/AIQC_Docker_Workspace ~/Desktop/AIQC_Docker_Workspace/AppDirs`
^ Docker will mount those folders and create files within them.


# Editing files.
TLDR: changes on host not propagated to container, but changes in container are passed to host.
`--volumes` acts like a snapshot at the time of boot. 
If you truly want to edit files on the host, then you are supposed to restart container.
But here are some workarounds:

- You can programmatically `aiqc.setup()` & `aiqc.destroy_db()` from within Docker, and it will impact both container/host filesys.
- You can programmatically `!touch file` & `!rm file` from within Docker, and it will impact both container/host filesys.
- You can install packages with `--user` from the Jupyter Terminal.
- You can edit the documentation in the Jupyter Editor 
- You can build the documentation in the Jupyter Terminal.
- However, you can edit the AIQC module's source code on the host, and `import aiqc` will refresh it.

- If you only edit/delete files on the host, then Docker's filesys gets out of sync.
- The Jupyter UI does not seem to have permission to add/remove files. Despite the Docker OS user having chown permissions on '/home/aiqc_usr/'
# OSError: [Errno 18] Invalid cross-device link: b'/home/aiqc_usr/file' -> b'/home/aiqc_usr/.local/share/Trash/files/file'


# OS
`cat /etc/os-release`
PRETTY_NAME="Debian GNU/Linux 10 (buster)"
NAME="Debian GNU/Linux"
VERSION_ID="10"
VERSION="10 (buster)"
VERSION_CODENAME=buster
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"


# PYTHON
`python --version`
Python 3.8.12
# The python binary is the same for root and non-root users.
`which python`
/usr/local/bin/python


# EXISTING BINARIES
`dpkg -s git | grep Version`
Version: 1:2.20.1-2+deb10u3

`dpkg -s openssl | grep Version`
Version: 1.1.1d-0+deb10u7

`pip show pip`
Name: pip
Version: 21.2.4
Summary: The PyPA recommended tool for installing Python packages.
Home-page: https://pip.pypa.io/
Author: The pip developers
Author-email: distutils-sig@python.org
License: MIT
Location: /usr/local/lib/python3.8/site-packages


# APPDIRS OF NON-ROOT.
`python`
>>> import appdirs
>>> appdirs.user_data_dir()
'/home/aiqc_usr/.local/share'
