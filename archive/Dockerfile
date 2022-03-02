FROM python:3.7.12-slim-bullseye

# ========= ROOT COMMANDS =========
# `jupyter lab` won't run as root, and root is bad practice.
# So we create a regular user.
RUN apt update
RUN apt install sudo
# Create user; 
RUN useradd --create-home --password RapidRigorReproduce aiqc_usr
# Make that user an admin; can't install apt-get dependencies without `sudo` prefix otherwise.
RUN usermod -aG sudo aiqc_usr
# Give that user permissions within their home directory, /var for apt-get, /usr/local for python packages.
RUN chown -R aiqc_usr /home/aiqc_usr /var /usr/local /usr/bin/dpkg /var/cache
# Switch to that user; root user's apt-get binaries are not shared w new user.
RUN su - aiqc_usr

# ========= USER COMMANDS =========
# Can't install nodejs without updating package manager.
# Only need to use the pw once when running sudo commands.
RUN echo "RapidRigorReproduce" | sudo -S apt update
RUN sudo apt upgrade -y
RUN sudo apt update

# Create a place to mount the source code so that it can be imported.
RUN mkdir /home/aiqc_usr/AIQC

# --- Binaries ---
# Add the registry that contains node
RUN sudo apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
# Install node
RUN sudo apt -y install nodejs

# For Sphinx documentation.
RUN sudo apt -y install pandoc

# --- Python packages ---
# I think `--no-cache-dir` is causing problems with tensorflow dependencies.
RUN pip install --no-cache-dir --default-timeout=100 --upgrade pip
# Developer packages
# if reqs.txt doesn't change then it will used a cached layer.
# Contains JupyterLab and I want this installed prior to plotly.
# Docker paths are can't access parent directories.
COPY requirements_dev.txt /
RUN pip install --no-cache-dir --default-timeout=100 -r requirements_dev.txt 
RUN rm requirements_dev.txt

# User packages
# Installing plotly>=5.0.0 includes the prebuilt jupyter extension.
COPY requirements.txt /
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt 
RUN rm requirements.txt
