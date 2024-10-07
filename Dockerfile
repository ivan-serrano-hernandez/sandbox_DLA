FROM dustynv/l4t-pytorch:r36.3.0-cu124

WORKDIR /app
COPY . /app

# Add manually ubuntu public key
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 1A127079A92F09ED


# Install necessary dependencies including gcc
RUN apt-get update \
    && apt-get install -y wget build-essential git cmake libzmq3-dev pkg-config curl vim python3 python3-pip sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt


#
# Setup environment variables
#

ENV TERM xterm-256color
CMD ["bash", "-l"]


