FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3


# Install required packages
RUN apt-get update && \
    apt-get install -y \
    sox libsndfile1 ffmpeg  \
    libportaudio2 libportaudiocpp0 portaudio19-dev

RUN python3 -m pip install --upgrade pip wheel

RUN mkdir /code
WORKDIR /code

COPY ./speaker_verification/requirements.txt .
RUN pip install -r requirements.txt


COPY ./speaker_verification .
RUN pip install -e .

