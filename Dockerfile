FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3

# update pip
RUN pip3 install --upgrade pip

WORKDIR /fedoras
COPY . /fedoras

# You'll likely need to upgrade ray to a more recent version
RUN python3 -m pip install -r requirements.txt
