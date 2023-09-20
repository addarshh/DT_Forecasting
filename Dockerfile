FROM continuumio/miniconda3

MAINTAINER Michael Catalano "Michael.Catalanot@wwt.com"
MAINTAINER Patrick McDermott "Patrick.McDermott@wwt.com"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev


RUN mkdir /scratch

#Set the working directory
WORKDIR /DTFORECASTING


# We copy just the requirements.txt first to leverage Docker cache
COPY ./pip_requirements.txt /DTFORECASTING/pip_requirements.txt

RUN pip install -r pip_requirements.txt

COPY . /DTFORECASTING

ENTRYPOINT [ "python" ]

