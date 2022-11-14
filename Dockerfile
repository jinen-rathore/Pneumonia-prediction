FROM centos:7

RUN yum install python3 -y

COPY . .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt