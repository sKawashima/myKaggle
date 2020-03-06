FROM python:3.7.4-slim-buster
RUN apt-get update -qy
RUN apt-get install -qy build-essential
RUN apt-get install -qy curl
RUN apt-get install -qy nodejs npm
RUN curl -L git.io/nodebrew | perl - setup
ENV PATH $HOME/.nodebrew/current/bin:$PATH
RUN echo 'export PATH=$HOME/.nodebrew/current/bin:$PATH' >> $HOME/.bashrc
RUN /bin/bash -c "source $HOME/.bashrc && nodebrew install-binary v12.13"
RUN /bin/bash -c "source $HOME/.bashrc && nodebrew use v12.13"
RUN mkdir /data_analysis
WORKDIR /data_analysis
COPY requirements.txt /data_analysis
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN jupyter labextension install @jupyterlab/plotly-extension
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip freeze > requirements.txt
COPY . /data_analysis
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser
