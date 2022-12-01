
FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
#COPY yoursql file
#COPY your bash script
RUN pip3 install --compile --no-cache-dir -r requirements.txt


# Run app
CMD sleep 999999
#bash script here
