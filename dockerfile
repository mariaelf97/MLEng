
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
COPY HW/test.sql .
COPY HW/mybashscript.sh .

# Run app
#RUN chmod +x HW/mybashscript.sh
CMD ./mybashscript.sh

#docker system prune -a --volumes to clean cache