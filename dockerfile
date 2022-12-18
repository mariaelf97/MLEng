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
     libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade mysql-connector-python

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt
COPY HW/batting_average.sql .
COPY HW/basbeall_fresh.py .
COPY HW/brute_force.py .
COPY HW/correlation.py .
COPY HW/define_data_type.py .
COPY HW/mean_of_response.py .
COPY HW/model.py .
COPY HW/predictor_response_plots.py .
COPY HW/mybashscript.sh .
COPY HW/regression.py .

# Run app
CMD ./mybashscript.sh