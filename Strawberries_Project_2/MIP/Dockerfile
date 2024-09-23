# Use the official Minizinc image as the base
FROM minizinc/minizinc:latest

# Set the working directory in the container
WORKDIR /src

# Copy the current directory contents into the container at /src
COPY . .

# Update the package lists and install necessary packages
RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y libffi-dev gcc python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install the dependencies
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

ENV GRB_LICENSE_FILE=/src/gurobi.lic

# Command to run your application
CMD /opt/venv/bin/python /src/MIP.py
