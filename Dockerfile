FROM python:3.10-slim-buster

COPY . /workspace

WORKDIR /workspace/

# Default installs.
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    gcc \
    make

# Create virtualenv.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install pip --upgrade && \
    /opt/venv/bin/python -m pip install -r /workspace/requirements.txt

# Make entrypoint executable.
RUN chmod +x entrypoint.sh

# Run the app.
CMD ["./entrypoint.sh"]