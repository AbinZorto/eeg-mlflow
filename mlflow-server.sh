#!/bin/bash

PORT=5000
conda init
conda activate eeg-env
if [ "$1" == "start" ]; then
    echo "Starting MLflow UI on port $PORT..."
    mlflow ui --port $PORT &
    echo $! > mlflow.pid
    echo "MLflow started with PID $(cat mlflow.pid)"
elif [ "$1" == "stop" ]; then
    echo "Stopping MLflow..."
    fuser -k 5000/tcp
    if [ -f mlflow.pid ]; then
        kill $(cat mlflow.pid) && rm mlflow.pid
        echo "MLflow stopped."
    else
        echo "No PID file found. Attempting to kill gunicorn manually..."
        pkill -f "gunicorn -b 127.0.0.1:$PORT"
    fi
else
    echo "Usage: $0 {start|stop}"
fi
