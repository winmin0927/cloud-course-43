import os
from flask import Flask, jsonify
import redis
import pandas as pd
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD if REDIS_PASSWORD else None,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    r.ping()
    redis_available = True
    logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    redis_available = False
    logger.warning(f"Redis unavailable: {e}")


@app.route("/api/ping", methods=["GET"])
def ping():
    logger.info("GET /api/ping - request received")
    response = {"status": "ok"}
    if redis_available:
        response["redis"] = "connected"
    return jsonify(response)


@app.route("/api/info", methods=["GET"])
def info():
    return jsonify({
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
        "redis_available": redis_available,
        "pandas_version": pd.__version__,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
