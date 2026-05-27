import paho.mqtt.client as mqtt
import redis
import os
import json
import sys
import time

REDIS_HOST = os.getenv("REDIS_HOST", "redis-svc")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto-svc")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

print(f"Starting subscriber... Redis: {REDIS_HOST}:{REDIS_PORT}, MQTT: {MQTT_BROKER}:{MQTT_PORT}", flush=True)

# Redis 连接
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD or None, decode_responses=True)
    r.ping()
    print("Redis connected.", flush=True)
except Exception as e:
    print(f"Redis error: {e}", flush=True)
    sys.exit(1)

# MQTT 回调
def on_connect(client, userdata, flags, rc):
    print(f"MQTT connected with result code {rc}", flush=True)
    client.subscribe("sensor/data")
    print("Subscribed to sensor/data", flush=True)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        print(f"Received: {data}", flush=True)
        r.lpush("sensor_data", json.dumps(data))
        r.ltrim("sensor_data", 0, 99)
    except Exception as e:
        print(f"Error processing message: {e}", flush=True)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("Connecting to MQTT...", flush=True)
    client.loop_forever()
except Exception as e:
    print(f"MQTT connection error: {e}", flush=True)
    sys.exit(1)