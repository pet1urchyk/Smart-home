import aiomqtt
import asyncio

import json
import os

import random

basetemp = 24
basehumidity = 67

addr = os.getenv("MQTT_HOST", "mosquitto")
sensor_id = os.getenv("SENSOR_ID", "1")

async def main():
    async with aiomqtt.Client(hostname=addr, port=1883) as client:
        msg_count = 0
        
        while True:
            payload = {"sensor_type":"temp_hum","temperature":round(basetemp + random.uniform(-0.5, 0.5), 1),
                       "humidity":round(max(0.0,min(100.0, basehumidity + random.uniform(-0.5, 0.5))), 1)}
            
            payload = json.dumps(payload)
            await client.publish(f"sensors/{sensor_id}/events",payload)

            await asyncio.sleep(5)

            msg_count += 1
            if msg_count % 5 == 0:
                print(f"It`s our {msg_count} message, and the data is {payload}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Connection closed")