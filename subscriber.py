import asyncio
import json
from aiomqtt import Client
import httpx
import os

API_URL = "http://web:8000"
MQTT_HOST = os.getenv("MQTT_HOST", "192.168.8.171")

async def main():
    async with httpx.AsyncClient() as http_client:
        async with Client(hostname=MQTT_HOST,port=1883) as client:
            print(["CONNECTED"])

            await client.subscribe("sensors/#")
            print("Succesfully subscribed to sensors/#")

            async for message in client.messages:
                topic = str(message.topic)
                payload = message.payload.decode("utf-8")
                print(f"Recieved: Topic: {topic}| Data: {payload}")

                parts = topic.split("/")
                if len(parts) == 3 and parts[0] == "sensors" and parts[2] == "events":
                    try:
                        sensor_id = int(parts[1])
                        data = json.loads(payload)

                        response = await http_client.post(
                                f"{API_URL}/sensors/{sensor_id}/events",
                                json=data
                            )
                        
                        if response.status_code >= 400:
                            print(f"[API ERROR] Status {response.status_code}: {response.text}")
                        else:
                            print(f"[OK] Sensor {sensor_id}: saved event (status {response.status_code})")

                    except json.JSONDecodeError:
                        print("[ERROR] Wrong JSON structure in payload")
                    except ValueError:
                        print(f"[ERROR] Id isnt ID: {parts[1]}")
                    except httpx.RequestError as e:
                        print(f"[API ERROR] Server is down {e}")
                    
            
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Connection closed")