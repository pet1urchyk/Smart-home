import network
import json
import time
import machine
import dht
from umqtt.simple import MQTTClient

SSID = "HUAWEI_B818_6389"
PASSWORD = "QB33JJT7M7G"

MQTT_HOST = "192.168.8.171"
MQTT_PORT = 1883

CLIENT_ID = "esp32-1"
TOPIC = b"sensors/2/events"

INTERVAL = 5

PIR_PIN = 14
pir = machine.Pin(PIR_PIN, machine.Pin.IN)
sensor = dht.DHT22(machine.Pin(4))


def conn_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(SSID, PASSWORD)
        while not wlan.isconnected():
            time.sleep(0.5)
    print("wifi ok:", wlan.ifconfig())


def conn_mqtt():
    client = MQTTClient(CLIENT_ID, MQTT_HOST, port=MQTT_PORT)
    client.connect()
    print("mqtt is conneted")
    return client


def create_msg_motion():
    try:
        payload = {
            "sensor_type": "motion",
            "detected": True if pir.value() == 1 else False,
        }
        return json.dumps(payload)
    except OSError as e:
        print(f"[SENSOR ERROR] {e}")
        return None


def create_msg_tempandhum():
    try:
        sensor.measure()
        payload = {
            "sensor_type": "temp_hum",
            "temperature": round(sensor.temperature(), 1),
            "humidity": round(sensor.humidity(), 1),
        }
        return json.dumps(payload)
    except OSError as e:
        print(f"[SENSOR ERROR] {e}")
        return None


def main():
    conn_wifi()
    client = conn_mqtt()
    while True:
        try:
            payload_m = create_msg_motion()
            payload_th = create_msg_tempandhum()

            if payload_m is not None:
                client.publish(TOPIC, payload_m)
                print("msg:", payload_m)

            if payload_th is not None:
                client.publish(TOPIC, payload_th)
                print("msg:", payload_th)

        except Exception as e:
            print(f"error:{e}")
            time.sleep(2)
            try:
                conn_wifi()
                client = conn_mqtt()

                payload_m = create_msg_motion()
                payload_th = create_msg_tempandhum()
                if payload_m is not None:
                    client.publish(TOPIC, payload_m)
                    print("msg:", payload_m)
                if payload_th is not None:
                    client.publish(TOPIC, payload_th)
                    print("msg:", payload_th)

            except Exception as e2:
                print(f"connection failed:{e2}")
        time.sleep(INTERVAL)


main()