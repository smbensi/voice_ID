import paho.mqtt.client as mqtt

from speaker_verification import LOGGER
from speaker_verification.settings import mqtt_settings

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag = True
        if client._client_id.decode('UTF-8') == "stream":
            client.subscribe(mqtt_settings.TOPICS_TO_BRAIN["APP_ALIVE"])
        else:
            topics_to_sub = [(topic,2) for topic in mqtt_settings.TOPICS_FROM_BRAIN.values()]
            client.subscribe(topics_to_sub)
            client.subscribe(mqtt_settings.TOPICS_TO_BRAIN["VOICE_RECOGNIZED"])
            client.subscribe(mqtt_settings.TOPICS_TO_BRAIN["VOICE_VECTOR_BUILT"])
            # client.subscribe(mqtt_settings.TOPICS_TO_BRAIN["APP_ALIVE"])
    else:
        print("Bad connection, Error code: " + rc)
        client.loop_stop()


def on_disconnect(client, userdata, rc):
    print(f"Disconnected and the return code is {rc}")
    


def connect_to_broker():
    client.connect(mqtt_settings.broker, mqtt_settings.port, 120)
    LOGGER.debug("#### successfully connected to mqtt   ####")
    client.loop_start()


def init_mqtt_connection(name="voice_verif"):
    mqtt.Client.connected_flag = False
    global client
    client = mqtt.Client(client_id=name, clean_session=True)
    
    # client.username_pw_set(username="nir", password="xtend_m2")
    client.on_disconnect = on_disconnect
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_to_broker = connect_to_broker
    client.connect_to_broker()
    LOGGER.debug(f"client is {client._client_id.decode('UTF-8')}")
    return client

def on_message(client, userdata, message):

    if message.topic == mqtt_settings.TOPICS_FROM_BRAIN["IS_ALIVE"]:
        mqtt_settings.VOICE_CLIENT.publish(mqtt_settings.TOPICS_TO_BRAIN["APP_ALIVE"],json.dumps(mqtt_settings.msg_alive))
