TOPICS_FROM_BRAIN = {
    
    "START_RECORD": "/start_record",
    "STOP_RECORD": "/stop_record",
    "START_RECOGNIZE": "/robot/from_brain/voice/recognize_voice",
    "IS_ALIVE": "/is_alive",
}

TOPICS_TO_BRAIN = {
    "APP_ALIVE": "/app_alive",
    "VOICE_RECOGNIZED": "/robot/to_brain/voice/voicefound",
    "VOICE_VECTOR_BUILT": "/voice_vector_built",
}

port = 1883
broker = "localhost"
VOICE_CLIENT = None

msg_alive = {"app":"voice_recognition","alive":True}
