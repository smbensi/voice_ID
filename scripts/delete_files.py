import os

folder_path = "/home/mat/Documents/voice_ID/huberman/2"
for f in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, f)) and "mono" not in f:
        os.remove(os.path.join(folder_path, f))
        print(f"Deleted: {f}")