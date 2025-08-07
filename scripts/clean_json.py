import json

input_path = "/home/mat/Documents/voice_ID/data/long_audio/adina_sagi.json"
output_path = "/home/mat/Documents/voice_ID/data/long_audio/adina_sagi_no_words.json"

with open(input_path, "r") as f:
    data = json.load(f)

# Remove all "words" keys from phrases
if "phrases" in data:
    for phrase in data["phrases"]:
        if "words" in phrase:
            del phrase["words"]

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"All 'words' sections removed. Output saved to {output_path}.")