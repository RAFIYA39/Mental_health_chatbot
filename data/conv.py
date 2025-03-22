import json
import yaml

# Load JSON data
with open("dataset.json", "r") as file:
    data = json.load(file)

nlu_data = {"version": "3.1", "nlu": []}
domain_data = {"version": "3.1", "intents": [], "responses": {}}

# Convert JSON to NLU & domain.yml format
for intent in data["intents"]:
    intent_name = intent["tag"]
    
    # Add intent to domain.yml
    domain_data["intents"].append(intent_name)

    # Convert patterns to nlu.yml format
    nlu_data["nlu"].append({
        "intent": intent_name,
        "examples": "\n".join([f"- {p}" for p in intent["patterns"]])
    })

    # Convert responses to domain.yml format
    domain_data["responses"][f"utter_{intent_name}"] = [{"text": r} for r in intent["responses"]]

# Save as nlu.yml
with open("data/nlu.yml", "w") as file:
    yaml.dump(nlu_data, file, allow_unicode=True, default_flow_style=False)

# Save as domain.yml
with open("domain.yml", "w") as file:
    yaml.dump(domain_data, file, allow_unicode=True, default_flow_style=False)

print(" JSON dataset converted to Rasa format successfully!")
