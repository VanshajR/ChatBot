import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Axiom"
context = {"talking_about": None}  # Initialize context with talking_about key

def set_context(tag, context_dict):
    # If the tag indicates we are talking about Vanshaj, set the context accordingly
    if "vanshaj" in tag.lower():
        context_dict["talking_about"] = "vanshaj"
    # No else condition to reset context to None if "vanshaj" is not mentioned

def resolve_pronouns(sentence, context_dict):
    # If we are talking about Vanshaj, replace "he" and "him" with "Vanshaj"
    if context_dict.get("talking_about") == "vanshaj":
        sentence = sentence.replace(" he ", " Vanshaj ").replace(" him ", " Vanshaj ")
        sentence = sentence.replace("^he ", "Vanshaj ").replace(" him$", " Vanshaj")
        sentence = sentence.replace(" he$", " Vanshaj").replace("^he ", "Vanshaj ")
        sentence = sentence.replace(" him$", " Vanshaj").replace("^him ", "Vanshaj ")
    return sentence

def get_intent_tag(sentence):
    # Tokenize the sentence and get the intent tag using the trained model
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    return tags[predicted.item()]

def get_response(intent_tag):
    # Retrieve a random response based on the intent tag
    for intent in intents['intents']:
        if intent_tag == intent["tag"]:
            return random.choice(intent['responses'])
    return None

print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Resolve pronouns in the user input based on the current context
    user_input = resolve_pronouns(user_input, context)
    
    # Get the intent tag for the user input
    intent_tag = get_intent_tag(user_input)
    
    # Update context based on the intent tag
    set_context(intent_tag, context)

    # Get a response based on the intent tag
    response = get_response(intent_tag)
    
    if response:
        # Resolve pronouns in the response based on the current context
        response = resolve_pronouns(response, context)
        print(f"{bot_name}: {response}")
    else:
        print(f"{bot_name}: I do not understand...")
