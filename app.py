import streamlit as st
import torch
import random
import json
import torch.nn as nn
from model import NeuralNet  # Ensure this import works by placing model.py in the same directory
from nltk_utils import tokenize, stem, bag_of_words
import os

# Ensure paths are correct for deployment
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "data.pth")
intents_path = os.path.join(current_directory, "intents.json")

# ----------------------------
# Load the trained chatbot model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents (ensure file exists in the correct path)
with open(intents_path, "r") as f:
    intents = json.load(f)

# Load trained model parameters (ensure file exists in the correct path)
if os.path.exists(model_path):
    # Modify torch.load to use map_location to load on CPU if CUDA is not available
    data = torch.load(model_path, map_location=device)
else:
    st.error(f"Model file '{model_path}' not found!")
    st.stop()

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# ----------------------------
# Helper function to get chatbot response
# ----------------------------
def get_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).to(device).unsqueeze(0)

    # Model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Confidence threshold
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that. Can you rephrase?"

# ----------------------------
# Streamlit UI for chatbot
# ----------------------------
st.title("🤖 Chat with Cypher")
st.write("Ask me anything about Vanshaj Raghuvanshi!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    # User message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Bot response
    response = get_response(user_input)
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
