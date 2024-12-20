
# Chatbot with Streamlit Interface

This project implements a chatbot capable of answering questions about me. The chatbot leverages a custom-trained model and is deployed using a **Streamlit** web application for an intuitive user interface. Check it out here: [Cypher](https://cypher-chat.streamlit.app/)

## Features

- **Custom Intent Recognition**: Understands user intents defined in `intents.json`.
- **Deep Learning Model**: Uses a neural network for processing and responding to user inputs.
- **Streamlit Integration**: Provides a user-friendly web interface for interaction.
- **Expandable**: Easily customizable intents and responses for additional functionality.

---

## Repository Structure

| File/Folder       | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `app.py`          | The main file for running the Streamlit-based chatbot application.         |
| `chat.py`         | Contains core logic for processing user queries and generating responses.  |
| `data.pth`        | Saved PyTorch model file containing the trained chatbot model.             |
| `intents.json`    | JSON file defining chatbot intents and corresponding responses.            |
| `model.py`        | Script for building and training the neural network model.                 |
| `nltk_utils.py`   | Utility functions for text preprocessing, including tokenization and stemming. |
| `requirements.txt`| List of required Python libraries for setting up the environment.          |

---

## Installation

Follow the steps below to set up the project on your local machine:

### Prerequisites

Ensure you have Python 3.8+ installed.

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VanshajR/chatbot-with-streamlit.git
   cd chatbot-with-streamlit
   ```

2. **Install dependencies**:
   Use the provided `requirements.txt` to install all necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   Start the chatbot interface with Streamlit:
   ```bash
   streamlit run app.py
   ```

4. **Interact with the bot**:
   Open the Streamlit app URL in your browser and start chatting with the bot.

---

## Usage

- Modify the `intents.json` file to customize the chatbot's understanding of queries and responses.
- Train the chatbot model using `model.py` if you make significant changes to intents or data.

---

## Deployment

The Streamlit app can be deployed using platforms like **Streamlit Community Cloud**, **Heroku**, or **Docker**.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
