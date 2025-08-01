import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Knowledge base (you can expand this)
# Expanded knowledge base for the AI chatbot
KB = [
    # Greetings
    "Hello! How can I help you today?",
    "Hi there! I am your CODSOFT project assistant.",
    "Goodbye! Best of luck with the internship.",

    # Project guidance
    "You are working on three tasks: Rule-Based Chatbot, Tic-Tac-Toe AI, and Recommendation System.",
    "To run the chatbot, activate the virtual environment and execute python chatbot_ai.py.",
    "The Tic-Tac-Toe AI uses Q-Learning or Minimax to make smart moves.",
    "The Recommendation System suggests items based on your interests using TF-IDF and cosine similarity.",

    # Git & Setup
    "To activate the virtual environment in PowerShell: .\\.venv\\Scripts\\Activate.ps1",
    "If PowerShell blocks activation, set execution policy using Set-ExecutionPolicy -Scope CurrentUser RemoteSigned.",
    "You can commit changes to GitHub using git add ., git commit -m 'message', and git push.",

    # Python & Pip
    "Install all project dependencies using pip install -r requirements.txt.",
    "Upgrade pip with python -m pip install --upgrade pip.",

    # Internship Notes
    "These tasks are part of the CODSOFT internship program.",
    "Make sure to document your tasks and upload your code to GitHub for evaluation.",

    # Extra helpful responses
    "I can assist you with running tasks, project setup, or explaining the project structure.",
    "Try exploring the Tic-Tac-Toe game to see how AI learns to play perfectly.",
    "The recommendation system works best when you provide keywords like 'space', 'romance', or 'comedy'."
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast

# Precompute KB embeddings
kb_embeddings = model.encode(KB, convert_to_tensor=False)

def retrieve_response(user_input: str, top_k=1):
    inp_emb = model.encode([user_input], convert_to_tensor=False)
    sims = cosine_similarity(inp_emb, kb_embeddings)[0]
    best_idx = np.argmax(sims)
    score = sims[best_idx]
    if score < 0.4:  # threshold for fallback
        return "Sorry, I don't have a good answer for that. Can you rephrase?"
    return KB[best_idx]

def main():
    print("AI Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bot: Goodbye!")
            break
        response = retrieve_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()