# CODSOFT Internship Projects

This repository contains three AI/ML/Python-based tasks implemented as part of the CODSOFT internship assignment. Each task is self-contained in its own folder, with runnable code and with minimal dependencies.



## 📌 Task Details

### 🔹 Task 1: AI Chatbot (Retrieval-Based)

* **Location:** `task1_chatbot/chatbot.py`

* **Description:**
  A retrieval-based chatbot using sentence embeddings (`sentence-transformers`). It matches user input to the closest entry in a curated knowledge base and replies accordingly.

* **Run:**

  ```bash
  cd task1_chatbot
  python chatbot.py
  ```

* **Example Queries:**

  * `Hello`
  * `How do I run the project?`
  * `Tell me about the recommendation system`
  * `What is Tic-Tac-Toe AI?`
  * `Help with Git setup`

* **Customizing:**
  Expand or edit the `KB` list inside `chatbot_ai.py` to add domain-specific knowledge or more responses.

---

### 🔹 Task 2: Tic-Tac-Toe AI (Q-Learning)

* **Location:** `task2_tictactoe/tictactoe.py`

* **Description:**
  A self-learning Tic-Tac-Toe agent trained with Q-Learning against a random opponent. The agent persists its knowledge in `q_table.pkl` for reuse.

* **Run:**

  ```bash
  cd task2_tictactoe
  python tictactoe.py
  ```

* **Gameplay:**

  * You play as `O`
  * AI plays as `X`
  * Input your move by entering an index 0–8 corresponding to board position:

    ```
    0|1|2
    3|4|5
    6|7|8
    ```

* **Training:**

  * On first run, the agent trains itself (default 20,000 episodes).
  * After training, a Q-table is saved to `q_table.pkl` for faster subsequent startup.

---

### 🔹 Task 4: AI Recommendation System

* **Location:** `task4_recommendation/recommend.py`

* **Description:**
  Content-based recommendation system using TF-IDF vectorization and cosine similarity. Given user interest keywords, it ranks a curated set of items (titles + descriptions) by relevance.

* **Run:**

  ```bash
  cd task4_recommendation
  python recommend.py
  ```

* **Example Inputs:**

  * `space galaxy aliens`
  * `romance love story`
  * `technology AI robotics`
  * `ocean nature`
  * `comedy food`



## 🧰 Suggested Improvements

* **Chatbot:**

  * Replace retrieval with a generative small LLM (via `transformers`).
  * Add logging, context retention between turns.
  * Wrap in a simple web UI (Flask/FastAPI).

* **Tic-Tac-Toe:**

  * Add human vs human mode.
  * Replace opponent with a minimax agent for benchmarking.
  * Visualize board using a GUI (`tkinter`, web).

* **Recommendation:**

  * Incorporate user ratings and build a collaborative filtering layer.
  * Use real datasets (e.g., MovieLens) and evaluate with precision/recall.
  * Expose as REST API for integration.

---

## 📦 Dependencies Summary

| Task           | Key Libraries                                                       |
| -------------- | ------------------------------------------------------------------- |
| Chatbot        | sentence-transformers, transformers, torch, faiss-cpu, scikit-learn |
| TicTacToe      | numpy                                                               |
| Recommendation | scikit-learn, pandas                                                |

---