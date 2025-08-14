import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample items (could be movies, products, etc.)
data = {
    "title": [
        "Space Odyssey",
        "Romantic Escape",
        "Deep Sea Documentary",
        "Future Tech Thriller",
        "Stand-up Comedy Night",
        "Alien Invasion",
        "Historical War Drama",
        "AI and Robotics",
        "Time Travel Adventure",
        "Mind-bending Mystery",
        "Cooking with Passion",
        "Nature and Wildlife",
        "The Science of Everything"
    ],
    "description": [
        "An adventure in outer space exploring galaxies and planets.",
        "A love story set on a tropical island during summer vacation.",
        "The mysterious life under the ocean, from coral reefs to deep-sea creatures.",
        "A futuristic thriller involving advanced technology and virtual reality.",
        "A humorous night with the best stand-up comedians across the country.",
        "Earth defends itself against a hostile alien civilization attacking major cities.",
        "A moving tale set during World War II with heroic sacrifices and strategy.",
        "How artificial intelligence and robotics are transforming the modern world.",
        "A mind-twisting adventure through wormholes and time paradoxes.",
        "A detective uncovers secrets in a psychological thriller full of plot twists.",
        "Delicious recipes, cooking techniques, and the joy of food preparation.",
        "Beautiful visuals of forests, animals, and ecosystems in the wild.",
        "A simplified explanation of complex scientific concepts for everyone."
    ]
}

df = pd.DataFrame(data)

# Build TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])


def recommend(query: str, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked_indices = sims.argsort()[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        results.append((df.loc[idx, "title"], float(sims[idx])))
    return results

def main():
    print("AI Recommendation System")
    user_input = input("Describe your interest (e.g., space future technology): ")
    recs = recommend(user_input)
    if recs:
        print("Top recommendations:")
        for title, score in recs:
            print(f"- {title} (score: {score:.2f})")
    else:
        print("No recommendations found. Try different keywords.")

if __name__ == "__main__":
    main()