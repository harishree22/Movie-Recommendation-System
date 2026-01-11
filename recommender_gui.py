import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- DATA PROCESSING ----------------

file_path = os.path.join(os.path.dirname(__file__), "data.csv")
data = pd.read_csv(file_path)

data["content"] = data["genre"] + " " + data["overview"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["content"])
similarity = cosine_similarity(tfidf_matrix)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend():
    movie_title = entry.get().strip()

    if movie_title not in data["title"].values:
        messagebox.showerror("Error", "Movie not found in database!")
        return

    idx = data[data["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    result_box.delete(0, tk.END)
    for i in scores[1:6]:
        result_box.insert(tk.END, data.iloc[i[0]]["title"])

# ---------------- GUI ----------------

root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x400")
root.configure(bg="#2c3e50")

tk.Label(root, text="🎬 Movie Recommendation System",
         font=("Arial", 16, "bold"), fg="white", bg="#2c3e50").pack(pady=15)

entry = tk.Entry(root, width=30, font=("Arial", 12))
entry.pack(pady=10)
entry.insert(0, "Enter movie name")

tk.Button(root, text="Get Recommendations",
          command=recommend, bg="#1abc9c",
          fg="white", font=("Arial", 11, "bold")).pack(pady=10)

tk.Label(root, text="Recommended Movies:",
         font=("Arial", 12, "bold"),
         fg="white", bg="#2c3e50").pack(pady=10)

result_box = tk.Listbox(root, width=40, height=8, font=("Arial", 11))
result_box.pack(pady=5)

root.mainloop()
