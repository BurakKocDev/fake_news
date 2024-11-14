import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox, ttk


df = pd.read_csv('train.csv')
df.fillna(" ", inplace=True)
df['content'] = df['title'] + " " + df['author']

# CountVectorizer ve TfidfTransformer 
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_counts = count_vectorizer.fit_transform(df['content'].values)
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Model
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'].values, test_size=0.2, random_state=49)
model = LogisticRegression()
model.fit(X_train, y_train)


train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

# gui
root = Tk()
root.title("Fake News Detection")
root.geometry("650x500")
root.configure(bg="#2C3E50")


title_label = Label(root, text="Fake News Detection Application", font=("Helvetica", 20, "bold"), fg="#ECF0F1", bg="#2C3E50")
title_label.pack(pady=20)

accuracy_label = Label(root, text=f"Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}", font=("Arial", 12), fg="#1ABC9C", bg="#2C3E50")
accuracy_label.pack()

label = Label(root, text="Enter a news headline or article to check if it is real or fake:", font=("Arial", 14), fg="#ECF0F1", bg="#2C3E50")
label.pack(pady=20)

entry = Text(root, height=10, width=60, font=("Arial", 12))
entry.pack(pady=10)

# Haber doğruluğunu kontrol eden fonksiyon
def check_news():
    user_input = entry.get("1.0", "end-1c")
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    
    user_input_counts = count_vectorizer.transform([user_input])
    user_input_tfidf = tfidf_transformer.transform(user_input_counts)
    
    prediction = model.predict(user_input_tfidf)
    if prediction[0] == 1:
        result_text = "The news is likely to be Real."
    else:
        result_text = "The news is likely to be Fake."
    messagebox.showinfo("Prediction Result", result_text)


style = ttk.Style(root)
style.configure("Accent.TButton", font=("Helvetica", 14, "bold"), foreground="#FFFFFF", background="#2980B9", padding=10)

button = ttk.Button(root, text="Check News", command=check_news, style="Accent.TButton")
button.pack(pady=20)

root.mainloop()
