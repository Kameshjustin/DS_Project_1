import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Import for image handling
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
mail_data = pd.read_csv(r"D:\Email_spam_detection\mail_data.csv")

# Train a Logistic Regression model
X = mail_data['Message']
y = mail_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change the classifier to Logistic Regression
model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))  # Using Logistic Regression with increased iterations
model.fit(X_train, y_train)

# Function to classify the message entered by the user
def classify_message():
    user_input = text_area.get("1.0", tk.END).strip()
    
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    
    prediction = model.predict([user_input])[0]
    
    # Change "ham" to "non-spam"
    prediction_text = "non-spam" if prediction == 'ham' else "spam"
    result_label.config(text=f"Prediction: {prediction_text}", fg='white', bg='green' if prediction == 'ham' else 'red')

# Initialize the Tkinter window
root = tk.Tk()
root.title("Email Spam Detection")
root.geometry("1200x600")  # Adjusted window size to make space for both images

# Load and display the logo (update the path as needed)
logo_image_path = r"D:\Email_spam_detection\Images\em.png" # Update with the correct logo path
logo_image = Image.open(logo_image_path)
logo_image = logo_image.resize((100, 50))  # Resize the logo
logo_photo = ImageTk.PhotoImage(logo_image)

# Create a label to display the logo
logo_label = tk.Label(root, image=logo_photo)
logo_label.pack(pady=10)  # Positioned at the top

# Load and display the left image (update the path as needed)
left_image_path = r"D:\Email_spam_detection\Images\pic2.png"  # Update with the correct left image path
left_image = Image.open(left_image_path)
left_image = left_image.resize((300, 300))  # Resize the image to fit the left side
left_photo = ImageTk.PhotoImage(left_image)

# Load and display the right image (update the path as needed)
right_image_path = r"D:\Email_spam_detection\Images\bgs.png"  # Update with the correct right image path
right_image = Image.open(right_image_path)
right_image = right_image.resize((300, 300))  # Resize the image to fit the right side
right_photo = ImageTk.PhotoImage(right_image)

# Create a label to display the left image
left_label = tk.Label(root, image=left_photo)
left_label.place(x=0, y=100)  # Position at the left corner

# Create a label to display the right image
right_label = tk.Label(root, image=right_photo)
right_label.place(x=1060, y=100)  # Position at the right corner (adjust as needed based on window size)

# Header label
header_label = tk.Label(root, text="Email Spam Detection", font=("Helvetica", 24, "bold"), fg="#ffffff", bg="#34495e", pady=10)
header_label.pack(pady=10, fill="x")

# Text area label
text_area_label = tk.Label(root, text="Enter the email/message:", font=("Helvetica", 16), fg="white", bg="#2c3e50", pady=10)
text_area_label.pack(pady=10)

# Text area for email/message input
text_area = tk.Text(root, height=8, width=60, font=("Helvetica", 14), fg="#34495e", bg="#ecf0f1", bd=2, relief="sunken")
text_area.pack(pady=10)

# Classify Button with a raised relief
predict_button = tk.Button(root, text="Classify", font=("Helvetica", 14, "bold"), fg="white", bg="#e74c3c", 
                           activebackground="#c0392b", padx=20, pady=10, relief="raised", command=classify_message)
predict_button.pack(pady=20)

# Result label
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50", pady=20)
result_label.pack(pady=20)

# Footer label
footer_label = tk.Label(root, text="Created using Logistic Regression", font=("Helvetica", 10), fg="#ffffff", bg="#34495e", pady=5)
footer_label.pack(side="bottom", pady=10, fill="x")

# Start the GUI loop
root.mainloop()

