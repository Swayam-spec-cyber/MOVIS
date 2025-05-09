import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os

from insightface.app import FaceAnalysis
from core.image_caption import generate_caption
from core.face_analyzer import analyze_face
from core.face_matcher import match_face, reload_personal_embeddings
from core.wiki_fetcher import fetch_wikipedia_summary
from core.personal_trainer import train_personal_faces, add_new_face

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

app = tk.Tk()
app.title("MOVIS - Modular Visual Intelligence System")
app.geometry("850x800")
app.configure(bg="#f3f4f6")

tk.Label(app, text="MOVIS: Modular Visual Intelligence System for Human Profiling & Contextual Analysis",
         font=("Helvetica", 14, "bold"), bg="#f3f4f6", fg="#1a202c").pack(pady=10)

#Uploading Image
img_frame_outer = tk.Frame(app, bg="#dbeafe", bd=2, relief="ridge", width=420, height=420)
img_frame_outer.pack(pady=10)
img_frame_outer.pack_propagate(False)

img_frame_inner = tk.Frame(img_frame_outer, bg="#e0ecff", bd=2, relief="solid")
img_frame_inner.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

img_label = tk.Label(img_frame_inner, text="📤 Upload Image", bg="#e0ecff", fg="#1f4e79", font=("Helvetica", 14, "bold"),
                     padx=20, pady=20)
img_label.pack()

#Output Text Box
output_box = tk.Text(app, wrap=tk.WORD, font=("Courier", 10), bg="#ffffff", fg="#333333", height=20, width=95)
output_box.pack(pady=20)

#Upload Image
def click_image_box(event=None):
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
        initialdir="assets"
    )
    if path:
        process_image(path)

img_label.bind("<Button-1>", click_image_box)
img_frame_inner.bind("<Button-1>", click_image_box)

def ask_name_popup():
    popup = tk.Toplevel()
    popup.geometry("300x130")
    popup.title("Unknown Person")
    tk.Label(popup, text="Enter this person's name:").pack(pady=5)
    entry = tk.Entry(popup)
    entry.pack(pady=5)
    result = {"name": None}
    def submit():
        result["name"] = entry.get().strip()
        popup.destroy()
    tk.Button(popup, text="Save", command=submit).pack(pady=5)
    popup.wait_window()
    return result["name"]

#Image Processing
def process_image(path):
    output_box.delete("1.0", tk.END)

    img = Image.open(path).resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk, text="")
    img_label.image = img_tk

    output_box.insert(tk.END, f"Caption:\n{generate_caption(path)}\n\n")

    #Demographics
    face_info = analyze_face(path)
    age = face_info.get("age", "N/A")
    gender = face_info.get("gender", "N/A")
    confidence = face_info.get("confidence", 0)
    output_box.insert(tk.END, f"👤 Demographics:\nAge: {age}  Gender: {gender} (Confidence: {confidence:.2f})\n\n")

    #Face Recognition
    faces = face_app.get(np.array(img.convert('RGB')))
    if not faces:
        output_box.insert(tk.END, "❌ No face detected.\n")
        return

    embedding = faces[0].normed_embedding
    name, score = match_face(embedding)

    output_box.insert(tk.END, f"Match:\n{name} (Score: {score:.2f})\n\n")

    #Wikipedia
    if name != "Unknown":
        wiki = fetch_wikipedia_summary(name.title())
        if wiki["summary"]:
            output_box.insert(tk.END, f"🌐 Wikipedia:\n{wiki['summary']}\n🔗 {wiki['url']}\n")
        else:
            output_box.insert(tk.END, f"🌐 Wikipedia:\n❌ Wikipedia: Page for '{name}' not found.\n")
    else:
        entered_name = ask_name_popup()
        if entered_name:
            msg = add_new_face(path, entered_name)
            output_box.insert(tk.END, f"\n✅ {msg}\n")

            reload_personal_embeddings()
            new_name, new_score = match_face(embedding)
            output_box.insert(tk.END, f"\n Re-Match:\n{new_name} (Score: {new_score:.2f})\n")

            wiki = fetch_wikipedia_summary(entered_name.title())
            if wiki["summary"]:
                output_box.insert(tk.END, f"🌐 Wikipedia:\n{wiki['summary']}\n {wiki['url']}\n")
            else:
                output_box.insert(tk.END, f"🌐 Wikipedia:\n❌ Wikipedia: Page for '{entered_name}' not found.\n")

#Upload Button
tk.Button(app, text="📤 Upload & Analyze", command=lambda: click_image_box(),
          font=("Helvetica", 11, "bold"), bg="#1a73e8", fg="white", padx=20, pady=10).pack(pady=5)


app.mainloop()
