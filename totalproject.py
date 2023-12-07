import tkinter as tk
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk  # Import Image and ImageTk from the PIL library
import cv2
import face_recognition
import time
import os
import datetime
import pandas as pd

class ImageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Attendance System")

        # Set the background color
        self.root.configure(bg='#E0E0E0')

        # Create and configure the background image
        self.background_image = Image.open("bg1.jpg")  # Change the path accordingly
        self.background_image = self.background_image.resize((800, 600))
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        # Create a label to hold the background image
        background_label = tk.Label(root, image=self.background_photo)
        background_label.place(relwidth=1, relheight=1)  # Cover the entire window with the background image

        # Create a style for the title label
        title_style = ttk.Style()
        title_style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), foreground='#1E90FF', background='#E0E0E0')

        # Add a title label
        title_label = ttk.Label(root, text="Camera Attendance System", style='Title.TLabel')
        title_label.pack(pady=10)

        # Create and configure buttons
        self.capture_button = ttk.Button(root, text="Start Capture", command=self.start_capture, style='Capture.TButton')
        self.capture_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop Capture", command=self.stop_capture, style='Stop.TButton')
        self.stop_button.pack(pady=10)
        self.stop_button["state"] = "disabled"

    def start_capture(self):
        self.capture_button["state"] = "disabled"
        self.stop_button["state"] = "normal"

        self.capture_thread = Thread(target=self.capture_and_recognize_images)
        self.capture_thread.start()

    def stop_capture(self):
        self.capture_button["state"] = "normal"
        self.stop_button["state"] = "disabled"

        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join()

    def load_known_faces(self, known_faces_folder):
        known_faces = {}
        known_faces_folder = "./training"
        for filename in os.listdir(known_faces_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(known_faces_folder, filename)
                known_image = face_recognition.load_image_file(image_path)
                known_face_encoding = face_recognition.face_encodings(known_image)[0]
                known_faces[name] = known_face_encoding
        return known_faces

    def recognize_faces(self, frame, known_faces):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]

            names.append(name)

        return names

    def capture_and_recognize_images(self):
        capture_period = 10  # seconds
        num_images_to_capture = 5  # adjust as needed
        known_faces_folder = "known_faces"  # folder containing known faces images

        known_faces = self.load_known_faces(known_faces_folder)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        try:
            for i in range(num_images_to_capture):
                ret, frame = cap.read()

                if not ret:
                    print("Error: Couldn't read frame.")
                    break

                names = self.recognize_faces(frame, known_faces)

                image_filename = f"captured_image_{i + 1}.jpg"
                cv2.imwrite(image_filename, frame)
                print(f"Image {i + 1} captured and saved as {image_filename}")

                if names:
                    print("Recognized faces:", ", ".join(names))
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(current_time)
                else:
                    print("No faces recognized.")

                time.sleep(capture_period)

        except Exception as e:
            print(f"Image capture error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecognitionApp(root)

    # Create and configure button styles
    button_style = ttk.Style()
    button_style.configure('Capture.TButton', font=('Helvetica', 12), foreground='#FFFFFF', background='#4CAF50')
    button_style.configure('Stop.TButton', font=('Helvetica', 12), foreground='#FFFFFF', background='#FF4500')

    root.geometry("800x600")  # Set the initial size of the window
    root.mainloop()
