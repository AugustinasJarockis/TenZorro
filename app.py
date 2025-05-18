import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
from pyvidplayer2 import Video 
import os 
import threading
from tenzorrovideoanalyser import modify_the_video

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 TenZorro Art - Video Analyzer")
        self.root.geometry("500x300")
        self.root.configure(bg="#2c2f33")

        self.video_path = None
        self.analyzed_video_path = None
        self.loading_window = None

        self.build_ui()

    def build_ui(self):
        title_label = tk.Label(
            self.root, 
            text="Video Analyzer Tool", 
            font=("Helvetica", 18, "bold"), 
            bg="#2c2f33", 
            fg="#ffffff"
        )
        title_label.pack(pady=20)

        self.upload_button = tk.Button(
            self.root, 
            text="📂 Upload Video", 
            font=("Helvetica", 12),
            bg="#7289da", 
            fg="#ffffff", 
            activebackground="#5b6eae", 
            command=self.upload_video
        )
        self.upload_button.pack(pady=10)

        self.status_label = tk.Label(
            self.root, 
            text="No video selected", 
            font=("Helvetica", 10), 
            bg="#2c2f33", 
            fg="#aaaaaa"
        )
        self.status_label.pack(pady=5)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", ".*")
            ]
        )
        if not self.video_path:
            return

        filename = os.path.basename(self.video_path)
        self.status_label.config(text=f"Selected: {filename}", fg="#ffffff")

        dir_path = os.path.dirname(self.video_path)
        new_filename = f"ARTANALYSIS_{filename}"
        self.analyzed_video_path = os.path.join(dir_path, new_filename)

        thread = threading.Thread(target=self.process_video)
        thread.start()
        self.show_loading_screen()

    def process_video(self):
        try:
            modify_the_video(self.video_path, self.analyzed_video_path)
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: Video(self.analyzed_video_path).preview())
        except Exception as e:
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not process video: {str(e)}"))

    def show_loading_screen(self):
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Processing...")
        self.loading_window.geometry("300x120")
        self.loading_window.configure(bg="#2c2f33")
        Label(
            self.loading_window, 
            text="⏳ Analyzing video, please wait...", 
            bg="#2c2f33", 
            fg="#ffffff", 
            font=("Helvetica", 11)
        ).pack(pady=30)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()

    def hide_loading_screen(self):
        if self.loading_window:
            self.loading_window.destroy()
            self.loading_window = None

def main():
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
