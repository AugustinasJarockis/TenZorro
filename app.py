import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label
from pyvidplayer2 import Video 
import os 
import threading
from tenzorrovideoanalyser import modify_the_video

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TenZorro Art - Video Analyzer")
        self.root.geometry("400x200")

        self.upload_button = tk.Button(
            root, 
            text="Upload Video", 
            command=self.upload_video
        )
        self.upload_button.pack(pady=10)

        self.video_path = None
        self.analyzed_video_path = None
        self.loading_window = None

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        if not self.video_path:
            return

        dir_path = os.path.dirname(self.video_path)
        filename = os.path.basename(self.video_path)
        new_filename = f"ART_ANALYSIS_{filename}"
        self.analyzed_video_path = os.path.join(dir_path, new_filename)

        # Paleidžiam analizę naujame threade
        thread = threading.Thread(target=self.process_video)
        thread.start()

        # Parodome „Loading“ langą
        self.show_loading_screen()

    def process_video(self):
        try:
            modify_the_video(self.video_path, self.analyzed_video_path)
            
            # Uždaryti loading langą ir rodyti rezultatą (UI veiksmai turi būti atliekami pagrindiniame threade)
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: Video(self.analyzed_video_path).preview())
        except Exception as e:
            self.root.after(0, self.hide_loading_screen)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not process video: {str(e)}"))

    def show_loading_screen(self):
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Processing...")
        self.loading_window.geometry("250x100")
        Label(self.loading_window, text="Analyzing video, please wait...").pack(pady=20)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()  # Modal

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
