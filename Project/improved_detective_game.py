import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import math
import random

class ClueHuntingGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Detective Clue Hunter Game")
        self.root.geometry("1200x800")
        
        # Game state
        self.current_level = 1
        self.original_image = None
        self.distorted_image = None
        self.current_image = None
        self.score = 0
        self.max_levels = 5
        self.game_started = False
        
        # History for undo functionality
        self.image_history = []
        self.max_history = 10  # Keep last 10 operations
        
        # Level-specific clues and answers
        self.level_clues = {
            1: {
                "question": "Who is the document addressed TO?",
                "answer": "olivia johnson",  
                "hint": "Too much dark right? Maybe try brightening the image."
            },
            2: {
                "question": "What is the meeting date mentioned?",
                "answer": "March 15 2024",
                "hint": "Oh my god!This much noise.Suppress the noise maybe?"
            },
            3: {
                "question": "At which day is the meeting scheduled?",
                "answer": "Tuesday",
                "hint": "Hmmm...What can be the opposite of bluring? Maybe its sharpening."
            },
            4: {
                "question": "Where is the meeting location?",
                "answer": "conference room",
                "hint": "Look for location details"
            },
            5: {
                "question": "Who is the sender of this document?",
                "answer": "michael smith",
                "hint": "Look at the signature or FROM field"
            }
        }

        # Preload clue images
        self.clue_images = []
        self.preload_clue_images()
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel - Game info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Level:").pack(side=tk.LEFT)
        self.level_label = ttk.Label(info_frame, text="1", font=("Arial", 12, "bold"))
        self.level_label.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(info_frame, text="Score:").pack(side=tk.LEFT)
        self.score_label = ttk.Label(info_frame, text="0", font=("Arial", 12, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=(5, 20))
        
        self.start_btn = ttk.Button(info_frame, text="Start Game", command=self.start_game)
        self.start_btn.pack(side=tk.RIGHT)
        
        # Question frame
        question_frame = ttk.LabelFrame(main_frame, text="Your Mission", padding=10)
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.question_label = ttk.Label(question_frame, text="Process the image and find the clue!", 
                                       font=("Arial", 12), wraplength=800)
        self.question_label.pack()
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Tools
        tools_frame = ttk.LabelFrame(content_frame, text="Image Processing Tools", padding=10)
        tools_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Noise Reduction Tools
        noise_frame = ttk.LabelFrame(tools_frame, text="Noise Reduction")
        noise_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(noise_frame, text="Gaussian Filter", 
                  command=self.apply_gaussian_filter).pack(fill=tk.X, pady=2)
        ttk.Button(noise_frame, text="Median Filter", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        ttk.Button(noise_frame, text="Bilateral Filter", 
                  command=self.apply_bilateral_filter).pack(fill=tk.X, pady=2)
        
        # Sharpening Tools
        sharp_frame = ttk.LabelFrame(tools_frame, text="Sharpening")
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(sharp_frame, text="Unsharp Masking", 
                  command=self.apply_unsharp_mask).pack(fill=tk.X, pady=2)
        ttk.Button(sharp_frame, text="Laplacian Sharpen", 
                  command=self.apply_laplacian_sharpen).pack(fill=tk.X, pady=2)
        
        # Enhancement Tools
        enhance_frame = ttk.LabelFrame(tools_frame, text="Enhancement")
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
    
        ttk.Button(enhance_frame, text="Gamma Correction", 
                  command=self.apply_gamma_correction).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Contrast Stretch", 
                  command=self.apply_contrast_stretch).pack(fill=tk.X, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(tools_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Undo Last Action", 
                  command=self.undo_last_action, style='Warning.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Reset Image", 
                  command=self.reset_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Get Hint", 
                  command=self.show_hint).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Submit Answer", 
                  command=self.submit_answer, style='Accent.TButton').pack(fill=tk.X, pady=5)
        
        # Right panel - Image display
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image canvas
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status display
        self.status_label = ttk.Label(image_frame, text="Click Start Game to begin your detective mission!", 
                                     font=("Arial", 12), foreground="blue")
        self.status_label.pack(pady=10)
        
    def preload_clue_images(self):
        # Load your document images
        image_paths = [
            "e:/Image Processing Lab/Project/Clue-1.png",
            "e:/Image Processing Lab/Project/Clue-2.png", 
            "e:/Image Processing Lab/Project/Clue-3.png",
            "e:/Image Processing Lab/Project/Clue-4.png",
            "e:/Image Processing Lab/Project/Clue-5.jpg",
        ]
        
        # For demonstration, create a sample document if files don't exist
        for i, path in enumerate(image_paths):
            try:
                img = cv2.imread(path)
                if img is not None:
                    # Resize if too large
                    h, w = img.shape[:2]
                    if h > 500 or w > 500:
                        scale = min(500/h, 500/w)
                        new_h, new_w = int(h*scale), int(w*scale)
                        img = cv2.resize(img, (new_w, new_h))
                    self.clue_images.append(img)
                else:
                    # Create a dummy document if file doesn't exist
                    self.clue_images.append(self.create_sample_document(i+1))
            except:
                # Create a dummy document if file can't be loaded
                self.clue_images.append(self.create_sample_document(i+1))
    
    # def create_sample_document(self, level_num):
    #     """Create a sample document if actual files aren't available"""
    #     img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
    #     # Add text based on your sample document
    #     font = cv2.FONT_HERSHEY_SIMPLEX
        
    #     cv2.putText(img, "DOCUMENT", (200, 50), font, 1, (0,0,0), 2)
    #     cv2.putText(img, "", (50, 100), font, 0.6, (0,0,0), 1)
    #     cv2.putText(img, "TO: Olivia Johnson", (50, 130), font, 0.6, (0,0,0), 1)
    #     cv2.putText(img, "FROM: Michael Smith", (50, 160), font, 0.6, (0,0,0), 1)
    #     cv2.putText(img, "SUBJECT: Meeting Reminder", (50, 190), font, 0.6, (0,0,0), 1)
    #     cv2.putText(img, "", (50, 220), font, 0.6, (0,0,0), 1)
    #     cv2.putText(img, "This is a reminder for our meeting", (50, 250), font, 0.5, (0,0,0), 1)
    #     cv2.putText(img, "scheduled on Tuesday, April 9, 2024,", (50, 280), font, 0.5, (0,0,0), 1)
    #     cv2.putText(img, "at 10:00 AM in the conference room.", (50, 310), font, 0.5, (0,0,0), 1)
    #     cv2.putText(img, "", (50, 340), font, 0.5, (0,0,0), 1)
    #     cv2.putText(img, "Regards,", (50, 370), font, 0.5, (0,0,0), 1)
    #     cv2.putText(img, "Michael Smith", (50, 400), font, 0.5, (0,0,0), 1)
        
    #     return img
        
    def start_game(self):
        if not self.game_started:
            self.game_started = True
            self.current_level = 1
            self.score = 0
            self.start_btn.config(text="Restart Game")
            self.level_label.config(text="1")
            self.score_label.config(text="0")
            self.load_level()
        else:
            # Restart game
            self.game_started = False
            self.start_btn.config(text="Start Game")
            self.canvas.delete("all")
            self.question_label.config(text="Process the image and find the clue!")
            self.status_label.config(text="Click Start Game to begin your detective mission!")
    
    def load_level(self):
        if self.current_level <= len(self.clue_images):
            self.original_image = self.clue_images[min(self.current_level-1, len(self.clue_images)-1)].copy()
            self.apply_hiding_distortion()
            
            # Update question
            clue_info = self.level_clues[self.current_level]
            self.question_label.config(text=f"Level {self.current_level}: {clue_info['question']}")
            self.status_label.config(text="Process the image to reveal the hidden information!")
    
    def apply_hiding_distortion(self):
        """Apply distortions that hide text but allow recovery"""
        if self.original_image is None:
            return
            
        img = self.original_image.copy()
        
        if self.current_level == 1:
             # Heavy blur that makes text unreadable
            img = cv2.GaussianBlur(img, (11, 11), 1.5)
            # Add slight contrast issue
            img = cv2.convertScaleAbs(img, alpha=0.4, beta=10)
            
        elif self.current_level == 2:
            # Very dark image - text nearly invisible
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=-20)
            # Add some noise
            noise = np.random.normal(0, 20, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif self.current_level == 3:
            # Heavy noise that obscures text but preserves structure
            img = cv2.GaussianBlur(img, (5, 5), 2.0)
            noise = np.random.normal(0, 25, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
           
            
        elif self.current_level == 4:
            # Combination: noise + darkness + slight blur
            img = cv2.GaussianBlur(img, (5, 5), 2.0)
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=-30)
            
        else:  # Level 5
            # Most challenging: multiple distortions
            # First blur
            img = cv2.GaussianBlur(img, (9, 9), 2.5)
            # Heavy noise
            noise = np.random.normal(0, 45, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Make very dark
            img = cv2.convertScaleAbs(img, alpha=0.35, beta=-40)
        
        self.distorted_image = img
        self.current_image = img.copy()
        
        # Clear history when loading new level
        self.image_history = []
        # Add initial distorted image to history
        self.save_to_history()
        
        self.display_image()
        
    def display_image(self):
        if self.current_image is None:
            return
            
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for display
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo, anchor=tk.CENTER)
    
    def save_to_history(self):
        """Save current image to history for undo functionality"""
        if self.current_image is not None:
            # Add current image to history
            self.image_history.append(self.current_image.copy())
            
            # Keep only last max_history images
            if len(self.image_history) > self.max_history:
                self.image_history.pop(0)
    
    def undo_last_action(self):
        """Undo the last image processing action"""
        if len(self.image_history) <= 1:
            messagebox.showinfo("Undo", "No more actions to undo!")
            return
        
        # Remove current state (last item)
        self.image_history.pop()
        
        # Restore previous state
        if self.image_history:
            self.current_image = self.image_history[-1].copy()
            self.display_image()
            self.status_label.config(text="Last action undone!")
        else:
            messagebox.showinfo("Undo", "No more actions to undo!")
    
    def reset_image(self):
        if self.distorted_image is not None:
            self.current_image = self.distorted_image.copy()
            # Clear history and add reset state
            self.image_history = [self.current_image.copy()]
            self.display_image()
            self.status_label.config(text="Image reset to original distorted state!")
    
    # Image Processing Methods (Updated with history saving)
    def apply_gaussian_filter(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 1.5)
            self.display_image()
    
    def apply_median_filter(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            self.current_image = cv2.medianBlur(self.current_image, 5)
            self.display_image()
    
    def apply_bilateral_filter(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            self.current_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)
            self.display_image()
    
    def apply_unsharp_mask(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 2.0)
            self.current_image = cv2.addWeighted(self.current_image, 1.5, gaussian, -0.5, 0)
            self.display_image()
    
    def apply_laplacian_sharpen(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_image()
    
    
    def apply_gamma_correction(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            gamma = 2.2  # Brightening (gamma > 1 brightens)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            self.current_image = cv2.LUT(self.current_image, table)
            self.display_image()
    
    def apply_contrast_stretch(self):
        if self.current_image is not None:
            self.save_to_history()  # Save before processing
            self.current_image = cv2.convertScaleAbs(self.current_image, alpha=1.5, beta=30)
            self.display_image()
    
    def show_hint(self):
        if self.game_started and self.current_level in self.level_clues:
            hint = self.level_clues[self.current_level]["hint"]
            messagebox.showinfo("Hint", hint)
    
    def submit_answer(self):
        if not self.game_started:
            messagebox.showwarning("Warning", "Please start the game first!")
            return
            
        # Get user answer
        answer = simpledialog.askstring("Submit Answer", 
                                       f"Level {self.current_level}: {self.level_clues[self.current_level]['question']}")
        
        if answer is None:  # User cancelled
            return
            
        # Check answer (flexible matching)
        correct_answer = self.level_clues[self.current_level]["answer"].lower()
        user_answer = answer.lower().strip()
        
        # Remove common punctuation and extra spaces
        import re
        user_answer = re.sub(r'[^\w\s]', '', user_answer)
        user_answer = re.sub(r'\s+', ' ', user_answer)
        
        if correct_answer in user_answer or user_answer in correct_answer:
            # Correct answer!
            points = 100
            self.score += points
            self.score_label.config(text=str(self.score))
            
            messagebox.showinfo("Correct!", 
                              f"Excellent detective work! You found: '{answer}'\n+{points} points!\n\nMoving to next level...")
        else:
            # Wrong answer - but still allow progression
            messagebox.showinfo("Incorrect Answer", 
                              f"That's not correct. The answer was: '{self.level_clues[self.current_level]['answer']}'\n\nNo points awarded, but moving to next level...")
        
        # Move to next level regardless of answer correctness
        if self.current_level < self.max_levels:
            self.current_level += 1
            self.level_label.config(text=str(self.current_level))
            self.load_level()
        else:
            # Game completed
            messagebox.showinfo("Game Complete!", 
                              f"Congratulations, Detective! You completed all levels!\nFinal Score: {self.score}/{self.max_levels * 100}")
            self.game_started = False
            self.start_btn.config(text="Start Game")
            self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    game = ClueHuntingGame(root)
    root.mainloop()