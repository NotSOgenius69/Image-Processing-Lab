import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import math
import random
from image_filters import apply_LoG_sharpen
class ClueHuntingGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Detective Clue Hunter Game")
        self.root.geometry("1200x800")
        
        self.current_level = 1
        self.original_image = None
        self.distorted_image = None
        self.current_image = None
        self.score = 0
        self.max_levels = 5
        self.game_started = False
        
        self.image_history = []
        self.max_history = 10 
        
        self.level_clues = {
            1: {
                "question": "Find out the ID no",
                "answer": "735918642A",  
                "hint": "Hmmm...What can be the opposite of bluring? Maybe its sharpening."
            },
            2: {
                "question": "What is the meeting date mentioned?",
                "answer": "25th",
                "hint": "Oh my god!This much noise.Suppress the noise maybe?"
            },
            3: {
                "question": "At which day is the meeting scheduled?",
                "answer": "Tuesday",
                "hint": ""
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

        self.clue_images = []
        self.preload_clue_images()
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        
        question_frame = ttk.LabelFrame(main_frame, text="Your Mission", padding=10)
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.question_label = ttk.Label(question_frame, text="Click Start Game to begin your detective mission!", 
                                       font=("Arial", 12), wraplength=800)
        self.question_label.pack()
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        tools_frame = ttk.LabelFrame(content_frame, text="Image Processing Tools", padding=10)
        tools_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        noise_frame = ttk.LabelFrame(tools_frame, text="Noise Reduction")
        noise_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(noise_frame, text="Gaussian Filter", 
                  command=self.apply_gaussian_filter).pack(fill=tk.X, pady=2)
        ttk.Button(noise_frame, text="Median Filter", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        ttk.Button(noise_frame, text="Bilateral Filter", 
                  command=self.apply_bilateral_filter).pack(fill=tk.X, pady=2)
        
        sharp_frame = ttk.LabelFrame(tools_frame, text="Sharpening")
        sharp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(sharp_frame, text="Unsharp Masking", 
                  command=self.apply_unsharp_mask).pack(fill=tk.X, pady=2)
        ttk.Button(sharp_frame, text="Laplacian Sharpen", 
                  command=self.apply_laplacian_sharpen).pack(fill=tk.X, pady=2)
        enhance_frame = ttk.LabelFrame(tools_frame, text="Enhancement")
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
    
        ttk.Button(enhance_frame, text="Gamma Correction", 
                  command=self.apply_gamma_correction).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Contrast Stretch", 
                  command=self.apply_contrast_stretch).pack(fill=tk.X, pady=2)
        
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
        
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def preload_clue_images(self):
        image_paths = [
            "e:/Image Processing Lab/Project/Clue-1.png",
            "e:/Image Processing Lab/Project/Clue-2e.png", 
            "e:/Image Processing Lab/Project/Clue-3.png",
            "e:/Image Processing Lab/Project/Clue-4-prev.png",
            "e:/Image Processing Lab/Project/Clue-5.jpg",
        ]
        
        for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    h,w=img.shape[:2]
                    if h>500 or w>500:
                        scale=min(500/h, 500/w)
                        new_h,new_w=int(h*scale),int(w*scale)
                        img=cv2.resize(img,(new_w,new_h))
                    self.clue_images.append(img)
                
        
    def start_game(self):
        if not self.game_started:
            self.game_started=True
            self.current_level=1
            self.score=0
            self.start_btn.config(text="Restart Game")
            self.level_label.config(text="1")
            self.score_label.config(text="0")
            self.load_level()
        else:
            self.game_started = False
            self.start_btn.config(text="Start Game")
            self.canvas.delete("all")
            self.question_label.config(text="Click Start Game to begin your detective mission!")
    
    def load_level(self):
        if self.current_level<=len(self.clue_images):
            self.original_image=self.clue_images[min(self.current_level-1, len(self.clue_images)-1)].copy()
            self.apply_hiding_distortion()
            clue_info=self.level_clues[self.current_level]
            self.question_label.config(text=f"Level {self.current_level}: {clue_info['question']}")
    
    def apply_hiding_distortion(self):
        if self.original_image is None:
            return
            
        img=self.original_image.copy()
        
        if self.current_level==1:
            img=cv2.GaussianBlur(img,(11, 11),1.8)
            img=cv2.convertScaleAbs(img,alpha=0.4,beta=10)
            
        elif self.current_level==2:
            img=cv2.convertScaleAbs(img, alpha=0.3, beta=-20)
            noise=np.random.normal(0,20,img.shape).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
            
        elif self.current_level==3:
            img=cv2.GaussianBlur(img, (5, 5), 5)
            noise=np.random.normal(0,25,img.shape).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
           
            
        elif self.current_level==4:
            img=cv2.GaussianBlur(img,(5, 5),2.0)
            noise=np.random.normal(0,15,img.shape).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
            img=cv2.convertScaleAbs(img,alpha=0.3,beta=-30)
            
        else:
            img=cv2.GaussianBlur(img,(9, 9),2.5)
            noise=np.random.normal(0,45,img.shape).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
            img=cv2.convertScaleAbs(img,alpha=0.35,beta=-40)
        
        self.distorted_image=img
        self.current_image=img.copy()
        
        self.image_history=[]
        self.save_to_history()
        
        self.display_image()
        
    def display_image(self):
        if self.current_image is None:
            return
            
        img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width>1 and canvas_height>1:
            img_pil.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        
        self.photo=ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo, anchor=tk.CENTER)
    
    def save_to_history(self):
        if self.current_image is not None:
            self.image_history.append(self.current_image.copy())
            
            if len(self.image_history)>self.max_history:
                self.image_history.pop(0)
    
    def undo_last_action(self):
        if len(self.image_history)<=1:
            messagebox.showinfo("Undo", "No more actions to undo!")
            return
        
        self.image_history.pop()
        self.current_image = self.image_history[-1].copy()
        self.display_image()
    
    def reset_image(self):
        if self.distorted_image is not None:
            self.current_image = self.distorted_image.copy()
            self.image_history = [self.current_image.copy()]
            self.display_image()
    
    def apply_gaussian_filter(self):
        if self.current_image is not None:
            self.save_to_history()
            self.current_image=cv2.GaussianBlur(self.current_image, (5, 5), 1.5)
            self.display_image()
    
    def apply_median_filter(self):
        if self.current_image is not None:
            self.save_to_history() 
            self.current_image = cv2.medianBlur(self.current_image, 5)
            self.display_image()
    
    def apply_bilateral_filter(self):
        if self.current_image is not None:
            self.save_to_history()
            self.current_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)
            self.display_image()
    
    def apply_unsharp_mask(self):
        if self.current_image is not None:
            self.save_to_history()
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 2.0)
            self.current_image = cv2.addWeighted(self.current_image, 1.5, gaussian, -0.5, 0)
            self.display_image()
    
    def apply_laplacian_sharpen(self):
        if self.current_image is not None:
            self.save_to_history()
            blurred=cv2.GaussianBlur(self.current_image, (3, 3), 1.0)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.current_image = cv2.filter2D(blurred, -1, kernel)
            self.display_image()
    
    
    def apply_gamma_correction(self):
        if self.current_image is not None:
            self.save_to_history()
            gamma=2.2
            inv_gamma=1.0/gamma
            table = np.array([((i/255.0)**inv_gamma)*255 
                            for i in np.arange(0, 256)]).astype("uint8")
            self.current_image = cv2.LUT(self.current_image, table)
            self.display_image()
    
    def apply_contrast_stretch(self):
        if self.current_image is not None:
            self.save_to_history() 
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
            
        answer=simpledialog.askstring("Submit Answer", 
                                       f"Level {self.current_level}: {self.level_clues[self.current_level]['question']}")
        
        if answer is None: 
            return
            
        correct_answer=self.level_clues[self.current_level]["answer"].lower()
        user_answer=answer.lower().strip()
        
        import re
        user_answer=re.sub(r'[^\w\s]', '', user_answer)
        user_answer=re.sub(r'\s+', ' ', user_answer)
        
        if correct_answer in user_answer or user_answer in correct_answer:
            points = 100
            self.score += points
            self.score_label.config(text=str(self.score))
            
            messagebox.showinfo("Correct!", 
                              f"Excellent detective work! You found: '{answer}'\n+{points} points!\n\nMoving to next level...")
        else:
            messagebox.showinfo("Incorrect Answer", 
                              f"That's not correct. The answer was: '{self.level_clues[self.current_level]['answer']}'\n\nNo points awarded, but moving to next level...")
        
        if self.current_level<self.max_levels:
            self.current_level+= 1
            self.level_label.config(text=str(self.current_level))
            self.load_level()
        else:
            messagebox.showinfo("Game Complete!", 
                              f"Congratulations, Detective! You completed all levels!\nFinal Score: {self.score}/{self.max_levels * 100}")
            self.game_started = False
            self.start_btn.config(text="Start Game")
            self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    game = ClueHuntingGame(root)
    root.mainloop()