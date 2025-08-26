# VR-Sudoku-Solver
Developed a Python application to solve Sudoku puzzles and overlay the solutions onto real images in real time.
# 🎯 Real-Time Sudoku Solver (OpenCV + Keras)

This is a Python project that can **detect Sudoku puzzles in real-time from a camera feed, solve them, and overlay the solution back onto the puzzle**.

The main aim of this project was to **learn and apply concepts of Computer Vision, Deep Learning, and Algorithm Design**.
---

## 🛠 Sample Output
<img src="https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/output/output-gif.gif" width="400">

---

## 🧠 Algorithm Used
- Normally, Sudoku is solved using a **Backtracking approach**, which is brute-force.  
- In this project, I used a **Greedy Best-First Search approach**, which is an optimized version of backtracking.  
  - It selects the cell with the **least number of possibilities** to try next, reducing search time.  

---

## 🚀 How to Run
1. Install the required libraries:  
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run all cells of `Sudoku testing.ipynb`.  
3. Hold a Sudoku puzzle in front of your webcam → the solution will appear on the puzzle in real-time.  
4. Press **Q** on your keyboard to stop the program.  

---

## ✍️ Notes
- I trained a **custom CNN model** for digit recognition.  
- You can either use my pre-trained model or train your own.  

---

## 📚 References
**Algorithms**  
- [Peter Norvig – Sudoku Solver](https://norvig.com/sudoku.html)  
- [Tech With Tim – Sudoku Solver](https://www.youtube.com/watch?v=lK4N8E6uNr4)  

**Computer Vision Concepts**  
- [Sudoku Extraction with OpenCV](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)  
- [Solving Sudoku – Part 1](https://medium.com/@neshpatel/solving-sudoku-part-i-7c4bb3097aa7)  
- [Solving Sudoku – Part 2](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)  

---

## 👨‍🎓 Author
Hi, I’m **Suryatop Sasmal**, a Computer Science student passionate about **AI, Computer Vision, and Cloud**.  
🤝 Connect with me on [LinkedIn](https://github.com/suryatop)
