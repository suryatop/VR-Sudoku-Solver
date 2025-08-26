# realtime_sudoku_solver_threaded.py
# ------------------------------------------------------------
# Real-Time Sudoku Solver (OpenCV + Keras + Threaded Webcam)
# ------------------------------------------------------------

import cv2
import numpy as np
import threading
from queue import Queue
from typing import Tuple, List, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ---------------- Model ----------------

def build_and_load_model(weights_path: str = "digitRecognition.h5",
                         input_shape: Tuple[int,int,int]=(28,28,1),
                         num_classes: int=9) -> Sequential:
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.load_weights(weights_path)
    print("✅ Loaded weights:", weights_path)
    return model

# ---------------- Geometry Helpers ----------------

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left
    rect[2] = pts[np.argmax(s)]        # bottom-right
    rect[1] = pts[np.argmin(diff)]     # top-right
    rect[3] = pts[np.argmax(diff)]     # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, size: int=450) -> Tuple[np.ndarray, np.ndarray]:
    rect = order_points(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (size,size))
    return warped, M

def invert_perspective(src: np.ndarray, M: np.ndarray, dst_shape: Tuple[int,int]) -> np.ndarray:
    Minv = np.linalg.inv(M)
    return cv2.warpPerspective(src, Minv, (dst_shape[1], dst_shape[0]))

# ---------------- Grid Detection ----------------

def preprocess_for_contours(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    inv = cv2.bitwise_not(thresh)
    return inv

def find_largest_quad(cnts: List[np.ndarray]) -> Optional[np.ndarray]:
    biggest = None
    max_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and area>max_area:
            biggest = approx
            max_area = area
    return biggest

def extract_grid(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pre = preprocess_for_contours(image)
    contours, _ = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = find_largest_quad(contours)
    if quad is None:
        raise RuntimeError("Could not find Sudoku grid.")
    warped, M = four_point_transform(image, quad.reshape(4,2))
    return warped, M

# ---------------- Cells & Digit Extraction ----------------

def binarize_for_cells(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    inv = cv2.bitwise_not(th)
    return inv

def split_into_cells(binary_grid: np.ndarray) -> List[np.ndarray]:
    size = binary_grid.shape[0]
    step = size // 9
    cells = []
    for r in range(9):
        for c in range(9):
            y1, y2 = r*step, (r+1)*step
            x1, x2 = c*step, (c+1)*step
            cells.append(binary_grid[y1:y2, x1:x2])
    return cells

def is_empty_cell(cell: np.ndarray, ink_thresh: float=0.02) -> bool:
    h,w = cell.shape
    white = cv2.countNonZero(cell)
    frac = white / float(h*w)
    return frac < ink_thresh

def center_crop_and_resize(cell: np.ndarray, out_size: int=28) -> np.ndarray:
    h,w = cell.shape
    margin = int(0.1*min(h,w))
    roi = cell[margin:h-margin, margin:w-margin]
    if cv2.countNonZero(roi)<10:
        return np.zeros((out_size,out_size), dtype=np.uint8)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((out_size,out_size), dtype=np.uint8)
    c = max(contours, key=cv2.contourArea)
    x,y,ww,hh = cv2.boundingRect(c)
    digit = roi[y:y+hh, x:x+ww]
    h2,w2 = digit.shape
    scale = 20.0 / max(h2,w2)
    digit_resized = cv2.resize(digit, (int(w2*scale), int(h2*scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size,out_size), dtype=np.uint8)
    y_off = (out_size - digit_resized.shape[0]) // 2
    x_off = (out_size - digit_resized.shape[1]) // 2
    canvas[y_off:y_off+digit_resized.shape[0], x_off:x_off+digit_resized.shape[1]] = digit_resized
    return canvas

def cells_to_model_inputs(cells: List[np.ndarray]) -> np.ndarray:
    imgs = []
    for cell in cells:
        if is_empty_cell(cell):
            imgs.append(np.zeros((28,28), dtype=np.uint8))
        else:
            imgs.append(center_crop_and_resize(cell, 28))
    X = np.array(imgs, dtype=np.float32)/255.0
    X = X.reshape((-1,28,28,1))
    return X

def pred_to_digit(pred: np.ndarray) -> int:
    return int(np.argmax(pred)) + 1

def recognize_digits(model: Sequential, cells_binary: List[np.ndarray]) -> np.ndarray:
    X = cells_to_model_inputs(cells_binary)
    preds = model.predict(X, verbose=0)
    digits = []
    for i,p in enumerate(preds):
        if np.count_nonzero(cells_binary[i])<10:
            digits.append(0)
        else:
            digits.append(pred_to_digit(p))
    return np.array(digits, dtype=int).reshape((9,9))

# ---------------- Sudoku Solver ----------------

def find_empty(board: np.ndarray) -> Optional[Tuple[int,int]]:
    for r in range(9):
        for c in range(9):
            if board[r,c]==0:
                return r,c
    return None

def valid(board: np.ndarray, num: int, pos: Tuple[int,int]) -> bool:
    r,c = pos
    if any(board[r,i]==num for i in range(9)):
        return False
    if any(board[i,c]==num for i in range(9)):
        return False
    br,bc = 3*(r//3),3*(c//3)
    if any(board[i,j]==num for i in range(br,br+3) for j in range(bc,bc+3)):
        return False
    return True

def solve_sudoku(board: np.ndarray) -> bool:
    empty = find_empty(board)
    if not empty: return True
    r,c = empty
    for num in range(1,10):
        if valid(board,num,(r,c)):
            board[r,c] = num
            if solve_sudoku(board):
                return True
            board[r,c] = 0
    return False

# ---------------- Overlay Solution ----------------

def draw_digits_on_warp(warp: np.ndarray, original_grid: np.ndarray, solved_grid: np.ndarray) -> np.ndarray:
    out = warp.copy()
    size = warp.shape[0]
    step = size // 9
    for r in range(9):
        for c in range(9):
            if original_grid[r,c]==0 and solved_grid[r,c]!=0:
                text = str(int(solved_grid[r,c]))
                x = c*step + step//3
                y = r*step + int(step*0.75)
                cv2.putText(out,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2,cv2.LINE_AA)
    return out

def blend_overlay(original: np.ndarray, overlay: np.ndarray, M: np.ndarray, alpha: float=0.8) -> np.ndarray:
    warped_back = invert_perspective(overlay, M, original.shape[:2])
    gray = cv2.cvtColor(warped_back, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    blended = original.copy()
    blended[mask>0] = cv2.addWeighted(original[mask>0], 1-alpha, warped_back[mask>0], alpha,0)
    return blended

# ---------------- Pipeline ----------------

def recognize_sudoku_and_solve_image(bgr_img: np.ndarray, model: Sequential) -> np.ndarray:
    try:
        warp_color, M = extract_grid(bgr_img)
    except RuntimeError:
        return bgr_img
    warp_bin = binarize_for_cells(warp_color)
    cells = split_into_cells(warp_bin)
    detected_grid = recognize_digits(model, cells)
    board = detected_grid.copy()
    if not solve_sudoku(board):
        return bgr_img
    warp_solution = draw_digits_on_warp(warp_color, detected_grid, board)
    return blend_overlay(bgr_img, warp_solution, M, alpha=0.9)

# ---------------- Threaded Webcam ----------------

class VideoStreamWidget:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.queue = Queue(maxsize=1)
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.queue.full():
                try: self.queue.get_nowait()
                except: pass
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

# ---------------- Main ----------------

def main():
    model = build_and_load_model("digitRecognition.h5")
    vs = VideoStreamWidget(0)

    while True:
        frame = vs.read()
        solved_frame = recognize_sudoku_and_solve_image(frame, model)
        cv2.imshow("Real-Time Sudoku Solver", solved_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
