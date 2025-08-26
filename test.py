import cv2

def init_camera():
    """Initialize camera with AVFoundation (macOS friendly)."""
    # Try default camera with AVFoundation backend
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        print("❌ Failed to open camera. Try changing index (1, 2, 3).")
        return None
    return cap


def main():
    cap = init_camera()
    if cap is None:
        return

    print("🎥 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame. Try another camera index.")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
