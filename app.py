import time
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "yolo11s.pt" # Use "yolo11n.pt" if your computer is slow
VIDEO_PATH = "sample_video.mp4"  # Sample video included, or replace with your own
TARGET_CLASSES = {"cell phone", "remote"} # Sometimes the model detects phone as remote, so we add remote to the list
CONF_THRESHOLD = 0.4
COOLDOWN_SEC = 3.0
LATE_DROP_SEC = 0.25
STREAK_REQUIRED = 15
MIN_BOX_AREA_RATIO = 0.01
# ---------------

def get_screen_size():
    """Return (width, height) of the primary screen."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080  

SCREEN_W, SCREEN_H = get_screen_size()

model = YOLO(MODEL_PATH)
id2name = model.names
wanted_ids = {i for i, n in id2name.items() if n in TARGET_CLASSES}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

last_trigger = 0
streak_hits = 0 

def play_video_ffpy(path: str):
    ff_opts = {
        'out_format': 'rgb24',
        'sync': 'audio',
        'genpts': 1,
        'framedrop': True,
        'analyzeduration': 0,
        'probesize': 32,
        'fflags': 'nobuffer',
    }
    player = MediaPlayer(path, ff_opts=ff_opts)

    start_wall = None
    win = "Video"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    # Create dummy frame to initialize window
    dummy = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv2.imshow(win, dummy)
    cv2.waitKey(1)

    # Configure window properties
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    except Exception:
        pass

    video_width = SCREEN_W - 250 
    video_height = SCREEN_H + 100
    x_pos = (SCREEN_W - video_width) // 2
    y_pos = (SCREEN_H - video_height) // 2
    
    cv2.resizeWindow(win, video_width, video_height)
    cv2.moveWindow(win, x_pos, y_pos)

    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    try:
        while True:
            frame, val = player.get_frame()
            if val == 'eof':
                break
            if frame is None:
                time.sleep(0.005)
                continue

            img, pts = frame
            if start_wall is None:
                start_wall = time.time() - (pts if pts is not None else 0.0)

            if pts is not None:
                target = start_wall + pts
                delay = target - time.time()
                if delay < -LATE_DROP_SEC:
                    continue
                elif delay > 0:
                    time.sleep(min(delay, 0.05))

            w, h = img.get_size()
            bytearray_img = img.to_bytearray()[0]
            frame_bgr = np.frombuffer(bytearray_img, dtype=np.uint8).reshape(h, w, 3)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

            cv2.imshow(win, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        player.close_player()
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        cv2.destroyWindow(win)


print("[INFO] Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    frame_area = float(H * W)

    results = model(frame, verbose=False)[0]
    annotated = results.plot()

    # Check for phone detection with size filter
    hit = False
    if results.boxes and len(results.boxes) > 0:
        for (cls_id, conf, xyxy) in zip(results.boxes.cls.tolist(),
                                        results.boxes.conf.tolist(),
                                        results.boxes.xyxy.tolist()):
            if cls_id in wanted_ids and conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = xyxy
                box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if box_area / frame_area >= MIN_BOX_AREA_RATIO:
                    hit = True
                    break

    # Debounce with consecutive streak
    if hit:
        streak_hits += 1
    else:
        streak_hits = 0  

    now = time.time()
    if streak_hits >= STREAK_REQUIRED and (now - last_trigger) > COOLDOWN_SEC:
        last_trigger = now
        streak_hits = 0  
        print("[EVENT] Phone detected → playing video.")
        play_video_ffpy(VIDEO_PATH)   # blocks until video is done (or 'q')

    cv2.imshow("YOLOv11 Realtime Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
