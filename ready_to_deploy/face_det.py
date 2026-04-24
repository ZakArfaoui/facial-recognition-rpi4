import os, json, time
import numpy as np
import cv2

# Works on Windows (TF) and Pi (tflite-runtime)
try:
    import tflite_runtime.interpreter as tflite
    USING = "tflite_runtime"
except ImportError:
    import tensorflow.lite as tflite
    USING = "tensorflow.lite"

# ================== CONFIG ==================
MODEL_PATH = "facenet.tflite"

DB_PATH = "face_db_compact.npz"        # <- your compact DB
THRESH_PATH = "threshold_compact.json" # <- your compact threshold json

CAMERA_INDEX = 0        # 0 = default webcam; try 1 if needed
FRAME_W = 640
FRAME_H = 480

# Haar cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Speed-up detection: downscale before Haar
DETECT_DOWNSCALE_MAX_W = 640

# Optional: require minimum face size (avoid tiny false detections)
MIN_FACE_SIZE = 80  # pixels (after mapping back to original)
# ============================================

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def l2_normalize(v, eps=1e-10):
    return v / (np.linalg.norm(v) + eps)

def detect_biggest_face(bgr):
    h, w = bgr.shape[:2]
    scale = 1.0
    if w > DETECT_DOWNSCALE_MAX_W:
        scale = DETECT_DOWNSCALE_MAX_W / w
        bgr_small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None  # face, bbox

    x, y, fw, fh = max(faces, key=lambda r: r[2]*r[3])

    # map back
    if scale != 1.0:
        inv = 1.0 / scale
        x, y, fw, fh = int(x*inv), int(y*inv), int(fw*inv), int(fh*inv)

    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return None, None

    pad = int(0.20 * fw)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(bgr.shape[1], x + fw + pad)
    y2 = min(bgr.shape[0], y + fh + pad)

    face = bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    return face, (x1, y1, x2, y2)

# -------- load model --------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

itp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
itp.allocate_tensors()
inp = itp.get_input_details()[0]
out = itp.get_output_details()[0]
H, W = int(inp["shape"][1]), int(inp["shape"][2])

print("✅ Using:", USING)
print("Model input:", inp["shape"], inp["dtype"], "output:", out["shape"], out["dtype"])

def preprocess(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (rgb - 127.5) / 128.0
    return np.expand_dims(x, axis=0).astype(np.float32)

def embed(face_bgr):
    x = preprocess(face_bgr)
    itp.set_tensor(inp["index"], x)
    itp.invoke()
    e = itp.get_tensor(out["index"])[0].astype(np.float32)
    return l2_normalize(e)

# -------- load DB + threshold --------
if not os.path.isfile(DB_PATH):
    raise FileNotFoundError(f"Missing DB: {DB_PATH}")

db_npz = np.load(DB_PATH, allow_pickle=True)

names = []
protos = []
for k in db_npz.files:
    E = db_npz[k].astype(np.float32)  # shape (K,512) where K=1 if centroid
    names.append(k)
    protos.append(E)

# Stack all prototypes into one matrix for fast dot products
# Keep an index map to recover the name for each row
proto_mat = []
proto_name_idx = []
for name, E in zip(names, protos):
    for row in E:
        proto_mat.append(row)
        proto_name_idx.append(name)

proto_mat = np.stack(proto_mat, axis=0)  # (M,512)

# Threshold
if not os.path.isfile(THRESH_PATH):
    print("⚠️ threshold file missing, using default 0.65")
    threshold = 0.65
else:
    with open(THRESH_PATH, "r") as f:
        threshold = float(json.load(f).get("threshold", 0.65))

print("Loaded identities:", sorted(set(proto_name_idx)))
print("Total prototypes:", proto_mat.shape[0])
print("Threshold:", threshold)

# -------- camera --------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera. Try CAMERA_INDEX=1 or 2.")

prev_t = time.time()
fps = 0.0

print("\nPress 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Camera frame read failed")
        break

    t1 = time.time()

    face, bbox = detect_biggest_face(frame)

    label = "No face"
    best_sim = -1.0

    if face is not None:
        e = embed(face)

        sims = proto_mat @ e  # (M,)
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        best_name = proto_name_idx[idx]

        if best_sim >= threshold:
            label = best_name
        else:
            label = "Unknown"

        # draw bbox
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # FPS
    dt = t1 - prev_t
    prev_t = t1
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # overlay text
    cv2.putText(frame, f"{label}  sim={best_sim:.3f}  thr={threshold:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Face Recognition (Embedding + Similarity)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
