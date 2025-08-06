import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort.sort import Sort
from collections import deque, Counter
from util import get_car, read_license_plate, write_csv

# ==============================
# 1. Inisialisasi
# ==============================
results = {}
mot_tracker = Sort()
final_plates = {}

MAX_MISSING_FRAMES = 3        # reset jika hilang 3 frame
INITIAL_HISTORY = 3           # awalnya 3 frame
MAX_HISTORY = 10              # panjang maksimal deque

plate_history = {}  # {car_id: deque([...])}
last_seen = {}      # {car_id: frame_number}

# Fungsi untuk update history
def update_plate_history(car_id, plate_text, frame_nmr):
    """Update history plat dengan adaptive deque + voting"""
    last_seen[car_id] = frame_nmr

    # Mobil baru muncul
    if car_id not in plate_history:
        plate_history[car_id] = deque(maxlen=INITIAL_HISTORY)

    # Tambahkan hasil OCR terbaru
    plate_history[car_id].append(plate_text)

    # Adaptive deque
    if len(plate_history[car_id]) == plate_history[car_id].maxlen and plate_history[car_id].maxlen < MAX_HISTORY:
        plate_history[car_id] = deque(plate_history[car_id], maxlen=plate_history[car_id].maxlen + 1)

    # Voting hasil OCR
    vote = Counter(plate_history[car_id])
    most_common_plate, _ = vote.most_common(1)[0]
    return most_common_plate

# Fungsi reset jika mobil hilang
def reset_missing_cars(current_ids, frame_nmr):
    for car_id in list(last_seen.keys()):
        if car_id not in current_ids and frame_nmr - last_seen[car_id] > MAX_MISSING_FRAMES:
            plate_history.pop(car_id, None)
            last_seen.pop(car_id, None)

# ==============================
# 2. Load Model YOLO
# ==============================
coco_model = YOLO(
    r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\automatic-number-plate-recognition-python-yolov8-main\yolov8n_detection.pt'
)
license_plate_detector = YOLO(
    r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\runs\train\first\weights\best.pt'
)

# Gunakan webcam (0 untuk internal, 1 untuk eksternal)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_webcam.mp4', fourcc, fps, (width, height))

# COCO IDs for vehicles
vehicles = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
vehicle_labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ==============================
# 3. Loop Webcam
# ==============================
frame_nmr = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}

    # --- 3a. Deteksi Kendaraan ---
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Tracking kendaraan
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))

    # Reset history jika mobil hilang
    current_ids = set(track_ids[:, 4]) if len(track_ids) > 0 else set()
    reset_missing_cars(current_ids, frame_nmr)

    # --- 3b. Deteksi Plat Nomor ---
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Cek mobil terdekat
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop plat
            x1p, y1p, x2p, y2p = int(x1), int(y1), int(x2), int(y2)
            license_plate_crop = frame[y1p:y2p, x1p:x2p]

            # OCR
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            if license_plate_text:
                most_common_plate = update_plate_history(car_id, license_plate_text, frame_nmr)

                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': most_common_plate,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

                # --- 3c. Visualisasi ---
                # Nama kendaraan dari YOLO
                vehicle_name = vehicle_labels.get(int(detections.boxes.cls[0]), "Vehicle")

                color_vehicle = (0, 255, 0)
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), color_vehicle, 2)
                vehicle_label_text = f"{vehicle_name} ID:{int(car_id)}"
                cv2.putText(frame, vehicle_label_text, (int(xcar1), int(ycar1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_vehicle, 2)

                # Bounding box plat
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, most_common_plate, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 3d. Tampilkan & Simpan ---
    cv2.imshow("Webcam Vehicle + Plate Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 4. Simpan Hasil
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

write_csv(results, './test_webcam.csv')
print("Proses selesai! Hasil tersimpan di output_webcam.mp4 dan test_webcam.csv")
