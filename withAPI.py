import cv2
import numpy as np
import requests
from ultralytics import YOLO
from sort.sort import Sort
from collections import deque, Counter
from datetime import datetime
from util import get_car, read_license_plate

# ==============================
# 1. Konfigurasi ORDS
# ==============================
ORDS_URL = "http://idtbintranetdev/ords/intranet/intranet/plate_log/"
ORDS_USER = None  # isi jika pakai Basic Auth
ORDS_PASS = None  # isi jika pakai Basic Auth

# ==============================
# 2. Konfigurasi Tracking & OCR
# ==============================
mot_tracker = Sort()
plate_history = {}  # {car_id: deque([...])}
last_seen = {}      # {car_id: frame_number}
posted_ids = set()  # simpan PRIMARY_ID yang sudah di-post
results = {}

MAX_MISSING_FRAMES = 3
INITIAL_HISTORY = 3
MAX_HISTORY = 10

vehicles = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
vehicle_labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ==============================
# 3. Fungsi Tracking Plat
# ==============================
def update_plate_history(car_id, plate_text, frame_nmr):
    """Update history plat dengan adaptive deque + voting"""
    last_seen[car_id] = frame_nmr
    if car_id not in plate_history:
        plate_history[car_id] = deque(maxlen=INITIAL_HISTORY)

    plate_history[car_id].append(plate_text)

    # Perbesar deque adaptif
    if len(plate_history[car_id]) == plate_history[car_id].maxlen and plate_history[car_id].maxlen < MAX_HISTORY:
        plate_history[car_id] = deque(plate_history[car_id], maxlen=plate_history[car_id].maxlen + 1)

    # Voting hasil OCR
    vote = Counter(plate_history[car_id])
    most_common_plate, _ = vote.most_common(1)[0]
    return most_common_plate

def reset_missing_cars(current_ids, frame_nmr):
    """Hapus history jika mobil hilang"""
    for car_id in list(last_seen.keys()):
        if car_id not in current_ids and frame_nmr - last_seen[car_id] > MAX_MISSING_FRAMES:
            plate_history.pop(car_id, None)
            last_seen.pop(car_id, None)

# ==============================
# 4. Load Model YOLO
# ==============================
coco_model = YOLO(
    r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\automatic-number-plate-recognition-python-yolov8-main\yolov8n_detection.pt'
)
license_plate_detector = YOLO(
    r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\runs\train\first\weights\best.pt'
)

# ==============================
# 5. Inisialisasi Kamera
# ==============================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_webcam.mp4', fourcc, fps, (width, height))

frame_nmr = -1


# ==============================
# 6. Loop Webcam
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}
    # --- 6a. Deteksi Kendaraan ---
    detections = coco_model(frame)[0]
    detections_ = [
        [x1, y1, x2, y2, score]
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
        if int(class_id) in vehicles
    ]

    # Tracking kendaraan
    track_ids = mot_tracker.update(np.asarray(detections_)) if detections_ else np.empty((0, 5))
    current_ids = set(track_ids[:, 4]) if len(track_ids) > 0 else set()
    reset_missing_cars(current_ids, frame_nmr)

    # --- 6b. Deteksi Plat Nomor ---
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

                # Visualisasi
                vehicle_type = vehicle_labels.get(int(detections.boxes.cls[0]), "Vehicle")
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_type} ID:{int(car_id)}", (int(xcar1), int(ycar1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, most_common_plate, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- 6c. Kirim Data ke ORDS hanya sekali per plat unik ---
                timestamp_vhc = datetime.now().strftime("%Y%m%d%H%M%S")  # format untuk ID unik
                primary_id = f"{int(car_id)}_{most_common_plate}_{timestamp_vhc}"

                if primary_id not in posted_ids:
                    posted_ids.add(primary_id)  # tandai sudah dikirim

                    payload = {
                        "primary_id": primary_id,
                        "vehicle_id": int(car_id),
                        "plate_number": most_common_plate,
                        "vehicle_type": vehicle_type,
                        "timestamp_vhc": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    if ORDS_USER and ORDS_PASS:
                        resp = requests.post(ORDS_URL, json=payload, auth=(ORDS_USER, ORDS_PASS))
                    else:
                        resp = requests.post(ORDS_URL, json=payload)

                    if resp.status_code in (200, 201):
                        print(f"[OK] Data terkirim: {payload}")
                    else:
                        print(f"[ERROR {resp.status_code}] {resp.text}")

    # --- 6d. Tampilkan & Simpan ---
    cv2.imshow("Webcam Vehicle + Plate Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 7. Cleanup
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()
print("Proses selesai! Data terkirim ke ORDS dan video disimpan.")
