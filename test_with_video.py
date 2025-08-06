import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

# ==============================
# 1. Inisialisasi
# ==============================
results = {}
mot_tracker = Sort()
final_plates = {}

# Load YOLO models
coco_model = YOLO(r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\automatic-number-plate-recognition-python-yolov8-main\yolov8n_detection.pt')
license_plate_detector = YOLO(r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\runs\train\first\weights\best.pt')

# Load video
video_path = r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\automatic-number-plate-recognition-python-yolov8-main\sample.mp4'
cap = cv2.VideoCapture(video_path)

# # Gunakan webcam (0 untuk webcam default, 1 untuk eksternal)
# cap = cv2.VideoCapture(1)

# Video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_detected5.mp4', fourcc, fps, (width, height))

# COCO IDs for vehicles
vehicles = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# ==============================
# 2. Loop Baca Frame
# ==============================
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # -----------------------------
    # 2a. Deteksi Kendaraan
    # -----------------------------
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Tracking kendaraan
    track_ids = mot_tracker.update(np.asarray(detections_))

    # -----------------------------
    # 2b. Deteksi Plat Nomor
    # -----------------------------
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign plat ke mobil terdekat
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop plat nomor
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # Preprocessing OCR
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # OCR plat nomor
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

                # -----------------------------
                # 2c. Visualisasi di Frame
                # -----------------------------
                # Bounding box kendaraan
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                # Bounding box plat
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # Teks plat
                cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("License Plate Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 3. Simpan Hasil
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

write_csv(results, './test.csv')
print("Proses selesai! Hasil tersimpan di output_detected.mp4 dan test.csv")

# 4. Hitung Plat Nomor Final per Mobil
# ==============================
df = pd.read_csv('./test.csv')
df_valid = df[df['license_number'] != '0']

final_plates = {}

for car_id in df_valid['car_id'].unique():
    df_car = df_valid[df_valid['car_id'] == car_id]
    freq = df_car['license_number'].value_counts()
    most_frequent_plate = freq.index[0]
    avg_confidence = df_car[df_car['license_number'] == most_frequent_plate]['license_number_score'].mean()

    final_plates[car_id] = {
        'plate_number': most_frequent_plate,
        'average_confidence': avg_confidence,
        'frequency': freq.iloc[0]
    }

print("\nPlat final per mobil:")
for car_id, info in final_plates.items():
    print(f"Car ID {car_id}: {info}")

# Simpan hasil ke Excel
pd.DataFrame.from_dict(final_plates, orient='index').to_excel('final_plates.xlsx')
print("Plat final per mobil tersimpan di final_plates.xlsx")