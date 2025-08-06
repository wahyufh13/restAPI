# util.py - Optimized Version
import re
import cv2
import string
import ultralytics
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR sekali saja dengan konfigurasi optimal
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=True, show_log=False)

# Mapping dictionaries untuk koreksi OCR
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Cache untuk preprocessing hasil
preprocessing_cache = {}

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def license_complies_format(text):
    """
    Validasi format plat Indonesia:
    * 1 karakter pertama harus huruf
    * Karakter kedua sampai ke-8 bisa huruf bisa angka
    * Karakter terakhir pasti huruf
    * Panjang minimal 3, maksimal 9 karakter
    """
    text = text.upper().replace(' ', '')
    
    # Cek panjang
    if len(text) < 3 or len(text) > 9:
        return False
    
    # Karakter pertama harus huruf
    if not text[0].isalpha():
        return False
    
    # Karakter terakhir harus huruf
    if not text[-1].isalpha():
        return False
    
    # Karakter tengah (index 1 sampai -2) bisa huruf atau angka
    for char in text[1:-1]:
        if not (char.isalpha() or char.isdigit()):
            return False
    
    return True

def format_license(text):
    """Koreksi format plat nomor Indonesia agar huruf/angka benar"""
    text = text.upper().replace(' ', '')
    
    if len(text) == 0:
        return text
    
    # Optimized: hitung digit tengah lebih efisien
    digit_count = 0
    for i, c in enumerate(text[1:], 1):
        if c.isdigit() or c in dict_char_to_int:
            digit_count += 1
        else:
            break
    
    # Pisah bagian
    front_letter = text[0]
    middle_digits = text[1:1+digit_count]
    back_letters = text[1+digit_count:]
    
    # Koreksi menggunakan dict.get untuk efisiensi
    front_letter = dict_int_to_char.get(front_letter, front_letter)
    fixed_middle = ''.join(dict_char_to_int.get(c, c) for c in middle_digits)
    fixed_back = ''.join(dict_int_to_char.get(c, c) for c in back_letters)
    
    return front_letter + fixed_middle + fixed_back

def preprocess_license_plate(license_plate_crop):
    """Optimized preprocessing with caching"""
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None
    
    # Simple hash untuk caching (tidak perfect tapi cukup untuk frame sequence)
    crop_hash = hash(license_plate_crop.tobytes())
    
    if crop_hash in preprocessing_cache:
        return preprocessing_cache[crop_hash]
    
    # Preprocessing yang lebih efisien
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Coba metode yang lebih cepat dulu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Simpan ke cache (batasi ukuran cache)
    if len(preprocessing_cache) > 100:
        preprocessing_cache.clear()
    
    preprocessing_cache[crop_hash] = thresh
    return thresh

def read_license_plate(license_plate_crop, conf_threshold=0.9):
    """OCR plat nomor menggunakan PaddleOCR + optimized preprocessing"""
    processed_image = preprocess_license_plate(license_plate_crop)
    if processed_image is None:
        return None, None
    
    try:
        # OCR dengan timeout implisit
        result = ocr.ocr(processed_image, cls=False)  # Disable cls untuk speed
        if not result or result[0] is None:
            return None, None
        
        best_text = None
        best_score = 0
        
        for line in result[0]:
            raw_text = line[1][0]
            score = line[1][1]
            
            text = clean_text(raw_text)
            
            # Debug print untuk melihat semua hasil OCR
            print(f"OCR Raw: '{raw_text}' -> Clean: '{text}' -> Score: {score:.3f} -> Valid: {license_complies_format(text)}")
            
            if (license_complies_format(text) and 
                score > best_score and 
                score >= conf_threshold):
                best_text = format_license(text)
                best_score = score
                print(f"✓ Best OCR Updated: '{best_text}' with score {best_score:.3f}")
        
        return (best_text, best_score) if best_text else (None, None)
    
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None

def get_car(license_plate, vehicle_track_ids):
    """Optimized car matching with early exit"""
    if len(vehicle_track_ids) == 0:
        return -1, -1, -1, -1, -1
    
    x1, y1, x2, y2, score, class_id = license_plate
    
    # Vectorized approach untuk efisiensi
    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    
    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    """Write results to CSV - unchanged for compatibility"""
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if ('car' in results[frame_nmr][car_id].keys() and 
                    'license_plate' in results[frame_nmr][car_id].keys() and 
                    'text' in results[frame_nmr][car_id]['license_plate'].keys()):
                    
                    car_bbox = results[frame_nmr][car_id]['car']['bbox']
                    lp_bbox = results[frame_nmr][car_id]['license_plate']['bbox']
                    
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, car_id,
                        '[{} {} {} {}]'.format(*car_bbox),
                        '[{} {} {} {}]'.format(*lp_bbox),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score'])
                    )

# main.py - Optimized Version
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort.sort import Sort
from collections import deque, defaultdict
from util import get_car, read_license_plate, write_csv

class OptimizedLicensePlateRecognition:
    def __init__(self):
        # ==============================
        # 1. Inisialisasi
        # ==============================
        self.results = {}
        self.mot_tracker = Sort()
        self.final_plates = {}
        
        # History plat & last seen - menggunakan defaultdict untuk efisiensi
        self.plate_history = defaultdict(lambda: deque(maxlen=10))  # Increase history untuk stability
        self.plate_scores = defaultdict(list)  # Track scores untuk setiap detection
        self.last_seen = {}
        self.confirmed_plates = {}  # Store plat yang sudah dikonfirmasi dengan confidence tinggi
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.skip_frames = 0  # Skip frame counter untuk optimization
        
        # Load YOLO models dengan optimizations
        print("Loading YOLO models...")
        self.coco_model = YOLO(r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\automatic-number-plate-recognition-python-yolov8-main\yolov8n_detection.pt'
)  # Gunakan model standard jika path tidak ada
        self.license_plate_detector = YOLO(r'C:\Users\adity\Downloads\automatic-number-plate-recognition-python-yolov8-main\runs\train\first\weights\best.pt')  # Fallback model
        
        # Optimasi YOLO
        self.coco_model.fuse()  # Fuse conv dan bn layers untuk speed
        # self.license_plate_detector.fuse()
        
        # Video setup
        self.setup_video()
        
        # Constants
        self.vehicles = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
        self.vehicle_labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        self.MAX_MISSING_FRAMES = 10
        self.CONFIDENCE_THRESHOLD = 0.5
        self.SKIP_FRAME_INTERVAL = 2  # Process every 2nd frame for license plates
        self.MIN_OCR_CONFIDENCE = 0.8  # Minimum OCR confidence untuk menampilkan hasil
        
    def setup_video(self):
        """Setup video capture dan output dengan optimasi"""
        self.cap = cv2.VideoCapture(1)
        
        # Optimasi webcam settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer untuk real-time
        
        # Video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter('output_webcam_optimized.mp4', fourcc, fps, (width, height))
        
    def update_performance_stats(self, frame_time):
        """Update dan display performance statistics"""
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) >= 10:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\rAvg FPS: {fps:.1f} | Frame Time: {frame_time*1000:.1f}ms", end="")
    
    def cleanup_tracking_history(self, current_ids, frame_nmr):
        """Cleanup tracking history untuk mobil yang hilang"""
        for car_id in list(self.last_seen.keys()):
            if (car_id not in current_ids and 
                frame_nmr - self.last_seen[car_id] > self.MAX_MISSING_FRAMES):
                # Reset history untuk mobil yang hilang
                self.plate_history.pop(car_id, None)
                self.plate_scores.pop(car_id, None)
                self.confirmed_plates.pop(car_id, None)
                self.last_seen.pop(car_id, None)
                print(f"Car ID {car_id} removed from tracking (missing for {self.MAX_MISSING_FRAMES} frames)")
    
    def process_vehicle_detection(self, frame):
        """Process vehicle detection dengan optimasi"""
        # Gunakan confidence threshold yang lebih tinggi untuk mengurangi false positives
        detections = self.coco_model(frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # Tracking kendaraan
        if len(detections_) > 0:
            track_ids = self.mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = np.empty((0, 5))
            
        return track_ids, detections
    
    def process_license_plate_detection(self, frame, track_ids, frame_nmr):
        """Process license plate detection dengan frame skipping dan improved logic"""
        # Skip frames untuk license plate detection (computationally expensive)
        if frame_nmr % self.SKIP_FRAME_INTERVAL != 0:
            return
        
        license_plates = self.license_plate_detector(frame, conf=0.3, verbose=False)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Assign plat ke mobil terdekat
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # Update last seen
                self.last_seen[car_id] = frame_nmr
                
                # Crop plat nomor dengan bounds checking
                x1p = max(int(x1), 0)
                y1p = max(int(y1), 0)
                x2p = min(int(x2), frame.shape[1])
                y2p = min(int(y2), frame.shape[0])
                
                if x2p > x1p and y2p > y1p:  # Valid crop area
                    license_plate_crop = frame[y1p:y2p, x1p:x2p]
                    
                    # OCR plat nomor dengan confidence threshold tinggi
                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop, conf_threshold=self.MIN_OCR_CONFIDENCE
                    )
                    
                    # Hanya simpan jika hasil OCR memenuhi confidence threshold
                    if license_plate_text is not None and license_plate_text_score >= self.MIN_OCR_CONFIDENCE:
                        print(f"✓ Frame {frame_nmr}: Car {car_id} -> '{license_plate_text}' (score: {license_plate_text_score:.3f})")
                        
                        # Simpan ke history dengan score
                        self.plate_history[car_id].append(license_plate_text)
                        self.plate_scores[car_id].append(license_plate_text_score)
                        
                        # Limit scores history sesuai dengan plate history
                        if len(self.plate_scores[car_id]) > 10:
                            self.plate_scores[car_id] = self.plate_scores[car_id][-10:]
                        
                        # Tentukan plat terbaik berdasarkan voting dan confidence
                        best_plate = self.get_best_plate(car_id)
                        
                        if best_plate:
                            # Konfirmasi plat jika sudah stabil
                            self.confirmed_plates[car_id] = best_plate
                            
                            self.results[frame_nmr][car_id] = {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': best_plate['text'],
                                    'bbox_score': score,
                                    'text_score': best_plate['score']
                                }
                            }
                    else:
                        if license_plate_text is None:
                            print(f"✗ Frame {frame_nmr}: Car {car_id} -> No valid OCR result")
                        else:
                            print(f"✗ Frame {frame_nmr}: Car {car_id} -> '{license_plate_text}' (score: {license_plate_text_score:.3f}) - Below threshold")
    
    def get_best_plate(self, car_id):
        """Dapatkan plat nomor terbaik berdasarkan voting dan confidence score"""
        if car_id not in self.plate_history or len(self.plate_history[car_id]) == 0:
            return None
        
        plates = list(self.plate_history[car_id])
        scores = list(self.plate_scores[car_id])
        
        # Buat dictionary untuk menghitung votes dan average score
        plate_analysis = {}
        
        for i, plate in enumerate(plates):
            if plate not in plate_analysis:
                plate_analysis[plate] = {'count': 0, 'scores': [], 'avg_score': 0}
            
            plate_analysis[plate]['count'] += 1
            if i < len(scores):
                plate_analysis[plate]['scores'].append(scores[i])
        
        # Hitung average score untuk setiap plat
        for plate_data in plate_analysis.values():
            if plate_data['scores']:
                plate_data['avg_score'] = sum(plate_data['scores']) / len(plate_data['scores'])
        
        # Pilih plat dengan kombinasi terbaik dari vote count dan average score
        best_plate_text = None
        best_combined_score = 0
        
        for plate_text, data in plate_analysis.items():
            # Combined score: (vote_weight * normalized_count) + (confidence_weight * avg_score)
            vote_weight = 0.3
            confidence_weight = 0.7
            
            max_votes = max(d['count'] for d in plate_analysis.values())
            normalized_count = data['count'] / max_votes
            
            combined_score = (vote_weight * normalized_count) + (confidence_weight * data['avg_score'])
            
            print(f"    Plate '{plate_text}': votes={data['count']}, avg_score={data['avg_score']:.3f}, combined={combined_score:.3f}")
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_plate_text = plate_text
        
        if best_plate_text:
            return {
                'text': best_plate_text,
                'score': plate_analysis[best_plate_text]['avg_score'],
                'votes': plate_analysis[best_plate_text]['count']
            }
        
        return None
    
    def draw_visualizations(self, frame, track_ids, detections, frame_nmr):
        """Draw bounding boxes dan text dengan optimasi - hanya tampilkan hasil OCR terbaik"""
        # Draw vehicle tracks
        for track in track_ids:
            x1, y1, x2, y2, car_id = track
            car_id = int(car_id)
            
            # Default vehicle type
            vehicle_name = "Vehicle"
            
            # Find vehicle type (simplified lookup)
            for det in detections.boxes.data.tolist():
                det_class = int(det[5])
                if det_class in self.vehicles:
                    vehicle_name = self.vehicle_labels.get(det_class, "Vehicle")
                    break
            
            # Draw vehicle bounding box
            color_vehicle = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_vehicle, 2)
            
            # Draw vehicle label
            vehicle_label_text = f"{vehicle_name} ID:{car_id}"
            cv2.putText(frame, vehicle_label_text, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_vehicle, 2)
        
        # Draw HANYA license plate results yang sudah dikonfirmasi
        for car_id in track_ids[:, 4].astype(int):
            if car_id in self.confirmed_plates:
                plate_data = self.confirmed_plates[car_id]
                
                # Cari bounding box untuk car_id ini dari current results atau track_ids
                car_bbox = None
                lp_bbox = None
                
                # Cek dari results frame ini
                if frame_nmr in self.results and car_id in self.results[frame_nmr]:
                    if 'license_plate' in self.results[frame_nmr][car_id]:
                        lp_data = self.results[frame_nmr][car_id]['license_plate']
                        lp_bbox = lp_data['bbox']
                
                # Jika ada license plate bbox, gambar
                if lp_bbox:
                    x1, y1, x2, y2 = lp_bbox
                    
                    # Draw license plate bounding box dengan warna yang menunjukkan confidence
                    confidence = plate_data['score']
                    if confidence >= 0.9:
                        color = (0, 255, 0)  # Hijau untuk confidence tinggi
                    elif confidence >= 0.8:
                        color = (0, 255, 255)  # Kuning untuk confidence medium
                    else:
                        color = (0, 0, 255)  # Merah untuk confidence rendah
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw license plate text dengan informasi tambahan
                    plate_text = f"{plate_data['text']} ({confidence:.2f})"
                    votes_text = f"Votes: {plate_data['votes']}"
                    
                    # Background untuk text agar lebih jelas
                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - 35), 
                                (int(x1) + text_size[0] + 10, int(y1)), (0, 0, 0), -1)
                    
                    cv2.putText(frame, plate_text, (int(x1) + 5, int(y1) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, votes_text, (int(x1) + 5, int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main loop dengan performance monitoring"""
        frame_nmr = -1
        
        print("Starting optimized license plate recognition...")
        print("Press 'q' to quit")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_nmr += 1
                self.results[frame_nmr] = {}
                
                # Process vehicle detection
                track_ids, detections = self.process_vehicle_detection(frame)
                
                # Update tracking history
                current_ids = set(track_ids[:, 4]) if len(track_ids) > 0 else set()
                self.cleanup_tracking_history(current_ids, frame_nmr)
                
                # Process license plate detection (with frame skipping)
                self.process_license_plate_detection(frame, track_ids, frame_nmr)
                
                # Draw visualizations
                self.draw_visualizations(frame, track_ids, detections, frame_nmr)
                
                # Performance info on frame
                frame_time = time.time() - start_time
                fps = 1.0 / frame_time if frame_time > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display dan save
                cv2.imshow("Optimized Vehicle + Plate Detection", frame)
                self.out.write(frame)
                
                # Update performance stats
                self.update_performance_stats(frame_time)
                
                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # Save results
        if self.results:
            write_csv(self.results, 'output_results_optimized.csv')
            print("Results saved to output_results_optimized.csv")

if __name__ == "__main__":
    recognizer = OptimizedLicensePlateRecognition()
    recognizer.run()