import re
import cv2
import string
from paddleocr import PaddleOCR

# Initialize PaddleOCR sekali saja
ocr = PaddleOCR(lang='en', use_angle_cls=True)

# Mapping dictionaries untuk koreksi OCR
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Bersihkan simbol selain huruf/angka
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def license_complies_format(text):
    """
    Validasi format plat Indonesia:
    1-2 huruf + 1-4 angka + 1-3 huruf
    """
    text = text.upper().replace(' ', '')

    if len(text) < 3 or len(text) > 10:
        return False

    # # --- Cari huruf depan (maks 2)
    # prefix_len = 1
    # if len(text) >= 2 and text[1].isalpha():
    #     prefix_len = 2


    # # Hitung angka tengah
    # digit_count = 0
    # for c in text[prefix_len:]:
    #     if c.isdigit() or c in dict_char_to_int:
    #         digit_count += 1
    #     else:
    #         break

    # # Minimal 1 angka, maksimal 4 angka
    # if digit_count < 1 or digit_count > 4:
    #     return False

    # # Sisa huruf belakang maksimal 3
    # back_len = len(text) - (prefix_len + digit_count)
    # if back_len < 0 or back_len > 3:
    #     return False

    return True



def format_license(text):
    """
    Koreksi format plat nomor Indonesia agar huruf/angka benar
    """
    text = text.upper().replace(' ', '')

    # Hitung digit tengah
    digit_count = 0
    for c in text[1:]:
        if c.isdigit() or c in dict_char_to_int:
            digit_count += 1
        else:
            break

    # Pisah bagian
    front_letter = text[0] if len(text) > 0 else ''
    middle_digits = text[1:1+digit_count]
    back_letters = text[1+digit_count:]

    # Koreksi huruf depan
    if front_letter in dict_int_to_char:
        front_letter = dict_int_to_char[front_letter]

    # Koreksi angka tengah
    fixed_middle = ''.join(dict_char_to_int.get(c, c) for c in middle_digits)

    # Koreksi huruf belakang
    fixed_back = ''.join(dict_int_to_char.get(c, c) for c in back_letters)

    return front_letter + fixed_middle + fixed_back


def read_license_plate(license_plate_crop, conf_threshold=0.8):
    """
    OCR plat nomor menggunakan PaddleOCR + preprocessing
    """
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    # Preprocessing
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)  # tingkatkan kontras
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # OCR
    result = ocr.ocr(thresh, cls=True)
    if not result or result[0] is None:
        return None, None  # tidak ada teks terdeteksi

    best_text = None
    best_score = 0

    for line in result[0]:
        raw_text = line[1][0]
        score = line[1][1]

        # Bersihkan simbol dan kapitalisasi
        text = clean_text(raw_text)

        if license_complies_format(text) and score > best_score and score >= conf_threshold:
            best_text = format_license(text)
            best_score = score

    return (best_text, best_score) if best_text else (None, None)


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

