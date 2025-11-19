from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
from PIL import Image,ImageOps
import io
import base64
import cv2
import re
import numpy as np
from typing import Dict
import paddle
from paddleocr import PaddleOCR
from qreader import QReader
import cv2
from pyaadhaar.decode import AadhaarSecureQr,AadhaarOldQr,AadhaarQRXML
import xml.dom.minidom
import xml.etree.ElementTree as ET
from starlette.concurrency import run_in_threadpool # Import run_in_threadpool
from pyzbar.pyzbar import decode

gpu_available  = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)

qreader_instance = QReader()
ocr = PaddleOCR(use_angle_cls=True, lang='en')
app = FastAPI(
    title="Identify Face Verification API",
    description="An API to verify two faces.",
    version="1.0.0",
)

@app.post("/verify_faces")
async def verify_faces(
    file1: UploadFile = File(..., description="Aadhaar / Pancard Image"),
    file2: UploadFile = File(..., description="Customer Identity Image")
):
    """
    Verifies if two faces in the provided images belong to the same person.

    Expects two image files as input.
    Returns a dictionary with verification results from DeepFace.
    """
    try:
        # Read image bytes
        contents1 = await file1.read()
        contents2 = await file2.read()

        # Open images using PIL and convert to RGB (DeepFace expects RGB or BGR, but PIL default is RGB)
        # DeepFace can also handle numpy arrays directly, which we convert to
        image1 = Image.open(io.BytesIO(contents1)).convert("RGB")
        image2 = Image.open(io.BytesIO(contents2)).convert("RGB")

        # Convert PIL Image to NumPy array (DeepFace accepts NumPy arrays)
        image1_np = np.array(image1)
        image2_np = np.array(image2)

        # Perform face verification using DeepFace
        # You can customize model_name, detector_backend, distance_metric
        # and enforce_detection based on your needs.
        # enforce_detection=True (default) will raise an error if no face is detected.
        # enforce_detection=False will proceed even if no face is detected, but results
        # might be less reliable.
        verification_result = DeepFace.verify(
            img1_path=image1_np,
            img2_path=image2_np,
            model_name="ArcFace",  # Common choices: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace"
            detector_backend="retinaface", # Common choices: "opencv", "ssd", "dlib", "mtcnn", "retinaface"
        )

        return {
            "status": "success",
            "message": "Face verification completed.",
            "result": verification_result
        }

    except ValueError as e:
        # DeepFace raises ValueError if no face is detected (if enforce_detection=True)
        raise HTTPException(status_code=400, detail=f"Face detection error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

def extract_aadhar(decoded_texts):
    try:
        # Print the results
        for text in decoded_texts:
            print(text)
            try:
                parsedxml = ET.fromstring(text.data, parser=ET.XMLParser(encoding="utf-8"))
                obj = AadhaarQRXML(text.data)
            except:
                print(text)
                obj = AadhaarSecureQr(int(text.data))
            return obj.decodeddata()
    except Exception as e:
        return {
            "status" : "Error",
            "Error" : str(print(e))
        }
        

def extract_info(texts):
    joined_text = ' '.join(texts).upper()

    # PAN Card Detection
    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', joined_text)
    dob_match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', joined_text)

    if "INCOME TAX" in joined_text or "GOVT. OF INDIA" in joined_text or pan_match:
        name = next((t for t in texts if re.match(r'^[A-Z]{3,}$', t) and 'GOVT' not in t and 'INCOME' not in t), None)
        return {
            "document_type": "PAN",
            "name": name,
            "pan_number": pan_match.group(1) if pan_match else None,
            "dob": dob_match.group(1) if dob_match else None,
            "valid": True
        }

    # Aadhaar Detection
    aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', joined_text)
    if "AADHAAR" in joined_text or aadhaar_match:
        name = next((t for t in texts if re.match(r'^[A-Z ]{3,}$', t) and not any(c in t for c in '0123456789')), None)
        return {
            "document_type": "AADHAAR",
            "name": name,
            "aadhaar_number": aadhaar_match.group(0) if aadhaar_match else None,
            "dob": dob_match.group(1) if dob_match else None,
            "valid": True
        }

    # Voter ID Detection
    if "ELECTION" in joined_text or "ELECTOR" in joined_text or "VOTER" in joined_text:
        voter_id = next((t for t in texts if re.match(r'^[A-Z]{3}[0-9]{7}$', t)), None)
        name = next((t for t in texts if re.match(r'^[A-Z ]{3,}$', t)), None)
        return {
            "document_type": "VOTER ID",
            "name": name,
            "voter_id_number": voter_id,
            "dob": dob_match.group(1) if dob_match else None,
            "valid": True
        }

    return {
        "document_type": "UNKNOWN",
        "valid": False,
        "text":texts
    }

# gender detection with this:
def extract_gender(text: str) -> str:
    """Simplified gender extraction from text (case-insensitive)"""
    if not text:
        return None
    
    text_lower = text.lower()  # Convert once for case-insensitive check
    
    # Check for female (any case, including substrings like "Quu/Female")
    if "female" in text_lower:
        return "FEMALE"
    # Check male (accounting for OCR errors like "ma1e")
    elif "male" in text_lower or "ma1e" in text_lower:
        return "MALE"
    # Check transgender variations
    elif "transgender" in text_lower or "trans" in text_lower:
        return "TRANSGENDER"
    
    return None

@app.post("/validate-id")
async def validate_id_proof(file1: UploadFile = File(..., description="Aadhaar / Pancard Image"),):
    try:
        # Decode base64 image
        contents1 = await file1.read()
        id_image = io.BytesIO(contents1)
        #image_data = base64.b64decode(contents1)
        #pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        

        pil_image = Image.open(id_image).convert("RGB")
        # Enhance for OCR
        gysc_image = ImageOps.grayscale(pil_image)
        autoctrst_image = ImageOps.autocontrast(gysc_image)
        image = cv2.cvtColor(np.array(autoctrst_image), cv2.COLOR_GRAY2BGR)

        
        # OCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        results = ocr.predict(image)
        result_dict = results[0]
        texts = result_dict.get("rec_texts", [])
        scores = result_dict.get("rec_scores", [])

        # print(texts)
        # return texts

        if not texts:
            raise HTTPException(status_code=400, detail="OCR failed: No text detected")

        combined_text = " ".join(texts).upper()
        high_conf_texts = [
            t for t, s in zip(texts, scores)
            if s > 0.7 and t.isalpha() and t.isupper() and len(t) > 8
        ]

        document_type = "UNKNOWN"

        pan_regex = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
        aadhaar_regex = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
        voter_regex = r"\b[A-Z]{3}\d{7}\b"

        if re.search(pan_regex, combined_text):
            document_type = "PAN"
        elif re.search(aadhaar_regex, combined_text) or "AADHAAR" in combined_text or "UIDAI" in combined_text:
            document_type = "AADHAAR"
            print("Document type is Aadhaar")
            #decoded_texts = qreader_instance.detect_and_decode(image=aadhaar_image,return_detections=False)
            #decoded_texts = await run_in_threadpool(qreader_instance.detect_and_decode, image=image_np)
             # Use pyzbar to decode the QR code
            qr_codes = decode(autoctrst_image)
            print(qr_codes)
            return extract_aadhar(qr_codes)
        elif re.search(voter_regex, combined_text) or "ELECTION COMMISSION" in combined_text or "EPIC" in combined_text:
            document_type = "VOTER_ID"

        # -------------------------
        # Extract common fields
        # -------------------------
        name = None
        dob = None
        id_number = None
        gender = None  # <--- ADD THIS

        if document_type == "PAN":
            pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", combined_text)
            id_number = pan_match.group(1) if pan_match else None
            dob_match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", combined_text)
            dob = dob_match.group(1) if dob_match else None
            # PAN name logic
            for i in range(len(texts)):
                line = texts[i].strip().upper()
                if "INCOMETAXDEPARTMENT" in line or "INCOME TAX" in line:
                    for next_line in texts[i+1:i+3]:
                        if re.fullmatch(r"[A-Z\s]{6,}", next_line):
                            name = next_line.strip()
                            break
                if name:
                    break
            # ------------------------
        elif document_type == "AADHAAR":
            # Try full match first
            aadhaar_match = re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", combined_text)
            if aadhaar_match:
                id_number = aadhaar_match.group(0).replace("-", " ").replace("\u200c", "").strip()
            else:
                # Collect all numeric chunks that are 4 to 5 digits (to catch partial splits)
                numeric_chunks = [t for t in texts if re.fullmatch(r"\d{4,5}", t)]
                combined = ''.join(numeric_chunks)
                if len(combined) >= 12:
                    id_number = f"{combined[:4]} {combined[4:8]} {combined[8:12]}"
                else:
                    id_number = None


            dob_match = re.search(r"(DOB|YOB|DATE OF BIRTH)?[:\s/]*([0-9]{2}/[0-9]{2}/[0-9]{4}|[0-9]{4})", combined_text)
            dob = dob_match.group(2) if dob_match else None
            # Extract gender
            # ---------------------------------
            gender = extract_gender(combined_text)  # Call this instead of your current logic
              # 🔍 Aadhaar name extraction based on position before DOB
            if dob:
                for i in range(len(texts)):
                    if dob in texts[i]:
                        for j in range(i-1, max(i-4, -1), -1):
                            candidate = texts[j].strip()
                            if re.fullmatch(r"[A-Za-z\s]{6,}", candidate) and "GOVERNMENT" not in candidate.upper():
                                name = candidate
                                break
                        break
        elif document_type == "VOTER_ID":
            voter_match = re.search(r"\b([A-Z]{3}[0-9]{7})\b", combined_text)
            id_number = voter_match.group(1) if voter_match else None
            dob_match = re.search(r"\b(\d{2}/\d{2}/\d{4}|\d{4})\b", combined_text)
            dob = dob_match.group(1) if dob_match else None
            for line in texts:
                match = re.search(r"ELECTOR['’`S\s]*NAME[:\s]*(.*)", line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    break

        # -------------------------
        # Extract name
        # -------------------------
        blacklist = {
            "INCOMETAXDEPARTMENT", "GOVTOFINDIA", "PERMANENTACCOUNTNUMBER",
            "UNIQUEIDENTIFICATIONAUTHORITY", "INDIA", "ELECTIONCOMMISSION",
            "ELECTORPHOTOIDENTITYCARD", "GOVERNMENT", "GOVT"
        }
        # possible_names = [t for t in high_conf_texts if t.replace(" ", "") not in blacklist]
        # name = possible_names[0] if possible_names else None
        def is_probable_name(text):
            return (
                8 <= len(text) <= 30 and
                text.replace(" ", "").isalpha() and
                not any(word.upper() in blacklist for word in text.split())
            )

        if not name:
            possible_names = [t for t, s in zip(texts, scores) if s > 0.7 and is_probable_name(t)]
            if not possible_names:
                possible_names = [t for t in texts if is_probable_name(t)]
            name = possible_names[0] if possible_names else None

    
        result = {
            "document_type": document_type,
            "name": name,
            "dob": dob,
            "gender": gender,
            "valid": False
        }

        if document_type == "PAN":
            result["pan_number"] = id_number
            result["valid"] = bool(id_number and dob and name)
        elif document_type == "AADHAAR":
            result["aadhaar_number"] = id_number
            result["valid"] = bool(id_number and dob and name and gender)
        elif document_type == "VOTER_ID":
            result["voter_id_number"] = id_number
            result["valid"] = bool(id_number and name)

        # return result
        result["extracted_texts"] = texts
        result["extracted_text_combined"] = combined_text
        return result


    except Exception as e:
        print("Error during OCR:", str(e))
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.post("/validate-aadhar")
async def validate_id_proof(file1: UploadFile = File(..., description="Aadhaar / Pancard Image")):
    try:
        contents = await file1.read()
        id_image = io.BytesIO(contents)

        pil_image = Image.open(id_image).convert("RGB")

        # Enhance for OCR
        gysc_image = ImageOps.grayscale(pil_image)
        autoctrst_image = ImageOps.autocontrast(gysc_image)
        image_np = cv2.cvtColor(np.array(autoctrst_image), cv2.COLOR_GRAY2BGR)

        # OCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        results = ocr.predict(image_np)

        result_dict = results[0]
        texts = result_dict.get("rec_texts", [])
        scores = result_dict.get("rec_scores", [])

        if not texts:
            raise HTTPException(status_code=400, detail="OCR failed: No text detected")


        # ----------- Final Output -----------
        return {
            "document_type": "AADHAAR",
     
            "valid": True,
            "texts": texts,
            "scores":scores
        
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)