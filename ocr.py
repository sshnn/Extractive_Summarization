from PIL import Image
import pytesseract 
from wand.image import Image as Img


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = Image.open("./image_frames/frame0.png")
text = pytesseract.image_to_string(img, lang='eng')
f = open("doc.txt", "a")
f.write(text)
