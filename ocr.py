import pytesseract
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
tessdata_dir = os.path.join(project_dir, 'tessdata')
os.environ['TESSDATA_PREFIX'] = tessdata_dir


class ImageExtractor:
    def __init__(self, custom_config = r'--oem 1 --psm 3 -l ktp+nik'):
        self._custom_config = custom_config

    def __call__(self, image):
      filtered = []
      d = pytesseract.image_to_string(image, config=self._custom_config)
      clear_data = d.split('\n')
      
      for item in clear_data:
          if item.isspace() or len(item) <= 2:
              continue
          else:
              filtered.append(item)

      return filtered
        