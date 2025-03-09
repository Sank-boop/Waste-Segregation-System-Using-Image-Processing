# Waste-Segregation-System-Using-Image-Processing and AI

Waste Segregation System using Image Processing and AI; OpenCV This project utilizes a Keras deep learning model with OpenCV to detect and classify waste types in real-time using a webcam. It also integrates with an ESP32 via serial communication to send classification results. Perfect for smart waste management applications!

This project is a **Waste Detection System** that uses a **Keras deep learning model** and a webcam to classify waste types in real time. The system can also send classification results to an **ESP32 microcontroller** via a serial connection.

---

## Features

- **Real-time Waste Detection**: Uses OpenCV to capture webcam images and classify waste.
- **Deep Learning Model**: Utilizes a trained Keras model (`keras_Model.h5`) for waste classification.
- **ESP32 Communication**: Sends classification results to an ESP32 microcontroller via serial communication.
- **Confidence Score Display**: Shows the prediction confidence for better accuracy analysis.

---

## Requirements

### Software:
- Python 3.x
- TensorFlow & Keras
- OpenCV (`opencv-python`)
- NumPy
- PySerial (if using ESP32 communication)

### Hardware:
- Webcam (for image capture)
- ESP32 (for serial communication, if needed)

---

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Pawan-Makhare/Waste-Detection-System.git
   cd Waste-Detection-System
   ```

2. **Install Dependencies**:
   ```sh
   pip install tensorflow opencv-python numpy pyserial
   ```

3. **Ensure the model and labels are present**:
   - Place the `keras_Model.h5` file in the project directory.
   - Ensure `labels.txt` contains the class names.

---

## How to Run

### Without ESP32 Communication
Run the waste detection system using:
```sh
python waste_detection.py
```

### With ESP32 Communication
Ensure ESP32 is connected and update the correct serial port (`COM4`, `/dev/ttyUSB0`, etc.) in the script. Then, run:
```sh
python waste_detection_esp32.py
```

---

## Usage

1. The webcam captures images and resizes them to `224x224` pixels.
2. The model predicts the waste type and displays the confidence score.
3. If ESP32 is connected, it receives the classification result.
4. Press `Esc` to exit the application.

---

## Example Output
```
Class: Plastic, Confidence: 95%
Class: Organic, Confidence: 88%
Sent to ESP32: Class: Paper, Confidence: 92%
```

---

## Future Enhancements
- Improve model accuracy with more training data.
- Integrate automatic waste sorting using a robotic system.
- Enhance GUI for better user experience.

---

## License
This project is open-source and available under the MIT License. Feel free to modify and distribute it as needed.

---

## Contact
For questions or feedback, feel free to reach out to [Your Email/Contact Info].

---

**Enjoy using the Waste Detection System! ♻️🚀**
