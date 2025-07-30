# -*- coding: utf-8 -*-

# unzip the best.pt file
!unzip best.pt

#Load your trained YOLO model
model = YOLO("best.pt")  # Upload best.pt to Colab first

# Generate a beep sound (saved as beep.wav)
def generate_beep(filename="beep.wav", freq=600, duration=0.4, rate=44100):
    t = np.linspace(0, duration, int(rate * duration))
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    write(filename, rate, data.astype(np.float32))

generate_beep()

# Detection logic
def detect_dents(image):
    results = model(image)
    img = results[0].orig_img.copy()

    dent_detected = False

    for box in results[0].boxes:
        dent_detected = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls]}: {conf:.2f}"

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if dent_detected:
        display(Audio("beep.wav", autoplay=True))
    else:
        # If no dent, show "PASS" message on image
        cv2.putText(img, "PASS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Launch Gradio UI
gr.Interface(
    fn=detect_dents,
    inputs=gr.Image(type="numpy", label="Upload Car Image"),
    outputs=gr.Image(label="Result"),
    title="🚗 Car Dent Detector",
    description="Get a beep and bounding box if dent is detected. If no dent, see 'PASS' message."
).launch(share=True)



