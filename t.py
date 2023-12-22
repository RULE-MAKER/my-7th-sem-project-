import openai
import speech_recognition as sr
import pyttsx3
import cv2
import time
import threading
import paho.mqtt.client as mqtt

# Set your OpenAI GPT-3 API key
openai.api_key = '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'

# Global variable to track whether the system is active
is_active = False

MQTT_BROKER = "91.121.93.94"  # Replace with your MQTT broker IP address
MQTT_PORT = 1883
MQTT_TOPIC = "home/assistant/light"

def listen_for_keyword(keyword="Jarvis"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening for keyword...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"User: {text}")
                if keyword.lower() in text.lower():
                    return True
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

def listen():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Waiting for the keyword to wake up...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"User: {text}")
                if "jarvis" in text.lower():
                    speak("Yes sir")
                    break
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    # Now that the wake-up keyword is detected, listen for the actual user command
    with microphone as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing command...")
        command = recognizer.recognize_google(audio)
        print(f"User command: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def ask_gpt(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=200,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()

def facial_recognition():
    global is_active

    while True:
        ret, img = cam.read()

        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])
            print("ID:", id, "Accuracy:", accuracy)
            print("Number of Faces Detected:", len(faces))

            if accuracy < 60:
                # Display the image only once
                cv2.imshow('camera', img)
                if not is_active:
                    is_active = True
                    speak("Optical Face Recognition Done. Welcome")
                    time.sleep(2)  # Wait for 2 seconds before releasing the camera
                    cam.release()  # Release the camera
                    cv2.destroyAllWindows()
                    threading.Thread(target=jarvis).start()  # Start Jarvis in a new thread
                break
            else:
                speak("Optical Face Recognition Failed")
                break

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global is_active
    message = msg.payload.decode()

    if "turn on light" in message.lower():
        speak("Turning on the light...")
        # Send command to ESP NodeMCU to turn on the light
        # You need to implement the code to send MQTT commands to your ESP
    elif "turn off light" in message.lower():
        speak("Turning off the light...")
        # Send command to ESP NodeMCU to turn off the light
        # You need to implement the code to send MQTT commands to your ESP

def send_mqtt_command(user_command):
    # Send user command to MQTT
    client.publish(MQTT_TOPIC, user_command)

def jarvis():
    global is_active

    while is_active:
        user_input = listen()

        if "exit" in user_input:
            speak("Goodbye! Have a great day.")
            is_active = False
            break

        # Check if the user command is related to turning on or off the light
        if "turn on light" in user_input.lower() or "turn off light" in user_input.lower():
            # Send user command to MQTT without waiting for GPT response
            threading.Thread(target=send_mqtt_command, args=(user_input,)).start()
        else:
            # Get GPT response only for non-light related commands
            response = ask_gpt(user_input)
            print(response)

            spoken_response = ' '.join(response.split()[:25])
            speak(spoken_response)
            time.sleep(1)
            speak("Remaining response is printed")

if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Face recognition/trainer/nagendra1.yml')
    cascadePath = "Face recognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 2

    names = ['', 'nagendra', 'nagu']

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    # Initialize MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message  # Added this line

    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    threading.Thread(target=client.loop_start).start()  # Start MQTT loop in a new thread

    threading.Thread(target=facial_recognition).start()
