import openai
import speech_recognition as sr
import pyttsx3
import cv2
import time




# Set your OpenAI GPT-3 API key
openai.api_key = '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'

def listen():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"User: {text}")
        return text
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

def jarvis():
    speak("Hello! I'm Jarvis. How can I assist you today?")

    while True:
        user_input = listen().lower()

        if "exit" in user_input:
            speak("Goodbye! Have a great day.")
            exit
            break


        response = ask_gpt(user_input)
        print(response)
        
        # Speak only the first 15 words or less
        spoken_response = ' '.join(response.split()[:25])
        speak(spoken_response)
        time.sleep(1)
        speak("Remaining response is printed")



if __name__ == '__main__':
   

    
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
    recognizer.read('Face recognition/trainer/nagendra1.yml')  # load trained model
    cascadePath = "Face recognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing Haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type

    id = 2  # number of persons you want to Recognize

    names = ['', 'nagendra','nagu']  # names, leave first empty because the counter starts from 0
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
    cam.set(3, 640)  # set video FrameWidth
    cam.set(4, 480)  # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        

        ret, img = cam.read()  # read the frames using the above-created object

        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image from one color space to another

        faces = faceCascade.detectMultiScale(
            converted_image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image
            print("ID:", id, "Accuracy:", accuracy)
            print("Number of Faces Detected:", len(faces))
            
           

            
            # Check if accuracy is less than 100 ==> "0" is a perfect match 
            if accuracy < 60:
                cv2.imshow('camera', img)
                time.sleep(2)
                
                
                # Do a bit of cleanup
                speak("Optical Face Recognition Done. Welcome")
                
                cv2.waitKey(1)  # You can adjust the delay (e.g., cv2.waitKey(50)) if needed
                cam.release()
                cv2.destroyAllWindows()
                jarvis()
            else:
                speak("Optical Face Recognition Failed")
                break
