import cv2
import numpy as np
from keras.models import load_model
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget

Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '480')


class EmotionRecognition(Image):
    def __init__(self, capture, fps, **kwargs):
        super(EmotionRecognition, self).__init__(**kwargs)
        self.capture = capture
        self.model = load_model('../cnn_model/trained_deep_model.h5')
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.face_cascade = cv2.CascadeClassifier(
            '../data/haarcascade_frontalface_default.xml')
        self.border = 2
        self.image = Image(source='../smiley/happy.png')
        self.label = None
        self.smiley = Image()

    def set_label(self, label):
        self.label = label

    def set_smiley(self, smiley):
        self.smiley = smiley

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                im = cv2.resize(gray[y + self.border:y + h - self.border, x + self.border:x + w - self.border],
                                (48, 48))
                im = np.divide(im, 255.0)
                img = im.reshape((1, 1, 48, 48))
                res = self.model.predict(img)
                res = res.tolist()[0]
                pos = res.index(max(res))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), self.border)

                self.label.text = "[color=ff0000]You are feeling " + self.emotions[pos] + '[/color]'
                self.smiley.source = "../smiley/" + self.emotions[pos] + ".png"

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture


class CamApp(App):
    def build(self):
        parent = FloatLayout(size=(640, 480))

        self.capture = cv2.VideoCapture(0)
        self.label = Label(text='You are feeling', pos_hint={'x': -.35, 'y': .45}, markup=True)
        self.my_camera = EmotionRecognition(capture=self.capture, fps=60.0, resolution=(1366, 768))
        self.my_camera.set_label(self.label)
        parent.add_widget(self.my_camera)
        parent.add_widget(self.label)
        self.smiley = Image(source="../smiley/happy.png", pos_hint={'x': 0, 'y': 0}, size_hint=(0.2, 0.2))
        self.my_camera.set_smiley(self.smiley)
        parent.add_widget(self.smiley)

        return parent

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()
