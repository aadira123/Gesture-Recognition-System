import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emergency_sign_model.keras")

model = load_model()

classes = ['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief']
IMG_SIZE = 224

st.title("🚨 Emergency Sign Detection")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = np.expand_dims(img_resized.astype(np.float32)/255.0, 0)

        pred = model.predict(img_norm, verbose=0)
        gesture = classes[np.argmax(pred)]
        confidence = np.max(pred) * 100

        cv2.putText(img, f'{gesture}: {confidence:.1f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if gesture in ['help', 'accident'] and confidence > 70:
            cv2.putText(img, 'EMERGENCY ALERT!', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)