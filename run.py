import os
from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import tempfile
import cv2
import numpy as np
import altair as alt
import av
import requests

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html


from keras.models import load_model

def send_line_alert(output, msg):
    if output[0][0] < output[0][1]:
        try:
            TARGET_URL = 'https://notify-api.line.me/api/notify'
            TOKEN = 'wFj9X0rw11y1C7skYFOd6lRBrE5vCgF5O2UkGnMnbEs'		
            headers={'Authorization': 'Bearer ' + TOKEN}
            data={'message': msg}
            
            response = requests.post(TARGET_URL, headers=headers, data=data)

            st.write("Message sent !")

        except Exception as ex:
            print(ex)


INPUT_SHAPE=(224,224)

model=load_model("smoking_detector.model")

num_non_smoking=len(os.listdir("./not_smoking/"))
num_smoking=len(os.listdir("./smoking/"))

dataframe = pd.DataFrame.from_dict({"Smoking":['Smoking',num_smoking], "Non-Smoking":['Non-Smoking',num_non_smoking]},\
    columns=['name','num of data'], orient='index')

st.title('Smoking Detection')
preview_img=Image.open("./smoking/ggg599.JPG")
st.image(preview_img)

st.header('1. Train')
st.subheader("1.1. 학습 데이터 구성")
fig1 = px.pie(dataframe, values='num of data', names='name')
st.plotly_chart(fig1)

st.subheader("1.2. 학습 결과")
train_result_img=Image.open("./ploy.jpg")
st.image(train_result_img)

st.header('2. 모델 테스트')
st.subheader("2.1. 이미지 테스트")
uploaded_file = st.file_uploader("Please choose a file", type=['png', 'jpg'], help='Upload png or jpg Image file !')
if uploaded_file is not None:
    uploaded_byte = uploaded_file.read()
    st.image(uploaded_byte)
    
    # image processing
    image = load_img(uploaded_file, target_size=INPUT_SHAPE)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image,axis=0)
    output=model.predict(image)

    result = pd.DataFrame({
        "Probability":[output[0][0], output[0][1]],
        "Label":["Non-smoking", "Smoking"]
    })
    bar_chart = alt.Chart(result).mark_bar().encode(
        y='Probability',
        x='Label'
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # if smoking prob > non-smoking prob
    send_line_alert(output, msg="Smoking detected in Image")

st.subheader("2.2. 웹캠 테스트")
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super()
        self.smoking=0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.resize(img, (224,224))
        image = preprocess_input(image)
        image = np.array(image, dtype='float32')
        image = np.expand_dims(image,axis=0)

        output=model.predict(image)

        st.write("non-smoking :", output[0][0])
        st.write("smoking :", output[0][1])

        # if smoking prob > non-smoking prob
        if output[0][1]>=0.7:
            self.smoking+=1
        else:
            self.smoking=0
        
        if self.smoking>=25:
            send_line_alert(output, msg="Smoking detected in Video")
            self.smoking=0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_ctx = webrtc_streamer(
    key="TEST",
    video_processor_factory=VideoProcessor,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
