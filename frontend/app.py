import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

backend = 'http://backend:8000/segmentation'

def process(image, server_url: str):
    message = MultipartEncoder(fields={'file': ('filename', image, 'image/jpeg')})
    req =  requests.post(
        server_url, data=message, headers={'Content-Type': message.content_type}, timeout=8000
    )

    return req


# UI layout
st.title('DeepLabV3 Image Segmentation')

st.write(
    '''Obtain semantic segmentation maps of the image in input via DeepLabV3.
    Website built using streamlit and FastAPI'''
)

input_image = st.file_uploader('Insert Image')

if st.button('Get Segmentation Map'):
    column1, column2 = st.columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert('RGB')
        segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
        column1.header('Original Image')
        column1.image(original_image, use_column_width=True)
        column2.header('Segmentation Map')
        column2.image(segmented_image, use_column_width=True)
    else:
        st.write('Insert an Image')