import cv2
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.DEBUG)


class CellCounterApp:
    def __init__(self):
        st.set_page_config(page_title='Cell Counter App', layout='wide', initial_sidebar_state='auto')
        self.default_image_path = "../data/images/cell-1.jpg"

    def run(self):
        # streamlit title
        st.title("Cell Counter Example")


        tab_raw, tab_blur, tab_thresh, tab_result = st.tabs(['Raw', 'Blur', 'Threshold', 'Result'])
        with tab_raw:
            # load raw image
            img_raw = cv2.imread(self.default_image_path)
            # show raw image
            fig = px.imshow(img_raw, color_continuous_scale='gray')
            st.plotly_chart(fig)
            # documentation
            st.code(
                """
                # load raw image
                img_raw = cv2.imread(self.default_image_path)
                """,
                language='python'
            )
        with tab_blur:
            # add slider
            blur_size = st.slider(
                label="Blur Kernel Size",
                min_value=1,
                max_value=10,
                value=5
            )
            # convert to greyscale
            img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.blur(img_grey, (blur_size, blur_size))
            # show blurred image
            fig = px.imshow(img_grey, color_continuous_scale='gray')
            st.plotly_chart(fig)
            # documentation
            st.code(
                """
                # convert to greyscale
                img_grey = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.blur(img_grey, (blur_size, blur_size))
                """,
                language='python'
            )
        with tab_thresh:
            # add slider
            thresh = st.slider(
                label="Threshold image to isolate nuclei",
                min_value=0,
                max_value=255,
                value=75
            )
            # histogram
            bins = list(range(0, 256))
            counts, _ = np.histogram(img_blur.ravel(), bins=bins)
            fig = go.Figure(go.Bar(x=bins, y=counts))
            fig.add_vline(x=thresh, line_color='red')
            st.plotly_chart(fig)
            # threshold
            ret, img_thresh = cv2.threshold(img_blur, thresh, 255, cv2.THRESH_BINARY)
            fig = px.imshow(img_thresh, color_continuous_scale='gray')
            st.plotly_chart(fig)
            # documentation
            st.code(
                """
                # threshold
                ret, img_thresh = cv2.threshold(img_blur, thresh, 255, cv2.THRESH_BINARY)
                """,
                language='python'
            )
        with tab_result:
            # find contours
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # count contours
            total_cells = len(contours)
            img_cnts = img_raw.copy()  # drawContours changes source image
            cv2.drawContours(img_cnts, contours, -1, (0, 255, 0), 3)

            fig = px.imshow(img_cnts, color_continuous_scale='gray')
            st.plotly_chart(fig)

            st.write(f'Total cells counted: {total_cells}')
            st.code(
                """
                # find contours
                contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # count contours
                total_cells = len(contours)
                """,
                language='python'
            )

if __name__ == "__main__":
    app = CellCounterApp()
    app.run()
