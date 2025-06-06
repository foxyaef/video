import streamlit as st
import cv2
import numpy as np
import tempfile
import math
from PIL import Image
from pathlib import Path
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="회전속도 분석기", layout="centered")
st.title("🌀 2차원 충돌 실험: 회전속도 분석기 (중심 기준 회전 + HSV 자동 설정 + 시각화)")

video_file = st.file_uploader("🎥 충돌 실험 영상을 업로드하세요", type=["mp4", "avi", "mov"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.info(f"총 프레임: {total_frames} / FPS: {fps:.2f}")

    start_frame, end_frame = st.slider("🎬 분석 구간 지정 (충돌 후)", 0, total_frames - 1, (20, min(80, total_frames - 1)))
    mass = st.number_input("🔢 퍽의 질량 m (kg)", min_value=0.01, value=0.20, step=0.01)
    radius = st.number_input("🔢 퍽의 반지름 R (m)", min_value=0.01, value=0.05, step=0.01)

    # 특정 프레임 이미지 가져오기
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, ref_frame = cap.read()
    cap.release()

    if not ret:
        st.error("프레임을 불러올 수 없습니다.")
    else:
        st.markdown("### 📌 중심 스티커와 회전 마커 클릭")
        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(ref_rgb).save(tmp_file.name)

        coords = streamlit_image_coordinates("이미지에서 중심과 마커를 클릭하세요", tmp_file.name, key="click")

        if coords:
            x, y = int(coords["x"]), int(coords["y"])
            hsv_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2HSV)
            clicked_hsv = hsv_frame[y, x]
            st.write(f"🟡 클릭한 픽셀의 HSV: {clicked_hsv}")

            if "center_hsv" not in st.session_state:
                if st.button("이 값을 중심 스티커 HSV로 사용"):
                    st.session_state.center_hsv = clicked_hsv
            elif "marker_hsv" not in st.session_state:
                if st.button("이 값을 마커 HSV로 사용"):
                    st.session_state.marker_hsv = clicked_hsv

        if "center_hsv" in st.session_state and "marker_hsv" in st.session_state:
            def hsv_range(hsv_val, delta=20):
                h, s, v = hsv_val
                lower = np.array([max(h - delta, 0), max(s - delta, 0), max(v - delta, 0)])
                upper = np.array([min(h + delta, 179), min(s + delta, 255), min(v + delta, 255)])
                return lower, upper

            lower_center, upper_center = hsv_range(st.session_state.center_hsv)
            lower_marker, upper_marker = hsv_range(st.session_state.marker_hsv)

            st.success("중심/마커 HSV 범위 설정 완료. 분석 시작 가능!")

            if st.button("회전 분석 시작"):
                angles = []
                times = []
                display_frames = []
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_idx > end_frame:
                        break
                    if frame_idx < start_frame:
                        frame_idx += 1
                        continue

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    center_mask = cv2.inRange(hsv, lower_center, upper_center)
                    marker_mask = cv2.inRange(hsv, lower_marker, upper_marker)

                    contours_c, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_m, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours_c and contours_m:
                        c = max(contours_c, key=cv2.contourArea)
                        m = max(contours_m, key=cv2.contourArea)
                        Mc = cv2.moments(c)
                        Mm = cv2.moments(m)

                        if Mc["m00"] > 0 and Mm["m00"] > 0:
                            cx = int(Mc["m10"] / Mc["m00"])
                            cy = int(Mc["m01"] / Mc["m00"])
                            mx = int(Mm["m10"] / Mm["m00"])
                            my = int(Mm["m01"] / Mm["m00"])

                            angle = math.atan2(my - cy, mx - cx)
                            angles.append(angle)
                            times.append(frame_idx / fps)

                            # 시각화용 표시 추가
                            vis = frame.copy()
                            cv2.circle(vis, (cx, cy), 8, (255, 0, 0), 2)
                            cv2.circle(vis, (mx, my), 8, (0, 255, 0), 2)
                            cv2.line(vis, (cx, cy), (mx, my), (0, 255, 255), 2)
                            display_frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                    frame_idx += 1

                cap.release()

                if len(angles) >= 2:
                    angles = np.unwrap(angles)
                    delta_theta = angles[-1] - angles[0]
                    delta_t = times[-1] - times[0]
                    omega = delta_theta / delta_t
                    I = 0.5 * mass * (radius ** 2)
                    E_rot = 0.5 * I * (omega ** 2)

                    st.success(f"📐 평균 각속도 ω ≈ {omega:.3f} rad/s")
                    st.success(f"⚡ 회전 운동 에너지 ≈ {E_rot:.4f} J")

                    with st.expander("📊 세부 계산 보기"):
                        st.write(f"Δθ = {delta_theta:.4f} rad")
                        st.write(f"Δt = {delta_t:.4f} sec")
                        st.write(f"I = {I:.6f} kg·m²")

                    st.markdown("### 👁️ 마커 시각화 결과")
                    for vis_frame in display_frames[::max(1, len(display_frames)//10)]:
                        st.image(vis_frame, use_column_width=True)
                else:
                    st.error("충분한 마커 추적에 실패했습니다. 클릭한 색상 HSV 범위 또는 프레임 범위를 조정해주세요.")

        else:
            st.info("중심과 마커를 각각 클릭해서 HSV 값을 설정해주세요.")
