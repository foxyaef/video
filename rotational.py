import streamlit as st
import cv2
import numpy as np
import tempfile
import math

st.set_page_config(page_title="회전속도 분석기", layout="centered")
st.title("🌀 2차원 충돌 실험: 회전속도 분석기 (중심 기준 회전 + HSV 슬라이더 설정 + 시각화)")

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

    st.markdown("### 🎨 HSV 색상 범위 설정")
    st.markdown("**중심 스티커 HSV 범위**")
    h1_min = st.slider("H1 최소", 0, 179, 5)
    h1_max = st.slider("H1 최대", 0, 179, 25)
    s1_min = st.slider("S1 최소", 0, 255, 100)
    s1_max = st.slider("S1 최대", 0, 255, 255)
    v1_min = st.slider("V1 최소", 0, 255, 100)
    v1_max = st.slider("V1 최대", 0, 255, 255)

    st.markdown("**회전 마커 HSV 범위**")
    h2_min = st.slider("H2 최소", 0, 179, 40)
    h2_max = st.slider("H2 최대", 0, 179, 80)
    s2_min = st.slider("S2 최소", 0, 255, 100)
    s2_max = st.slider("S2 최대", 0, 255, 255)
    v2_min = st.slider("V2 최소", 0, 255, 100)
    v2_max = st.slider("V2 최대", 0, 255, 255)

    lower_center = np.array([h1_min, s1_min, v1_min])
    upper_center = np.array([h1_max, s1_max, v1_max])
    lower_marker = np.array([h2_min, s2_min, v2_min])
    upper_marker = np.array([h2_max, s2_max, v2_max])

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
            st.error("충분한 마커 추적에 실패했습니다. HSV 범위 또는 프레임 범위를 조정해주세요.")
