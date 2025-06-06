import streamlit as st
import cv2
import numpy as np
import tempfile
import math

st.set_page_config(page_title="회전속도 분석기", layout="centered")
st.title("🌀 2차원 충돌 실험: 회전속도 분석기 (마커 시각화 + 색상 수동 지정)")

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

    st.markdown("### 🎨 HSV 색상 범위 지정 (마커 색상)")
    h_min = st.slider("H 최소", 0, 179, 25)
    h_max = st.slider("H 최대", 0, 179, 35)
    s_min = st.slider("S 최소", 0, 255, 100)
    s_max = st.slider("S 최대", 0, 255, 255)
    v_min = st.slider("V 최소", 0, 255, 100)
    v_max = st.slider("V 최대", 0, 255, 255)

    visualize = st.checkbox("🖼️ 추적 시각화 보기 (느려질 수 있음)", value=True)

    if st.button("분석 시작"):
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        angles = []
        times = []
        display_frames = []

        frame_idx = 0
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx > end_frame:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    angle = math.atan2(cy, cx)
                    angles.append(angle)
                    times.append(frame_idx / fps)

                    # 시각화
                    if visualize:
                        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)
                        cv2.putText(frame, f"{frame_idx}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        display_frames.append(frame)

            frame_idx += 1

        cap.release()

        if visualize and display_frames:
            st.markdown("### 👁️ 추적된 프레임 시각화")
            for disp in display_frames[::max(len(display_frames)//10,1)]:
                st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB")

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
        else:
            st.error("⚠️ 회전 마커를 충분히 추적하지 못했습니다. 색상 범위 또는 프레임 구간을 조정해보세요.")
