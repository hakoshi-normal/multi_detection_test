import time
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

model = YOLO("best.pt")

counter = 0
start = time.time()
while True:
    try:
        frames = pipeline.wait_for_frames()
    except:
        continue
    counter += 1

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    frame = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    frame_depth= cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
    cv2.putText(frame_depth, "Depth Sensing", (30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_frame)
    annotated_frame = draw_landmarks_on_image(mp_frame.numpy_view(), detection_result)
    cv2.putText(annotated_frame, "Pose Detection", (30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))
    

    if detection_result.segmentation_masks is not None:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        visualized_mask = cv2.morphologyEx(visualized_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        visualized_mask = np.zeros(frame.shape).astype(np.uint8)
    cv2.putText(visualized_mask, "Human Segmentation", (30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

    mp_frame = cv2.hconcat([annotated_frame, visualized_mask])

    results = model(frame, verbose=False)
    yolo_frame = results[0].plot()
    cv2.putText(yolo_frame, "Object Detection", (30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))
    
    yolo_frame = cv2.hconcat([yolo_frame, frame_depth])

    frame = cv2.vconcat([mp_frame, yolo_frame])
    _, w, _ = frame.shape
    cv2.putText(frame, f"FPS:{round(counter/(time.time()-start),1)}", (w-150,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))
    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
