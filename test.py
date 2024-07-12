import time
import msvcrt
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
from multiprocessing import Process, Event, shared_memory


yolo_process_N = 7
mp_process_N = 5

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


def finish_monitor(stop_flag):
    while True:
        if msvcrt.kbhit() and msvcrt.getch() == b'q':
            print("Finished")
            stop_flag.set()
            break
    return

def view_frame(pre_fin_flag, stop_flag, output_frame_info):
    counter = 0
    start = time.time()
    while True:
        if pre_fin_flag.is_set():
            pre_fin_flag.clear()
            counter += 1
            shm = shared_memory.SharedMemory(name='result')
            frame = np.ndarray(shape=output_frame_info['shape'], dtype=output_frame_info['dtype'], buffer=shm.buf)
            _, w, _ = output_frame_info['shape']
            cv2.putText(frame, f"FPS:{round(counter/(time.time()-start),1)}", (w-150,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

            cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
    return

def yolo_predict_frame(yolo_pre_flag, pre_fin_flag, yolo_mem_name, stop_flag, frame_info, output_frame_info):
    model = YOLO("best.pt")
    while True:
        if yolo_pre_flag.is_set():
            yolo_pre_flag.clear()
            shm = shared_memory.SharedMemory(name=yolo_mem_name)
            frame = np.ndarray(shape=frame_info["shape"], dtype=frame_info["dtype"], buffer=shm.buf)
            results = model(frame, verbose=False)
            frame = results[0].plot()
            cv2.putText(frame, "Object Detection", (30,30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))
            shm = shared_memory.SharedMemory(name='result')
            frame_result = np.ndarray(shape=output_frame_info['shape'], dtype=output_frame_info['dtype'], buffer=shm.buf)
            h, w, _ = frame.shape
            frame_result[h:,:w]=frame[:]
            pre_fin_flag.set()
        if stop_flag.is_set():
            break
    return

def mp_predict_frame(mp_pre_flag, pre_fin_flag, mp_mem_name, stop_flag, frame_info, output_frame_info):
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    while True:
        if mp_pre_flag.is_set():
            mp_pre_flag.clear()
            shm = shared_memory.SharedMemory(name=mp_mem_name)
            frame = np.ndarray(shape=frame_info["shape"], dtype=frame_info["dtype"], buffer=shm.buf)

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

            frame = cv2.hconcat([annotated_frame, visualized_mask])

            shm = shared_memory.SharedMemory(name='result')
            frame_result = np.ndarray(shape=output_frame_info['shape'], dtype=output_frame_info['dtype'], buffer=shm.buf)
            h, w, _ = frame.shape
            frame_result[:h,:]=frame[:]
            pre_fin_flag.set()
        if stop_flag.is_set():
            break
    return

def rec_cam(yolo_process_N, yolo_pre_flags, mp_process_N, mp_pre_flags, stop_flag, frame_info, output_frame_info):
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

    counter = 0
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

        shm = shared_memory.SharedMemory(name='result')
        frame_result = np.ndarray(shape=output_frame_info['shape'], dtype=output_frame_info['dtype'], buffer=shm.buf)
        h, w, _ = frame_depth.shape
        frame_result[h:,w:]=frame_depth[:]

        yolo_mem_name = f'yolomemory{counter%yolo_process_N}'
        shm = shared_memory.SharedMemory(name=yolo_mem_name)
        mem_frame = np.ndarray(shape=frame_info["shape"], dtype=frame_info["dtype"], buffer=shm.buf)
        mem_frame[:] = frame[:]
        yolo_pre_flags[counter%yolo_process_N].set()

        mp_mem_name = f'mpmemory{counter%mp_process_N}'
        shm = shared_memory.SharedMemory(name=mp_mem_name)
        mem_frame = np.ndarray(shape=frame_info["shape"], dtype=frame_info["dtype"], buffer=shm.buf)
        mem_frame[:] = frame[:]
        mp_pre_flags[counter%mp_process_N].set()

        if stop_flag.is_set():
            break


if __name__ == '__main__':
    print("starting up......")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if frame is not None:
            break
    cap.release()

    frame_info = {'nbytes' : frame.nbytes,
                  'shape' : frame.shape,
                  'dtype' : frame.dtype}

    output_frame = cv2.hconcat([frame, frame])
    output_frame = cv2.vconcat([output_frame, output_frame])
    output_frame_info = {'nbytes' : output_frame.nbytes,
                        'shape' : output_frame.shape,
                        'dtype' : output_frame.dtype}

    processes = []
    mem_space = []
    mem_names = []

    # 表示用フレームメモリ
    shm = shared_memory.SharedMemory(create=True, size=output_frame_info['nbytes'], name='result')
    mem_space.append(shm)
    mem_names.append("result")

    # 停止プロセス
    stop_flag = Event()
    p = Process(target=finish_monitor, args=(stop_flag,))
    processes.append(p)

    # 表示プロセス
    pre_fin_flag = Event()
    p = Process(target=view_frame, args=(pre_fin_flag, stop_flag, output_frame_info))
    processes.append(p)

    # 推論プロセス
    yolo_processes = []
    yolo_pre_flags = []
    for i in range(yolo_process_N):
        yolo_pre_flag = Event()
        yolo_pre_flags.append(yolo_pre_flag)
        yolo_mem_name = f'yolomemory{i}'
        mem_names.append(yolo_mem_name)
        shm = shared_memory.SharedMemory(create=True, size=frame_info['nbytes'], name=yolo_mem_name)
        mem_space.append(shm)
        p = Process(target=yolo_predict_frame, args=(yolo_pre_flag, pre_fin_flag, yolo_mem_name, stop_flag, frame_info, output_frame_info))
        processes.append(p)

    mp_processes = []
    mp_pre_flags = []
    for i in range(mp_process_N):
        mp_pre_flag = Event()
        mp_pre_flags.append(mp_pre_flag)
        mp_mem_name = f'mpmemory{i}'
        mem_names.append(mp_mem_name)
        shm = shared_memory.SharedMemory(create=True, size=frame_info['nbytes'], name=mp_mem_name)
        mem_space.append(shm)
        p = Process(target=mp_predict_frame, args=(mp_pre_flag, pre_fin_flag, mp_mem_name, stop_flag, frame_info, output_frame_info))
        processes.append(p)
    
    # 撮影プロセス
    p = Process(target=rec_cam, args=(yolo_process_N, yolo_pre_flags, mp_process_N, mp_pre_flags, stop_flag, frame_info, output_frame_info))
    processes.append(p)

    for process in processes:
        process.start()

    for p in processes:
        p.join()
    for mem_name in mem_names:
        shm = shared_memory.SharedMemory(name=mem_name)
        shm.close()
        shm.unlink()
