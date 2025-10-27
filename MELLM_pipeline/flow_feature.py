import cv2
import numpy as np
import torch
from flow_vis import flow_to_image
import argparse
import cv2
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from roi_combine import get_roi_flow, describe_flow_direction, roi_name_map

face_model_path = "./weights/Alignment_RetinaFace.pth"
face_detector = FaceDetector(model_path=face_model_path, device="cuda")
landmark_model_path = "./weights/Landmark_98.pkl"
landmark_detector = LandmarkDetector(
    model_path=landmark_model_path, device="cuda", device_ids=[0]
)


@torch.no_grad()
def demo_data(save_path, model, image1, image2):
    H, W = image1.shape[2:]
    output = model.calc_flow(image1, image2)
    head_flow = output["head_flow"][-1]
    expression_flow = output["expression_flow"][-1]
    facial_flow = head_flow + expression_flow
    expression_flow = facial_flow - head_flow
    flow_vis_head = flow_to_image(
        head_flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True
    )
    flow_vis_expression = flow_to_image(
        expression_flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True
    )
    flow_vis_facial = flow_to_image(
        facial_flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True
    )
    res = np.concatenate([flow_vis_facial, flow_vis_head, flow_vis_expression], axis=1)
    cv2.imwrite(f"{save_path}/test.jpg", res)
    return expression_flow[0].permute(1, 2, 0).cpu().numpy()


def refine_feature(onset_path):
    onset_img = cv2.imread(onset_path)
    _, dets = face_detector.get_face(onset_path)
    if dets is not None and len(dets) > 0:
        landmarks = landmark_detector.detect_landmarks(onset_img, [dets[0]])[0]
    return landmarks


PROMPT = """Based on the following information, analyze the facial action unit(s) and micro-expression:
{}
Follow these steps for the analysis:
### 1. Analyze ROIs (Regions of Interest) that are significantly related to facial micro-expressions:
* Provide a brief analysis for regions showing clear movement in **magnitude and/or direction**, and infer the associated Action Unit(s).
* For regions without significant movement, **no attention or analysis is required**.
* For regions with noticeable movement but unrelated to micro-expressions, provide only a brief note without further analysis.
### 2. Infer the micro-expression category from the overall movement pattern (descriptive reasoning):
* Based on the combined movement of all significant ROIs (direction and magnitude distribution), provide step-by-step causal/inductive reasoning that explains how local movements converge into a specific micro-expression conclusion.
### 3. **Summarize the AUs and the micro-expression ( Negative | Positive | Surprise ).**
"""


def get_prompt(model, onset_path, offset_path, device):
    image1 = cv2.imread(onset_path)
    image2 = cv2.imread(offset_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)

    flow = demo_data(
        f"/data/zyzhang/flow_code/MELLM/MELLM_pipeline/data/test",
        model,
        image1,
        image2,
    )
    onset_landmark = refine_feature(onset_path)

    nose_landmark = onset_landmark[54]
    x, y = map(int, nose_landmark)
    half_size = 2
    y_min = max(0, y - half_size)
    y_max = min(flow.shape[0], y + half_size + 1)
    x_min = max(0, x - half_size)
    x_max = min(flow.shape[1], x + half_size + 1)
    print(flow.shape)
    roi = flow[y_min:y_max, x_min:x_max, :]
    mean_flow = np.mean(roi, axis=(0, 1))
    flow = flow - mean_flow
    flow_dic, _ = get_roi_flow(flow, onset_landmark)

    feat_info = ""
    for roi, flow in flow_dic.items():
        arr = np.asarray(flow, dtype=np.float64)
        if arr.size == 0:
            continue
        mag = np.linalg.norm(arr.reshape(-1, 2), axis=1)
        dir_des = describe_flow_direction(flow)
        feat_info += f'{{"region":"{roi_name_map[roi]},"mean_magnitude":{round(mag.mean(), 3)},"max_magnitude":{round(mag.max(), 3)},"direction":{dir_des}}}\n'

    prompt_ = PROMPT.format(feat_info)
    return prompt_
