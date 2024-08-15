import os
import cv2

from mmengine.runner import Runner
from mmengine.config import Config
from mmaction.apis import inference_recognizer, init_recognizer

labels = ['Bunny Hop', 'Superman']

config = Config.fromfile('datasets/Tricks/config.py')

runner = Runner.from_cfg(config)

runner.train()

recognizer = init_recognizer(config,
                             'work_dirs/tricks/epoch_1577.pth',
                             device='cpu')

for (dir_name, _, file_names) in os.walk('datasets/Tricks/test'):
    for file_name in file_names:
        if file_name.endswith('.mp4'):
            inference_res = inference_recognizer(recognizer, dir_name + '/' + file_name)
            cap = cv2.VideoCapture(dir_name + '/' + file_name)
            out = cv2.VideoWriter('predicted/' + file_name,
                                  cv2.VideoWriter_fourcc(*'avc1'),
                                  30.0,
                                  (int(cap.get(3)), int(cap.get(4))))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame,
                            labels[inference_res.pred_label.item()],
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2)
                cv2.putText(frame,
                            f'pred_score = {inference_res.pred_score[inference_res.pred_label.item()].item():.2f}',
                            (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2)
                out.write(frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()
