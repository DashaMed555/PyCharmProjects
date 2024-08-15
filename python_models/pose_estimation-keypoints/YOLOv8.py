import os

import cv2
from ultralytics import YOLO

custom_skeleton = [[[0, 1], [1, 2], [2, 3]],
                   [[3, 4], [4, 21]],
                   [[21, 22], [22, 23]],
                   [[3, 13], [13, 14], [14, 15], [15, 16]],
                   [[3, 17], [17, 18], [18, 19], [19, 20]],
                   [[4, 5], [5, 6], [6, 7], [7, 8]],
                   [[4, 9], [9, 10], [10, 11], [11, 12]]]

colors_for_skeleton = [(255, 0, 139),
                       (184, 184, 114),
                       (84, 53, 233),
                       (0, 165, 255),
                       (0, 165, 255),
                       (0, 165, 255),
                       (0, 165, 255)]


def draw_model(res):
    for res_img in res:
        image = cv2.imread(res_img.save_dir + '/' + os.path.basename(res_img.path))
        for obj_num in range(res_img.boxes.shape[0]):
            for (edge_set, color) in zip(custom_skeleton, colors_for_skeleton):
                for (start, end) in edge_set:
                    cv2.line(image,
                             (int(res_img.keypoints.xy[obj_num, start, 0].item()),
                              int(res_img.keypoints.xy[obj_num, start, 1].item())),
                             (int(res_img.keypoints.xy[obj_num, end, 0].item()),
                              int(res_img.keypoints.xy[obj_num, end, 1].item())),
                             color)
        cv2.imwrite(res_img.save_dir + '/' + os.path.basename(res_img.path), image)


def main():
    model = YOLO('runs/pose/train_with_smaller_lr/weights/best.pt')
    # model.train(data='data.yaml', lr0=0.0001, lrf=0.0001, epochs=1000, patience=0, batch=-1, cache=True, workers=8, name='train_with_smaller_lr', plots=True)
    model.val(data='data.yaml')
    # res = model('images', save=True)
    #draw_model(res)


main()
