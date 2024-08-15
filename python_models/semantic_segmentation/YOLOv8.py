from ultralytics import YOLO

model = YOLO('runs/segment/train3/weights/best.pt')
# model.train(data='data-simple.yaml', epochs=200)
# print(model.val())
model('images', save=True)
