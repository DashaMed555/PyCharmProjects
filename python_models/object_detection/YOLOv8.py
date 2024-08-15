from ultralytics import YOLO, settings

settings.reset()
settings['datasets_dir'] = 'datasets'
print(settings)

model = YOLO('yolov8n.pt')
# print(model)
model.train(data='data.yaml', epochs=1)
# print(model.val())
# model('images/Park-Street-and-Tapawingo-stop-sign-1536x1152.jpg', save=True)
# model.export(format='onnx')
