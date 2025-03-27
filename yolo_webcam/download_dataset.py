from roboflow import Roboflow

rf = Roboflow(api_key="9kQIvRSyxM4t7TlyILdr")  # Replace with your API key if different
project = rf.workspace("jarivs").project("ather-hy9he")
version = project.version(3)
dataset = version.download("yolov8")

print("Dataset downloaded successfully!")

