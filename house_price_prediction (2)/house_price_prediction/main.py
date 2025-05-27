import os
import subprocess

if not os.path.exists('model'):
    os.makedirs('model')

print("Training the model...")
subprocess.run(["python", "src/train.py"])
print("Testing the model...")
subprocess.run(["python", "src/test.py"])
print("Model training and testing complete!")
