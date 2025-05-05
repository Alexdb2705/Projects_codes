import os
import shutil

# Define the source directory and the target directories
source_dir = os.path.join(os.getcwd(),'Img','ISAR')
target_dirs = {
    '1': os.path.join(source_dir, 'ISAR_x'),
    '2': os.path.join(source_dir, 'ISAR_y'),
    '3': os.path.join(source_dir, 'ISAR_z')
}

# Create target directories if they don't exist
for key in target_dirs:
    os.makedirs(target_dirs[key], exist_ok=True)

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    ending_number = filename.split('_')[-1].split('.')[0]
    if ending_number in target_dirs:
        # Move the file to the corresponding target directory
        shutil.move(os.path.join(source_dir, filename), os.path.join(target_dirs[ending_number], filename))

print("Files have been reorganized successfully.")