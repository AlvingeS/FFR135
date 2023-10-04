import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import cv2

# Step 1: Relative paths
csv_folder = "./boundaries"
plot_folder = "./plots"

# Get all CSV files
csv_files = glob.glob(f"{csv_folder}/*.csv")
csv_files.sort()

# Loop through CSV files, read them, plot, and save the plot
for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    plt.scatter(df[df['Label'] == -1]['X'], df[df['Label'] == -1]['Y'], color='red')
    plt.scatter(df[df['Label'] == 1]['X'], df[df['Label'] == 1]['Y'], color='green')
    plt.savefig(f"{plot_folder}/plot_{i}.png")
    plt.close()

# Step 2: Video Compilation

# Get all the plot images
plot_files = glob.glob(f"{plot_folder}/*.png")
plot_files.sort()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 24.0, (640, 480))

# Loop through each plot image to write to video
for plot_file in plot_files:
    frame = cv2.imread(plot_file)
    out.write(frame)

# Release video writer
out.release()
