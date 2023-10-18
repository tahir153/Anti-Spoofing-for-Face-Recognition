from ultralytics import YOLO

model = YOLO('l_version_1_300.pt')

def main():
    model.train(data='Dataset/SplitData/data.yaml', epochs=3)

    if __name__ == '__main__':
        main()
