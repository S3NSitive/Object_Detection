import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        fig, ax = plt.subplots(1)
        # print(len(landmarks))
        # print(landmarks)
        # print(landmarks[0][:])
        ax.imshow(image)
        for i in range(len(landmarks)):
            rect = patches.Rectangle((landmarks[i][0], landmarks[i][1]), landmarks[i][2], landmarks[i][3], linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
        # plt.show()
        plt.pause(0.001)  # pause a bit so that plots are updated
