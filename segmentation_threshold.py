import cv2
import numpy as np
#import imageio
import matplotlib.pyplot as plt
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#from scipy import ndimage
#import imutils


def plot_avg_intensity(avg_intensity_list):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_intensity_list, color='b')
    plt.title('Average Intensity Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Intensity')
    plt.grid(True)
    plt.show()

def split_image_into_two_vetrical(image):
    height, width = image.shape[:2]
    mid_width = width // 2
    left_tile = image[:, :mid_width]
    right_tile = image[:, mid_width:]
    return left_tile, right_tile

def split_image_into_two_horizontal(image):
    height, width = image.shape[:2]
    mid_height = height // 2
    left_tile = image[:mid_height, :]
    right_tile = image[mid_height:, :]
    return left_tile, right_tile


def main():

    video_capture = cv2.VideoCapture('/mnt/data10tb/DATASETS/BACTERIA/Videos/T1J (9h16-9h33).m4v')
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()



    element_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
    element_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))


    frames_with_contours = []
    background_intensity = []
    frame_number = 0  # Counter for frame number

    while True:
        # Capture frame-by-frame
        ret, img = video_capture.read()
        if not ret:
            break

        frame = img.copy()
        height, width, _ = frame.shape

        print(width, height)
        #frame = cv2.resize(frame, (width*2, height*2))

        """shifted = cv2.pyrMeanShiftFiltering(frame, 21, 51)
        cv2.imshow("Input", shifted)"""
        # Create a black image of the same size as video frames
        #black_image = np.zeros((height, width), dtype=np.uint8)
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray', gray_frame)
        dilated = cv2.dilate(gray_frame, element_dilate, iterations=2)
        dilated = cv2.erode(dilated, element_erode, iterations=2)
        #cv2.imshow('dilated', dilated)

        #cv2.waitKey()
        left, right = split_image_into_two_vetrical(dilated)
        l, r = split_image_into_two_vetrical(frame)
        #avg_intensity = np.mean(gray_frame)
        avg_intensity_l = np.mean(left)
        avg_intensity_r = np.mean(right)
        #print(avg_intensity_l, avg_intensity_r)
        #background_intensity.append(avg_intensity)

        #thresh = int(avg_intensity * 1.2)
        thresh_ceof = 1.5
        thresh_l = int(avg_intensity_l * thresh_ceof)
        thresh_r = int(avg_intensity_l * thresh_ceof)

        #_, thresholded_frame = cv2.threshold(dilated, thresh, 255, cv2.THRESH_BINARY) #90
        _, thresholded_frame_l = cv2.threshold(left, thresh_l , 255, cv2.THRESH_BINARY) #90
        _, thresholded_frame_r = cv2.threshold(right, thresh_r , 255, cv2.THRESH_BINARY) #90


        #contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_l, _ = cv2.findContours(thresholded_frame_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_r, _ = cv2.findContours(thresholded_frame_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # noise removal
        kernel = np.ones((5, 5), np.uint8)
        #opening = cv2.morphologyEx(thresholded_frame, cv2.MORPH_OPEN, kernel, iterations=2)
        opening_l = cv2.morphologyEx(thresholded_frame_l, cv2.MORPH_OPEN, kernel, iterations=1)
        opening_r = cv2.morphologyEx(thresholded_frame_r, cv2.MORPH_OPEN, kernel, iterations=1)

        # sure background area
        #sure_bg = cv2.dilate(opening, kernel, iterations=3)
        sure_bg_l = cv2.dilate(opening_l, kernel, iterations=1)
        sure_bg_r = cv2.dilate(opening_r, kernel, iterations=1)

        # Finding sure foreground area
        #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        dist_transform_l = cv2.distanceTransform(opening_l, cv2.DIST_L2, 0)
        dist_transform_r = cv2.distanceTransform(opening_r, cv2.DIST_L2, 0)

        #ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        ret, sure_fg_l = cv2.threshold(dist_transform_l, 0.7 * dist_transform_l.max(), 255, 0)
        ret, sure_fg_r = cv2.threshold(dist_transform_r, 0.7 * dist_transform_r.max(), 255, 0)

        # Finding unknown region
        #sure_fg = np.uint8(sure_fg)
        sure_fg_l = np.uint8(sure_fg_l)
        sure_fg_r = np.uint8(sure_fg_r)

        #unknown = cv2.subtract(sure_bg, sure_fg)
        unknown_l = cv2.subtract(sure_bg_l, sure_fg_l)
        unknown_r = cv2.subtract(sure_bg_r, sure_fg_r)


        # Marker labelling
        #ret, markers = cv2.connectedComponents(sure_fg)
        ret, markers_l = cv2.connectedComponents(sure_fg_l)
        ret, markers_r = cv2.connectedComponents(sure_fg_r)

        # Add one to all labels so that sure background is not 0, but 1
        #markers = markers + 1
        markers_l = markers_l + 1
        markers_r = markers_r + 1

        # Now, mark the region of unknown with zero
        #markers[unknown == 255] = 0
        markers_l[unknown_l == 255] = 0
        markers_r[unknown_r == 255] = 0

        """cv2.imshow('left', left)
        cv2.imshow('right', right)
        cv2.waitKey()
        """
        #markers = cv2.watershed(frame.copy(), markers)
        markers_l = cv2.watershed(l, markers_l)
        markers_r = cv2.watershed(r, markers_r)

        #frame[markers == -1] = [255, 0, 0]
        l[markers_l == -1] = [255, 0, 0]
        r[markers_r == -1] = [255, 0, 0]


        #mask = np.zeros_like(thresholded_frame, dtype=np.uint8) #thresholded_frame_l
        mask_l = np.zeros_like(thresholded_frame_l, dtype=np.uint8) #thresholded_frame_l
        mask_r = np.zeros_like(thresholded_frame_r, dtype=np.uint8) #thresholded_frame_l

        #mask[markers == -1] = 255
        mask_l[markers_l == -1] = 255
        mask_r[markers_r == -1] = 255

        #cv2.imshow('mask', mask)

        # Find contours on the mask
        #watershed_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        watershed_contours_l, _ = cv2.findContours(mask_l, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        watershed_contours_r, _ = cv2.findContours(mask_r, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


        # Exclude contours overlapping with watershed segments
        #filtered_contours = []
        """ for contour in contours:
            contour_area = cv2.contourArea(contour)
            for w_contour in watershed_contours:
                if cv2.contourArea(cv2.convexHull(contour)) > cv2.contourArea(cv2.convexHull(w_contour)):
                    intersection_area = cv2.contourArea(cv2.convexHull(cv2.subtract(w_contour, contour)))
            if contour_area < 10:
    
                break"""

        # Draw bounding boxes around the combined contours
        #combined_contours = watershed_contours + contours

        # Draw contours on the original frame
        frame_with_contours = frame.copy()
        for contour in contours_l:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for contour in contours_r:
            x , y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_contours, (x + width//2, y) , (x + w + width//2, y + h), (0, 255, 0), 2)
        frames_with_contours.append(frame_with_contours)

        for contour in contours_l:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Calculate bounding box coordinates in YOLO format
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            bbox_width = w / width
            bbox_height = h / height
            # Save label in YOLO format
            with open(f'/mnt/data10tb/DATASETS/BACTERIA/auto_labeled/vid9/frame_{frame_number}.txt', 'w') as label_file:
                label_file.write(f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')

        for contour in contours_r:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(frame_with_contours, (x + (width // 2), y), (x + (width // 2) + w, y + h), (0, 255, 0), 2)
            # Calculate bounding box coordinates in YOLO format
            x_center = (x + w / 2 + width / 2) / width
            y_center = (y + h / 2) / height
            bbox_width = w / width
            bbox_height = h / height
            # Save label in YOLO format
            with open(f'/mnt/data10tb/DATASETS/BACTERIA/auto_labeled/vid9/frame_{frame_number}.txt', 'a') as label_file:
                label_file.write(f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')

            # Save frame as image
        cv2.imwrite(f'/mnt/data10tb/DATASETS/BACTERIA/auto_labeled/vid9/frame_{frame_number}.jpg', img)
        frame_number += 1
        cv2.imshow('Frame with Contours Around Segmented Regions', frame_with_contours)

        # Wait for 25 milliseconds
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    #imageio.mimsave('output.gif', frames_with_contours, fps=25)

    #plot_avg_intensity(background_intensity)
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()