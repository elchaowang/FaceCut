import cv2
import dlib
import os


detector = dlib.get_frontal_face_detector()
output_path = './Faces/'


def get_faces(img_name):
    or_img = cv2.imread(img_name)
    faces = detector(or_img, 1)
    return faces, or_img


def get_im_name_list(lib_name):
    imnames = os.listdir(lib_name)
    print('\n', imnames.__len__(), 'images were found in the lib')
    return imnames


def output2Fold(faces, img):
    cur_index = os.listdir(output_path).__len__()
    for num, d in enumerate(faces):
        output_img = img[d.top():d.bottom(), d.left():d.right()]
        print('Saved to:', output_path + 'face_' + str(num + cur_index) + '.jpg')
        cv2.imwrite(output_path + 'face_' + str(num + cur_index) + '.jpg', output_img)


def main():
    lib_name = './IMG/'
    imgs = get_im_name_list(lib_name)
    for num, img in enumerate(imgs):
        faces, or_img = get_faces(lib_name + img)
        output2Fold(faces, or_img)


if __name__ == '__main__':
    main()



















