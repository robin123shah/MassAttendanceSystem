# the idea of this code is to verify is the subject present in the query image is
# present in the session image.
#
# D. Mery, UC, November, 2018
# http://dmery.ing.puc.cl
import os
import numpy as np
from utils import Facer
from utils import fr_str, num2fixstr, dirfiles, extract_rows, vector_distances, im_crop
from utils import im_concatenate, imshow1
import pandas as pd
from cv2 import VideoCapture,imwrite,waitKey,destroyAllWindows, imshow, CAP_PROP_FRAME_HEIGHT,CAP_PROP_FRAME_WIDTH,CAP_PROP_FPS
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import csv
import imutils
import requests 
url = "http://100.123.60.168:8080/video"

AddPerson = input("Add new Person(Y/N) :")
while AddPerson == 'Y':
    K = len(next(os.walk('facer-classroom/data/enroll/'))[1])
    Ks = num2fixstr(K+1,6)
    os.makedirs("facer-classroom/data/enroll/{}/".format(Ks))
    firstname,lastname = input("Full Name: ").split(" ")

    aList = [K+1,firstname,lastname]
    with open('facer-classroom/student_list.csv', 'a', newline='') as f:
    # Create a CSV writer object
        writer = csv.writer(f)
        # Write the new row to the CSV file
        writer.writerow(aList)
    camMod = input("Use Phone cam as input(Y/N): ")
    if camMod == 'Y':
        cam = VideoCapture(url)
        cam.set(CAP_PROP_FRAME_WIDTH, 320)
        cam.set(CAP_PROP_FRAME_HEIGHT, 180)
        cam.set(CAP_PROP_FPS, 25)
    else:
        cam = VideoCapture(0)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        imshow("test", frame)

        k = waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "{}_{}_{}.png".format(firstname,lastname,img_counter)
            imwrite("facer-classroom/data/enroll/{}/".format(Ks)  + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    destroyAllWindows()

    
    AddPerson = input("Add new Person(Y/N) :")



make_Sessions = input("Make new Session(Y/N) :")
while make_Sessions == 'Y':
    K = len(next(os.walk('facer-classroom/data/sessions/'))[1])
    os.makedirs("facer-classroom/data/sessions/LN0{}/".format(K+1))
    # Take_Pic = input("Do you want to take pic or Pick a file (1/2) :")
    # if Take_Pic == "1":
    camMod = input("Use Phone cam as input(Y/N): ")
    if camMod == 'Y':
        cam = VideoCapture(url)
        cam.set(CAP_PROP_FRAME_WIDTH, 320)
        cam.set(CAP_PROP_FRAME_HEIGHT, 180)
        cam.set(CAP_PROP_FPS, 25)
    else:
        cam = VideoCapture(0)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        imshow("test", frame)

        k = waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            imwrite("facer-classroom/data/sessions/LN0{}/".format(K+1)  + img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    destroyAllWindows()
    make_Sessions = input("Make new Session(Y/N) :")
    # else:
        
    #     root = Tk()
    #     root.title("File Dialog box")

    #     # Return the name and location of the file.
    #     root.filename = filedialog.askopenfile(initialdir="/Pictures", title="select a file", filetypes=(("png files"),("all file", "*.*")))

    #     # Display dir of file selected
    #     my_lbl = Label(root, text=root.filename).pack()

    #     # Display image
    #     my_img = Image.open(root.filename)
    #     imwrite("facer-classroom/data/sessions/LN0{}/".format(K+1)  + root.filename, my_img)

new_p, newsession = input("set train face, train session:(0/1 0/1) ").split(" ")
new_p, newsession  = int(new_p),int(newsession)
# definitions
img_path_sessions = 'facer-classroom/data/sessions/'
id_subjects       = [i+1 for i in range(0,len(next(os.walk('facer-classroom/data/enroll/'))[1]))]
id_sessions       = [i+1 for i in range(0,len(next(os.walk('facer-classroom/data/sessions/'))[1]))]    # id of the sessions
csv_list          = 'facer-classroom/student_list.csv'  # ID and names of the students
img_path_enroll   = 'facer-classroom/data/enroll/'
fd_method         = 0               # face detection (0:HOG, 1: CNN)
fr_method         = 2               # face recognition (0: Dlib, 1: Dlib+, 2: FaceNet)
sc_method         = 0               # 0 cosine similarity, 1 euclidean distance
uninorm           = 1               # 1 means descriptor has norm = 1
theta             = 0.45            # threshold for the recognition
print_scr         = 0               # print scores
show_img          = 1             # show images
extract_desc      = [new_p,newsession]          # enrollment, session
img_size          = [80, 60]        # output
echo              = 1               # print out progress comments
session_prefix    = 'LN'            # prefix of the folder that contains the session images
show_fd           = 1           # show face detection


# init
F = Facer()
F.echo           = echo
F.printComment("----------------------------------------------------------------")
F.fd_method      = fd_method
F.fr_method      = fr_method
F.sc_method      = sc_method
F.uninorm        = uninorm
F.theta          = theta
F.show_fd        = show_fd
F.printDefinitions()

# load deep learning model (if any) and define parameters
F.printComment("loading face recognition model " + fr_str[fr_method] + "...")
F.loadModel()
n                  = len(id_subjects) # number of subjects
m                  = len(id_sessions) # number of sessions
image_final        = []
image_session      = []
img_size_f         = [img_size[0]*len(id_subjects),img_size[1]]
F.scores           = np.zeros((n,m))
F.session_prefix   = session_prefix
F.id_subjects      = id_subjects
F.id_sessions      = id_sessions
F.csv_list         = csv_list

# in case the descriptors have not already extracted and saved
# --- extract and save descriptors for enrolled subjects
if extract_desc[0] == 1:
    F.img_path       = img_path_enroll
    F.extractDescriptorsEnrollment()
# --- extract and save descriptors for faces in session images
if extract_desc[1] == 1:
    F.img_path       = img_path_sessions
    F.fd_method      = fd_method
    F.extractDescriptorsSession()

# store in D all descriptors of enrolled subjects
F.extract_desc = 0
F.save_desc    = 0
F.img_path     = img_path_enroll
F.getDescriptorsEnrollment()
D              = F.descriptorsE
ix             = F.ixE # indices

# for all sessions
F.full            = 1
F.fd_method       = fd_method
list2 = []
for j in range(m):
    F.printComment(">>> session "+str(j)+"/"+str(m)+"...")
    img_path_session  = img_path_sessions + F.session_prefix + num2fixstr(id_sessions[j],2) + '/'
    img_names_session = dirfiles(img_path_session,'*.png')
    F.img_path        = img_path_session
    F.img_names       = img_names_session
    F.getDescriptorsImageList()
    Y                 = F.descriptors
    iy                = F.ix
    facesy            = F.bbox

    # for all enrolled images
    list1 = []
    for i in range(n):
        X = extract_rows(D,ix,i) # descriptors of subject i
        # computation of scores between enrolled faces and session images
        scr,scr_best,ind_best,face_detected = vector_distances(Y,X.T,sc_method,theta,print_scr)
        F.scores[i][j] = scr_best

        # construction of output image with the recognized faces per subject in each session
        if show_img == 1:
            if face_detected == 1:
                ii     = ind_best[0]
                jj     = iy[ind_best[0]].item()
                image  = im_crop(img_path_session+img_names_session[jj],facesy[ii],0)
                list1.append(1)
            else:
                image  = []
                list1.append(0)
            image_session = im_concatenate(image_session,image,img_size,0)

    if show_img == 1:
        list2.append(list1)
        image_final   = im_concatenate(image_final,image_session,img_size_f,1)
        image_session = []

# assistance report
if show_img == 1:
    imshow1(image_final)
    imwrite("image_final.png",image_final)
            
    df = pd.DataFrame(list2, columns=id_subjects)
    print(df)
    df.to_excel('Attandance.xlsx', index=False)
F.reportAssistance()
