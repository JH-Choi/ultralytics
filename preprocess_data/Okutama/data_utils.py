import cv2
import numpy as np 
import pdb

def read_raw_label(source_label_txt, target_labels, skip_interpolate=False):
    # read only detection bouding boxes
    bboxes = {}
    people = set()
    same_people = {}
    label_lines = []
    with open(source_label_txt,"r") as f:
        label_lines = f.readlines()
    
    for line in label_lines:
        line = line.replace("\n","")
        s = line.split(" ")
        if skip_interpolate:
            if int(s[8]) == 1: continue

        if len(s[9:]) == 1 or s[-1] == "":
            continue
        elif s[10][1:-1] in target_labels.keys():
            lbl = target_labels[s[10][1:-1]]

        frame = int(s[5])
        if frame not in bboxes:
            bboxes[frame] = set()
        new_coord = (int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[0]),int(lbl))
        curr_person = int(s[0]) # Tracking ID
        if curr_person not in same_people:
            same_people[curr_person] = set()
        xc = (new_coord[0] + new_coord[2]) / 2
        yc = (new_coord[1] + new_coord[3]) / 2
        if len(bboxes[frame]) == 0:
            bboxes[frame].add(new_coord)
        else:
            add_bbox = True
            for (x1,y1,x2,y2,t,lb) in bboxes[frame]:
                cxc = (x1 + x2) / 2
                cyc = (y1 + y2) / 2
                dist = np.sqrt((xc - cxc)**2 + (yc - cyc)**2)
                if dist < 20:
                    add_bbox = False
                    same_people[curr_person].add(t)
            if add_bbox:
                bboxes[frame].add(new_coord)
    return bboxes

def read_write_image(img_fn, width, height, out_file):
    raw_img = cv2.imread(str(img_fn))
    raw_img = cv2.resize(raw_img, (width, height))
    cv2.imwrite(str(out_file), raw_img)
    return raw_img

