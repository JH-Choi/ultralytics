import numpy as np 

def read_raw_label(source_label_txt, target_labels):
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
        # s[0]: Tracking ID
        # s[1]: x1 (top left x) s[2]: y1 (top left y) s[3]: x2 (bottom right x) s[4]: y2 (bottom right y)
        # s[5]: frame number s[6]: lost s[7]: occluded s[8]: generated 
        if len(s[9:]) == 1 or s[-1] == "":
            continue
        elif s[10][1:-1] in target_labels.keys():
            lbl = target_labels[s[10][1:-1]]

        tracking_id = int(s[0])
        frame = int(s[5])
        # if frame not in bboxes:
        #     bboxes[frame] = set()
        if tracking_id not in bboxes:
            bboxes[tracking_id] = []
        new_coord = (int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[5]), \
                     int(s[6]), int(s[7]), int(s[8]), s[10][1:-1])
        curr_person = int(s[0]) # Tracking ID
        if curr_person not in same_people:
            same_people[curr_person] = set()
        # xc = (new_coord[0] + new_coord[2]) / 2
        # yc = (new_coord[1] + new_coord[3]) / 2

        # bboxes[int(s[0])].add(new_coord)
        bboxes[tracking_id].append(new_coord)
        # if len(bboxes[frame]) == 0:
        #     bboxes[frame].add(new_coord)
        # else:
        #     add_bbox = True
        #     for (x1,y1,x2,y2,t,lb,_) in bboxes[frame]:
        #         cxc = (x1 + x2) / 2
        #         cyc = (y1 + y2) / 2
        #         dist = np.sqrt((xc - cxc)**2 + (yc - cyc)**2)
        #         if dist < 20:
        #             add_bbox = False
        #             same_people[curr_person].add(t)
        #     if add_bbox:
        #         bboxes[frame].add(new_coord)
    return bboxes
