'''
UCF Aerial Action Dataset: https://www.crcv.ucf.edu/data/UCF_Aerial_Action.php
'''
import cv2
import xmltodict





# List of all activity names to:extract the correct XGTF `attributes`
CLASSES = ['Standing', 'Walking', 'Running', 'Digging', 'Gesturing', 'Carrying',
           'Opening a Trunk', 'Closing a Trunk', 'Getting Into a Vehicle',
           'Getting Out of a Vehicle', 'Loading a Vehicle', 'Unloading a Vehicle',
           'Entering a Facility', 'Exiting a Facility']
