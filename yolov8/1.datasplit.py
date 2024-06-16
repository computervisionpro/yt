import os
import shutil
import random
print()

traindir = './train/images'
valdir = './valid/images'

traind_l = './train/labels'
valdir_l = './valid/labels'


# create directories
def create_dirs(path):

    if not os.path.exists(path):
        os.mkdir(path)
        print(path, 'created!')
    else:
        print(path, 'exists!')

create_dirs('train2')
create_dirs('valid2')


otrain_img = './train2/images'
create_dirs(otrain_img)

ovalid_img = './valid2/images'
create_dirs(ovalid_img)

otrain_labels = './train2/labels'
create_dirs(otrain_labels)

ovalid_labels = './valid2/labels'
create_dirs(ovalid_labels)
print()


# split data

def split_data(in_img, in_lb, out_img, out_lb, limit):

    """
    in_img: input image directory
    in_lb,: input label directory
    out_img: output image directory
    out_lb: output label directory
    
    """
    
    images1 = os.listdir(in_img)[:limit]
    random.shuffle(images1)

    labels_temp1 = os.listdir(in_lb)

    count =1
    for im in images1:
        #print(im)
        base = os.path.basename(im)
        fname = os.path.splitext(base)[0]

        print(count, fname)
        lname = fname+'.txt'

        if lname in labels_temp1:
            shutil.copy(in_img+'/'+im, out_img)
            shutil.copy(in_lb+'/'+lname, out_lb)


        count+=1

    print('Done!')



#split_data(traindir, traind_l, otrain_img, otrain_labels, 5000)
print('Training data-set created')

#split_data(valdir, valdir_l, ovalid_img, ovalid_labels, 300)
print('Valdation data-set created')












