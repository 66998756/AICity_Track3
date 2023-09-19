import os
from os import walk
import argparse
import csv

class record:
    def __init__(self, array):
        self.Filename = array[0]
        self.Camera_View = array[1]
        self.Activity_Type = array[2]
        self.Start_Time = array[3]
        self.End_Time = array[4]
        self.Label = array[5]
        self.Appearance_Block = array[6]
    

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok = True)
    #    print("makedir "+dir_path)    
    #else:
    #    print(dir_path+" already exist, no need to makedir.")

#找所有csv。
def show_csvs(path, all_csvs):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_csvs)
        else:
            if not cur_path.endswith(('csv')):
                continue
            else:
                all_csvs.append(cur_path )
    return all_csvs

#找單層dir。
def show_sinle_layer_dir(path, sub_dirs):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            sub_dirs.append(file)
        #else: skip do nothing
    return sub_dirs

def t2s(t):
    print(t)
    h,m,s = t.strip().split(":")
    return str(int(h)*3600+int(m)*60+int(s))

def fucking_fname_fix(filename):
    path = filename[:filename.rfind('_')+1].capitalize()+ 'NoAudio'+filename[filename.rfind('_'):]+'.MP4'
    if filename.find('Rearview') != -1: 
        path = path[:path.find('Rearview')+4] + '_' + path[path.find('Rearview')+4:]

    return path

def read_write_csv(input_path, log):
    headerline = 'Filename,Camera View,Activity Type,Start Time,End Time,Label (Primary),Appearance Block'
    #這一串沒意義不是data。

    f = open(input_path)
    #type(file)
    try:
        csvreader = csv.reader(f)

        temp_name = ''

        for row in csvreader:
            #不一樣才是data
            # if not row.__eq__(headerline.split(',')):
            if row[0] != 'Filename':
                #row_array = row.split(',')

                if row[0] == '':
                    row[0] = temp_name

                else:
                    temp_name = fucking_fname_fix(str(row[0]))
                    row[0] = temp_name
                context = row[0]+','+t2s(row[3])+','+t2s(row[4])+','+str(row[5])
                with open(log, 'a') as fd:
                    fd.write(f'\n{context}')
                
    except RuntimeError as err:
        print('RuntimeError context: ' + err.__context__)


def main(argv):
    #dirpath = pathlib.Path(argv.dirpath)
    input = argv.input
    log = argv.log
    log2 = argv.log2
    testset = argv.testset
    test_map={}
    for id in testset:
        test_map['user_id_'+id] = True
    #mkdir(argv.outputpath)

    id_dirs = show_sinle_layer_dir(input, [])
    for id in id_dirs:
        dir_path = os.path.join(input, id)
        logpath = log
        value = test_map.get(id)
        if value is not None:
            logpath = log2

        csvs = show_csvs(dir_path, [])
        for csv in csvs:
            read_write_csv(csv, logpath)

def input_label_dict(root):
    label_dict = {}
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith('.csv'):
                # print(file)
                with open(os.path.join(root, file), 'r') as label_csv:
                    labels = label_csv.readlines()
                    for label in labels:
                        if label[0] == "F": continue
                        # print(label)
                        label = label.split(',')
                        if label[0]: filename = label[0]
                        # print(filename, label[3], label[4], label[5])
                        
                        if filename not in label_dict.keys():
                            path = filename[:filename.rfind('_')+1].capitalize()+ \
                                'NoAudio'+filename[filename.rfind('_'):]+'.MP4'
                            if filename.find('Rearview') != -1: 
                                path = path[:path.find('Rearview')+4] + '_' + path[path.find('Rearview')+4:]
                            label_dict[filename] = {
                                'path': [os.path.join(root, path)],
                                'start_time': [_times2sec(label[3])],
                                'end_time': [_times2sec(label[4])],
                                'label': [label[5].split(' ')[1]]
                            }
                        else:
                            label_dict[filename]['start_time'].append(_times2sec(label[3]))
                            label_dict[filename]['end_time'].append(_times2sec(label[4]))
                            label_dict[filename]['label'].append(label[5].split(' ')[1])
    

    headerline = 'path,Start Time,End Time,Label\n'
    train_dict = open(root + '/train.csv', 'w')
    test_dict = open(root + '/test.csv', 'w')
    train_dict.write(headerline)
    test_dict.write(headerline)
    test_list = ['30932', '86952','96269']
    for key in label_dict.keys():
        for test_file in test_list:
            if test_file in key:
                test_dict.write(label_dict[key]['path']+[',']+label_dict[key]['start_time']+[',']+label_dict[key]['end_time']+[',']+label_dict[key]['label']+['\n'])
            else:
                train_dict.write(label_dict[key]['path']+[',']+label_dict[key]['start_time']+[',']+label_dict[key]['end_time']+[',']+label_dict[key]['label']+['\n'])
    train_dict.close()
    test_dict.close()

def _times2sec(times: str, fps=30):
    times = times.split(':')
    target_time = (int(times[0]) * 60 * 60 + int(times[1]) * 60 + int(times[2]))
    return target_time

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='/workspaces/mvl/ai-jam/city/A1', help='read from where?')#, required=True)
    # parser.add_argument('--log', type=str, default='/workspaces/mvl/ai-jam/city/log.txt', help='log to where')#, required=True)
    # parser.add_argument('--log2', type=str, default='/workspaces/mvl/ai-jam/city/log2.txt', help='log to where2')#, required=True)
    # parser.add_argument('--testset', type=list, default=['86952','96269','99882'], help='those belong to test set')#, required=True)

    # args = parser.parse_args()
    # main(args)
    input_label_dict("/mnt/Nami/2023_AI_City_challenge_datasets/Track_3/A1")