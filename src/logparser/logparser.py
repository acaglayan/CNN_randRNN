import argparse
import fnmatch
import os

import numpy as np

from basic_utils import Models, DataTypes, RunSteps, format_bytes


def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log-root", dest="log_root", default="../../logs",
                        help="Dir name for the log file")
    parser.add_argument("--log-dir", dest="log_dir", default=RunSteps.OVERALL_RUN + "/",
                        help="Dir name for the log file(s)")
    parser.add_argument("--net-model", dest="net_model", default=Models.AlexNet,
                        choices=Models.ALL, type=str.lower, help="Backbone CNN model")
    parser.add_argument("--data-type", dest="data_type", default=DataTypes.Depth, choices=DataTypes.ALL,
                        type=str.lower, help="Data type to process, crop for rgb, depthcrop for depth data")
    parser.add_argument("--split", dest="split", default=1, type=int, choices=range(1, 11), help="Split number")
    parser.add_argument("--mode", dest="mode", default="accuracy", choices=['accuracy', 'mem_time', 'avr_mem_time'],
                        help="Print mode choice")
    parser.add_argument("--debug-mode", dest="debug_mode", default=1, type=int, choices=[0, 1])
    parser.add_argument("--logfile", dest="logfile", default=1, type=int, choices=range(1, 5), help="Which trial log")

    params = parser.parse_args()
    return params


def get_accuracy(fp):
    fp.readline()
    fp.readline()
    result_line = fp.readline()
    result_end_ind = result_line.find('%')
    result_start_ind = result_end_ind - 5
    result = result_line[result_start_ind:result_end_ind]

    return result


def get_memory_time(line):
    mem_start_ind = line.find('consumed: ')
    mem_end_ind = line.find(';', mem_start_ind)
    memory_con = line[mem_start_ind + 10:mem_end_ind]

    time_start_ind = line.find('exec time: ')
    exec_time = line[time_start_ind + 11:-1]

    return memory_con, exec_time


def read_file(params, log_file):
    accuracies = []
    mem_cons, exec_times = [], []
    with open(log_file) as fp:
        line = fp.readline()
        while line:
            if 'Running Layer-' in line:
                if params.mode is 'accuracy':
                    accuracy = get_accuracy(fp)
                    accuracies.append(accuracy)
            elif ('before' and 'after') in line:
                if params.mode is not 'accuracy':
                    memory_con, exec_time = get_memory_time(line)
                    mem_cons.append(memory_con)
                    exec_times.append(exec_time)

            line = fp.readline()

    fp.close()
    if params.mode is 'accuracy':
        return accuracies
    else:
        return mem_cons, exec_times


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def calc_mem_record(record):
    is_byte = is_number(record[-2:-1])
    if not is_byte:
        rec_val = np.float32(record[:-2])
        mem_len_type = record[-2:].lower()
        if mem_len_type == 'kb':
            rec_val *= 1000
        elif mem_len_type == 'mb':
            rec_val *= 1e6
        elif mem_len_type == 'gb':
            rec_val *= 1e9
    else:
        rec_val = np.float32(record[:-1])

    return rec_val


def calc_exec_time_record(record):
    rec_hr, rec_min, rec_sec = record.split(':')
    rec_hr = np.int(rec_hr)
    rec_min = np.int(rec_min)
    rec_sec = np.int(rec_sec)

    return rec_hr, rec_min, rec_sec


def calc_average_times(time, num_file):
    total_hrs, total_mins, total_secs = time
    avr_hrs = np.divide(total_hrs, num_file)
    avr_mins = np.divide(total_mins, num_file)
    avr_sec = np.divide(total_secs, num_file)

    avr_mins += (avr_hrs - int(avr_hrs)) * 60
    avr_hrs = int(avr_hrs)
    avr_sec += (avr_mins - int(avr_mins)) * 60
    avr_mins = int(avr_mins)

    avr_mins += avr_sec // 60
    avr_sec = np.fmod(avr_sec, 60)

    avr_hrs += avr_mins // 60
    avr_mins = np.fmod(avr_mins, 60)

    avr_times = str(int(avr_hrs)) + ':' + str(int(avr_mins)) + ':' + \
                str(avr_sec)

    return avr_times


def process_one_log(params):
    if 'log_file' not in params:
        log_file = params.log_root + params.log_dir + params.timestamp + '_' + str(params.logfile) + '-' + \
                   params.net + '_' + params.data_type + '_split_' + str(params.split) + '.log'
    else:
        log_file = params.log_file
    if params.mode is 'accuracy':
        accuracies = read_file(params, log_file)
        for acc in accuracies:
            print('{}'.format(acc))
    elif params.mode is 'mem_time':
        mem_cons, exec_times = read_file(params, log_file)
        for i in range(len(mem_cons)):
            print('{}\t{}'.format(mem_cons[i], exec_times[i]))


def process_logs_from_dir(params):
    suffix = '*_' + params.data_type + '_*'
    path = os.path.join(params.log_root, params.log_dir)
    list_mem_cons, list_exec_times = [], []

    for logfile in fnmatch.filter(sorted(
            sorted(os.listdir(path), key=lambda x: int(x.split(params.net_model)[0][-2])),
            key=lambda x: int(x.split('split_')[1].split('.log')[0])
    ), suffix):
        if params.mode is 'avr_mem_time':
            if params.split == int(logfile.split('split_')[1].split('.log')[0]):
                mem_cons, exec_times = read_file(params, os.path.join(path, logfile))
                list_mem_cons.append(mem_cons)
                list_exec_times.append(exec_times)
        else:
            params.log_file = os.path.join(path, logfile)
            print('Processing file {}'.format(logfile))
            process_one_log(params)

    if params.mode is 'avr_mem_time':
        print('average memory and time stats for {} split-{} {} images'.format(params.net_model, params.split, suffix))
        num_file = len(list_mem_cons)
        file_record_len = len(list_mem_cons[0])
        total_mem_cons = np.zeros(shape=file_record_len, dtype=np.float32)
        total_exec_times = np.zeros(shape=(file_record_len, 3), dtype=np.int)
        for file_no in range(num_file):
            for record_no in range(file_record_len):
                current_mem_byte = calc_mem_record(list_mem_cons[file_no][record_no])
                total_mem_cons[record_no] += current_mem_byte

                current_hr, current_min, current_sec = calc_exec_time_record(list_exec_times[file_no][record_no])
                total_exec_times[record_no] += (current_hr, current_min, current_sec)

        for record_no in range(len(total_mem_cons)):
            curr_avr_mem_con = format_bytes(np.divide(total_mem_cons[record_no], num_file))
            curr_avr_exec_time = calc_average_times(total_exec_times[record_no], num_file)
            print('{}\t{}'.format(curr_avr_mem_con, curr_avr_exec_time))


def main():
    params = get_params()
    annex = ''
    if params.debug_mode:
        annex += '[debug]'
    params.log_root += annex + '/'

    process_logs_from_dir(params)


if __name__ == '__main__':
    main()
