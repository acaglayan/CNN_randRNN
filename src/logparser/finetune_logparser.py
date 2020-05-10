import fnmatch
import os


class Logfile:
    def __init__(self, data_type, batch_size, lr, momentum, step_size, gamma, trial):
        self._data_type = data_type
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = momentum
        self._step_size = step_size
        self._gamma = gamma
        self._trial = trial
        self._training_time = ""
        self._best_result = 0.0
        self._best_epoch = 0
        self._is_fail = True
        self._filename = ""

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, momentum):
        self._momentum = momentum

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def trial(self):
        return self._trial

    @trial.setter
    def trial(self, trial):
        self._trial = trial

    @property
    def training_time(self):
        return self._training_time

    @training_time.setter
    def training_time(self, training_time):
        self._training_time = training_time

    @property
    def best_result(self):
        return self._best_result

    @best_result.setter
    def best_result(self, best_result):
        self._best_result = best_result

    @property
    def best_epoch(self):
        return self._best_epoch

    @best_epoch.setter
    def best_epoch(self, best_epoch):
        self._best_epoch = best_epoch

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    @property
    def is_fail(self):
        return self._is_fail

    @is_fail.setter
    def is_fail(self, is_fail):
        self._is_fail = is_fail

    def print_log_info(self):
        print('data_type={}, batch_size={}, lr={}, momentum={}, step_size={}, gamma={}, trial={}, fine-tune-time={}, '
              'result={}'.format(self.data_type, self.batch_size, self.lr, self.momentum, self.step_size, self.gamma,
                                 self.trial, self.training_time, self.best_result))

    def print_log_info_v2(self):
        print('Processing file {}'.format(self.filename))
        print('data_type={}, batch_size={}, lr={}, momentum={}, step_size={}, gamma={}, trial={}, fine-tune-time={}, '
              'result={} at {} epoch'.format(self.data_type, self.batch_size, self.lr, self.momentum, self.step_size, self.gamma,
                                 self.trial, self.training_time, float(self.best_result) * 100, self.best_epoch))

    def print_log_info_fail_viewpoint(self):
        print('trial={}, batch_size={}, lr={}, momentum={}, step_size={}, gamma={}, data_type={}, is_fail={}'.
              format(self.trial, self.batch_size, self.lr, self.momentum, self.step_size, self.gamma, self.data_type,
                     self.is_fail))


def parse_params(line):
    params = line[line.find('(') + 1:line.find(')')]
    params = params.split(',')
    data_type = params[1].split('=')[1]
    batch_size = params[0].split('=')[1]
    lr = params[8].split('=')[1]
    momentum = params[9].split('=')[1]
    step_size = params[14].split('=')[1]
    gamma = params[6].split('=')[1]
    trial = params[15].split('=')[1]

    logfile = Logfile(data_type, batch_size, lr, momentum, step_size, gamma, trial)
    return logfile


def read_file(file_path):
    logfile = Logfile
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            if 'Running params:' in line:
                logfile = parse_params(line)
            elif 'Training complete in' in line:
                logfile.training_time = line[line.find(' in ') + 4:-1]
            elif 'Best val Acc' in line:
                next_line = fp.readline()
                if next_line:
                    line = next_line
                    continue
                logfile.best_result = float(line[line.find('Acc:') + 5:line.find(' at')])
                logfile.best_epoch = int(line[line.find(' at') + 4:line.find('. epoch')])

            line = fp.readline()
        fp.close()
    return logfile


def process_logs():
    logfiles = []
    log_dir = '../logs/finetune_params/one-stage-finetune/densenet121_crop_fine_tuning/'
    for filename in sorted(os.listdir(log_dir)):
        file_path = os.path.join(log_dir, filename)
        logfile = read_file(file_path)
        logfiles.append(logfile)

    logfiles.sort(key=lambda x: x.best_result, reverse=True)
    for lofile in logfiles:
        lofile.print_log_info()


def read_file_fail_viewpoint(file_path):
    logfile = Logfile
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            if 'Running params:' in line:
                logfile = parse_params(line)
            elif 'Using device "cuda"' in line:
                logfile.is_fail = False
            line = fp.readline()
        fp.close()
    return logfile


def search_fails():
    logfiles = []
    log_dir = '../../logs/finetune_params'
    for filename in sorted(os.listdir(log_dir)):
        file_path = os.path.join(log_dir, filename)
        logfile = read_file_fail_viewpoint(file_path)
        logfiles.append(logfile)

    logfiles.sort(key=lambda x: x.is_fail, reverse=True)
    fail_logfiles = []
    for logfile in logfiles:
        # logfile.print_log_info_fail_viewpoint()
        if logfile.is_fail:
            fail_logfiles.append(logfile)

    print('\nTotal fails is {}'.format(len(fail_logfiles)))
    return fail_logfiles


def is_not_completed_file(file_path):
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            if 'Training complete in' in line:
                return False

            line = fp.readline()
        fp.close()
    return True


def read_not_completed_file(file_path):
    best_val_acc = 0.0
    logfile = Logfile
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            if 'Running params:' in line:
                logfile = parse_params(line)
            elif ' Epoch ' in line:
                dash_line = fp.readline()
                train_loss_line = fp.readline()
                test_loss_line = fp.readline()
                if not test_loss_line:
                    break
                curr_val_acc = float(test_loss_line.split('Acc: ')[1])
                if curr_val_acc > best_val_acc:
                    best_val_acc = curr_val_acc

            line = fp.readline()
        fp.close()
    logfile.best_result = best_val_acc
    return logfile


def process_not_completed_logs():
    logfiles = []
    log_dir = '../logs/finetune_params/two-stage-finetune/stage-2/alexnet_depthcrop_fine_tuning/'
    for filename in sorted(os.listdir(log_dir)):
        file_path = os.path.join(log_dir, filename)
        if is_not_completed_file(file_path):
            logfile = read_not_completed_file(file_path)
        else:
            logfile = read_file(file_path)

        logfiles.append(logfile)

    logfiles.sort(key=lambda x: x.best_result, reverse=True)
    for lofile in logfiles:
        lofile.print_log_info()


def process_logs_from_dir():
    logfiles = []
    suffix = '*_color_*'
    log_dir = '../logs/fine_tuning/resnet101/'
    for logfile in fnmatch.filter(sorted(os.listdir(log_dir), key=lambda x: int(x.split('split_')[1].split('.log')[0])),
                                  suffix):
        file_path = os.path.join(log_dir, logfile)
        logfile = read_file(file_path)
        logfile.filename = file_path
        logfiles.append(logfile)

    logfiles.sort(key=lambda x: x.best_result, reverse=True)
    for logfile in logfiles:
        # lofile.print_log_info()
        logfile.print_log_info_v2()
        print()


def main():
    # process_not_completed_logs()
    # process_logs()
    # fail_logs = search_fails()
    process_logs_from_dir()


if __name__ == '__main__':
    main()
