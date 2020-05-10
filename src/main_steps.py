import os

from basic_utils import RunSteps, PrForm, DataTypes, Models, Pools
from alexnet_model import AlexNet
from densenet_model import DenseNet
from extract_cnn_features import fixed_extraction, finetuned_extraction
from finetune_models import process_finetuning
from main import init_logger, is_initial_params_suitable, is_suitable_rgbd_fusion, init_save_dirs, get_timestamp, \
    get_initial_parser
from resnet_models import ResNet
from save_colored_depth import process_depth_save
from sunrgbd_scene_save import process_dataset_save
from vgg16_model import VGG16Net


def is_suitable_level_fusion(params):
    is_suitable = True

    if params.data_type not in (DataTypes.RGB, DataTypes.Depth):
        if params.data_type == DataTypes.RGBD:
            is_suitable = is_suitable_rgbd_fusion(params)
        else:
            is_suitable = False
    else:
        confidence_scores_path = params.dataset_path + params.features_root + params.proceed_step + \
                                 '/svm_confidence_scores/'
        if not os.path.exists(confidence_scores_path):
            print('{}{}Failed to load the RGB/Depth scores! First, you need to run the system to create RGB/Depth '
                  'scores!{}'.format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
            print('{}{}Or set --fusion-levels param to 0!{}'.format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
            is_suitable = False
        else:
            pass  # TODO

    return is_suitable


def is_cnn_rnn_features_available(params, cnn):
    if params.proceed_step == RunSteps.FIX_RECURSIVE_NN:
        cnn_feats_path = params.dataset_path + params.features_root + RunSteps.FIX_EXTRACTION + '/' + \
                         params.net_model + '_results_' + params.data_type
        rnn_feats_path = params.dataset_path + params.features_root + params.proceed_step + '/' + \
                         params.net_model + '_results_' + params.data_type
    elif params.proceed_step == RunSteps.FINE_RECURSIVE_NN:
        cnn_feats_path = params.dataset_path + params.features_root + RunSteps.FINE_TUNING + \
                         '/split-' + str(params.split_no) + '/' + params.net_model + '_results_' + \
                         params.data_type
        rnn_feats_path = params.dataset_path + params.features_root + params.proceed_step + '/' + \
                         params.net_model + '_results_' + params.data_type + '/split_' + str(params.split_no)
    else:
        return True

    if cnn:
        path = cnn_feats_path
    else:
        path = rnn_feats_path

    if not os.path.exists(path):
        print('{}{}Failed to load the CNN/RNN features! Please run the system to create '
              'features first!{}'.format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        print('{}{}Please check the paths/params or run the code for feature extraction first!{}'.
              format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        return False
    else:
        return True


def save_sunrgbd_scene():
    from sunrgbd_loader import DataTypes as sunrgbd_datatype
    parser = get_initial_parser()
    params = parser.parse_args()
    params.debug_mode = 0
    params.dataset_path = "../data/sunrgbd/"
    if params.data_type == DataTypes.RGB:
        params.data_type = sunrgbd_datatype.RGB
    elif params.data_type == DataTypes.Depth:
        params.data_type = sunrgbd_datatype.Depth
    else:
        print('{}{}The parameter {}--data-type{} should be {} RGB or Depth{}!{}'.
              format(PrForm.BOLD, PrForm.RED, PrForm.BLUE, PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.END_FORMAT))
        return
    params.proceed_step = RunSteps.SAVE_SUNRGBD
    logfile_name = params.log_dir + '/' + params.proceed_step + '/' + params.data_type + '_save.log'
    init_logger(logfile_name, params)
    process_dataset_save(params)


def save_depth():
    params = get_save_depth_params()
    if params.data_type != DataTypes.Depth:
        print('{}{}The parameter {}--data-type{} should be {}depth{}!{}'.
              format(PrForm.BOLD, PrForm.RED, PrForm.BLUE, PrForm.RED, PrForm.GREEN, PrForm.RED, PrForm.END_FORMAT))
        return
    logfile_name = params.log_dir + params.proceed_step + '/' + get_timestamp() + '_' + params.data_type + \
                   '_colorized_save.log'
    init_logger(logfile_name, params)
    process_depth_save(params)


def eval_models(proceed_step):
    params = get_recursive_params(proceed_step)
    params = init_save_dirs(params)
    if not is_initial_params_suitable(params):
        return

    if params.fusion_levels and not is_suitable_level_fusion(params):
        return

    if params.load_features and not is_cnn_rnn_features_available(params, cnn=0):
        return

    if params.data_type != DataTypes.RGBD and not is_cnn_rnn_features_available(params, cnn=1):
        return

    logfile_name = params.log_dir + proceed_step + '/' + get_timestamp() + '_' + str(params.trial) + '-' + \
                   params.net_model + '_' + params.data_type + '_split_' + str(params.split_no) + '.log'

    init_logger(logfile_name, params)

    if params.net_model == Models.AlexNet:
        model = AlexNet(params)
    elif params.net_model == Models.VGGNet16:
        model = VGG16Net(params)
    elif params.net_model == Models.ResNet50 or params.net_model == Models.ResNet101:
        model = ResNet(params)
    elif params.net_model == Models.DenseNet121:
        model = DenseNet(params)
    else:
        print('{}{}Unsupported model selection! Please check your model choice in arguments!{}'
              .format(PrForm.BOLD, PrForm.RED, PrForm.END_FORMAT))
        return

    model.eval()


def finetune_model():
    params = get_finetune_params()
    params = init_save_dirs(params)
    if not is_initial_params_suitable(params):
        return

    logfile_name = params.log_dir + params.proceed_step + '/' + get_timestamp() + '_' + str(params.trial) + '-' + \
                   params.net_model + '_' + params.data_type + '_split_' + str(params.split_no) + '.log'
    init_logger(logfile_name, params)

    process_finetuning(params)


def extract_fixed_features():
    params = get_extraction_params()
    params = init_save_dirs(params)
    if not is_initial_params_suitable(params):
        return
    logfile_name = params.log_dir + params.proceed_step + '/' + get_timestamp() + '_' + params.net_model + '_' + \
                   params.data_type + '_cnn_extraction.log'
    init_logger(logfile_name, params)

    fixed_extraction(params)


def extract_finetuned_features():
    params = get_finetuned_extraction_params()
    params = init_save_dirs(params)
    if not is_initial_params_suitable(params):
        return

    logfile_name = params.log_dir + params.proceed_step + '/' + get_timestamp() + '_' + params.net_model + '_' + \
                   params.data_type + '_split_' + str(params.split_no) + '.log'
    init_logger(logfile_name, params)

    finetuned_extraction(params)


def get_save_depth_params():
    parser = get_initial_parser()
    params = parser.parse_args()
    params.net_model = 'all'
    params.proceed_step = RunSteps.COLORIZED_DEPTH_SAVE
    params = init_save_dirs(params)
    return params


def get_extraction_params():
    parser = get_initial_parser()
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int)
    params = parser.parse_args()
    params.proceed_step = RunSteps.FIX_EXTRACTION
    return params


def get_recursive_params(proceed_step):
    parser = get_initial_parser()
    parser.add_argument("--split-no", dest="split_no", default=1, type=int, choices=range(1, 11), help="Split number")
    parser.add_argument("--num-rnn", dest="num_rnn", default=128, type=int, help="Number of RNN")
    parser.add_argument("--save-features", dest="save_features", default=0, type=int, choices=[0, 1])
    parser.add_argument("--batch-size", dest="batch_size", default=1000, type=int)
    parser.add_argument("--trial", dest="trial", default=0, type=int,
                        help="Trial number is for running the same model with the same params for evaluation "
                             "the effect of randomness.")
    parser.add_argument("--reuse-randoms", dest="reuse_randoms", default=1, choices=[0, 1], type=int,
                        help="Handles whether the random weights are gonna save/load or not")
    parser.add_argument("--fusion-levels", dest="fusion_levels", default=0, choices=[0, 1], type=int,
                        help="Handles whether fusion is performed")
    parser.add_argument("--load-features", dest="load_features", default=0, type=int, choices=[0, 1])
    parser.add_argument("--pooling", dest="pooling", default=Pools.RANDOM, choices=Pools.ALL,
                        type=str.lower, help="Pooling type")
    params = parser.parse_args()
    params.proceed_step = proceed_step
    return params


def get_finetune_params():
    parser = get_initial_parser()
    parser.add_argument("--split-no", dest="split_no", default=1, type=int, choices=range(1, 11), help="Split number")
    parser.add_argument("--batch-size", dest="batch_size", default=16, type=int)
    parser.add_argument("--lr", dest="lr", default=0.0001, type=float, help='Initial learning rate')
    parser.add_argument("--momentum", dest="momentum", default=0.95, type=float, help='Momentum rate')
    parser.add_argument("--step-size", dest="step_size", default=10, type=int,
                        help='Number of epoch for each learning rate decay')
    parser.add_argument("--gamma", dest="gamma", default=0.1, type=float, help="Factor rate of learning rate decay")
    parser.add_argument("--num-epochs", dest="num_epochs", default=40, type=int)
    parser.add_argument("--trial", dest="trial", default=0, type=int,
                        help="Trial number is used to run the same model with the same params to evaluate "
                             "the effect of randomness.")
    params = parser.parse_args()
    params.proceed_step = RunSteps.FINE_TUNING
    return params


def get_finetuned_extraction_params():
    parser = get_initial_parser()
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int)
    parser.add_argument("--split-no", dest="split_no", default=1, type=int, choices=range(1, 11), help="Split number")
    params = parser.parse_args()
    params.proceed_step = RunSteps.FINE_EXTRACTION
    return params


def separated_steps_main():
    proceed_step = RunSteps.COLORIZED_DEPTH_SAVE

    assert proceed_step in (RunSteps.COLORIZED_DEPTH_SAVE, RunSteps.SAVE_SUNRGBD,
                            RunSteps.FIX_EXTRACTION, RunSteps.FIX_RECURSIVE_NN,
                            RunSteps.FINE_TUNING, RunSteps.FINE_EXTRACTION, RunSteps.FINE_RECURSIVE_NN)
    if proceed_step == RunSteps.SAVE_SUNRGBD:
        save_sunrgbd_scene()
    elif proceed_step == RunSteps.COLORIZED_DEPTH_SAVE:
        save_depth()
    elif proceed_step == RunSteps.FIX_EXTRACTION:
        extract_fixed_features()
    elif proceed_step == RunSteps.FINE_TUNING:
        finetune_model()
    elif proceed_step == RunSteps.FINE_EXTRACTION:
        extract_finetuned_features()
    else:  # "FIX_RECURSIVE_NN" OR "FINE_RECURSIVE_NN"
        eval_models(proceed_step)


if __name__ == '__main__':
    separated_steps_main()
