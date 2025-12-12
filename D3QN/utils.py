import configparser
from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file, encoding='utf-8')
    config = {}
    # === simulation ===
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    # === model ===
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')

    # === memory ===
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')

    # === agent ===
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')

    # === exploration ===
    config['epsilon_start'] = content['exploration'].getfloat('epsilon_start')
    config['epsilon_end'] = content['exploration'].getfloat('epsilon_end')
    config['decay_rate'] = content['exploration'].getfloat('decay_rate')
    config['decay_type'] = content['exploration'].get('decay_type')

    # === congestion_penalty ===
    config['congestion_threshold'] = content['congestion_penalty'].getint('congestion_threshold')
    config['congestion_penalty'] = content['congestion_penalty'].getfloat('congestion_penalty')
    config['waiting_time_penalty_scale'] = content['congestion_penalty'].getfloat('waiting_time_penalty_scale')

    # === state_normalization ===
    config['state_normalization'] = content['state_normalization'].get('state_normalization')

    # === training_stability ===
    config['target_value_clip_min'] = content['training_stability'].getfloat('target_value_clip_min')
    config['target_value_clip_max'] = content['training_stability'].getfloat('target_value_clip_max')

    # === experience_retention ===
    config['experience_retention_ratio'] = content['experience_retention'].getfloat('experience_retention_ratio')
    config['max_buffer_usage'] = content['experience_retention'].getfloat('max_buffer_usage')
    config['retention_strategy'] = content['experience_retention'].get('retention_strategy')
    config['high_value_ratio'] = content['experience_retention'].getfloat('high_value_ratio')
    config['diverse_ratio'] = content['experience_retention'].getfloat('diverse_ratio')

    # === d3qn ===
    config['target_update_freq'] = content['d3qn'].getint('target_update_freq')
    config['per_alpha'] = content['d3qn'].getfloat('per_alpha')
    config['per_beta'] = content['d3qn'].getfloat('per_beta')
    config['per_eps'] = content['d3qn'].getfloat('per_eps')
    config['per_beta_increment'] = content['d3qn'].getfloat('per_beta_increment')

    # === dir ===
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['construction_sumocfg_file_name'] = content['dir']['construction_sumocfg_file_name']

    # === reward ===
    config['flow_reward_weight'] = content['reward'].getfloat('flow_reward_weight')

    # === action_constraints ===
    config['max_green'] = content['action_constraints'].getint('max_green')
    config['imbalance_ratio'] = content['action_constraints'].getfloat('imbalance_ratio')
    config['max_wait_threshold'] = content['action_constraints'].getfloat('max_wait_threshold')

    # === PER ===
    config['use_per'] = content['PER'].getboolean('use_per')
    config['uniform_mix_ratio'] = content['PER'].getfloat('uniform_mix_ratio')
    config['priority_clip_max'] = content['PER'].getfloat('priority_clip_max')

    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file, encoding='utf-8')
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test')
    config['state_normalization'] = content['agent'].get('state_normalization')
    config['max_green'] = content['action_constraints'].getint('max_green')
    config['no_relief_limit'] = content['action_constraints'].getint('no_relief_limit')
    config['relief_drop'] = content['action_constraints'].getint('relief_drop')
    config['use_action_mask'] = content['action_constraints'].getboolean('use_action_mask')
    config['min_green'] = content['action_constraints'].getint('min_green')
    config['global_seed_offset'] = content['stability'].getint('global_seed_offset')
    config['tiny_eval_epsilon'] = content['stability'].getfloat('tiny_eval_epsilon')
    config['warmup_steps'] = content['stability'].getint('warmup_steps')
    config['relief_check_near_cells'] = content['stability'].getint('relief_check_near_cells')
    config['starvation_bias'] = content['stability'].getfloat('starvation_bias')
    config['waiting_time_penalty_scale'] = content['reward'].getfloat('waiting_time_penalty_scale')
    config['congestion_threshold'] = content['reward'].getint('congestion_threshold')
    config['congestion_penalty'] = content['reward'].getfloat('congestion_penalty')
    config['flow_reward_weight'] = content['reward'].getfloat('flow_reward_weight')
    
    # 添加调试信息并安全读取
    print("=== 配置读取调试信息 ===")
    print(f"配置文件路径: {config_file}")
    print(f"可用章节: {content.sections()}")
    print(f"读取到的配置键: {list(config.keys())}")
    print(f"global_seed_offset 值: {config.get('global_seed_offset')}")
    print(f"tiny_eval_epsilon 值: {config.get('tiny_eval_epsilon')}")
    print(f"warmup_steps 值: {config.get('warmup_steps')}")
    print(f"relief_check_near_cells 值: {config.get('relief_check_near_cells')}")
    print(f"starvation_bias 值: {config.get('starvation_bias')}")
    print("=== 配置读取完成 ===")
    print("=== 配置文件调试信息 ===")
    print(f"配置文件路径: {config_file}")
    print(f"可用章节: {content.sections()}")
    if 'dir' in content:
        print(f"[dir] 章节的所有键: {list(content['dir'].keys())}")
        print(f"[dir] 章节的原始内容:")
        for key, value in content['dir'].items():
            print(f"  '{key}' = '{value}'")
    
    # 尝试读取 construction_config_file_name (注意：配置文件中的键名)
    if 'construction_config_file_name' in content['dir']:
        config['construction_config_file_name'] = content['dir']['construction_config_file_name']
        print("成功读取 construction_config_file_name")
    else:
        print("警告: construction_config_file_name 不存在，使用默认值")
        config['construction_sumocfg_file_name'] = config['sumocfg_file_name']
    
    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and the model folder path for saving test results
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')