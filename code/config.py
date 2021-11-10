class Config:
    # override default hyper-parameters using user arguments
    def __init__(self, user_args):
        for k, v in user_args.items():
            if v is not None:
                setattr(self, k, v)

        print('user config: ')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

class ActivitynetC3D(Config):
    def __init__(self, user_args):
        self.dataset = "activitynet"
        self.feature = "c3d"
        self.feature_path = "/home/share/maziyang/HDRR/activitynet/activitynet.c3d.hdf5"
        self.train_path = "/home/share/maziyang/HDRR/activitynet/train_semantic.pkl"
        self.val_path = "/home/share/maziyang/HDRR/activitynet/val_semantic.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.0003
        self.weight_decay = 0
        self.dropout = 0.5
        self.alpha = 1e-3

        self.frame_feature_dim = 500
        self.word_feature_dim = 300
        self.node_dim = 256
        self.max_frames_num = 200
        self.max_words_num = 50
        self.gcn_layers_num = 1
        self.window_widths = (16, 32, 64, 96, 128, 160, 192)
        self.max_srl_num = 3
        super().__init__(user_args)


class CharadesC3D(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "c3d"
        self.feature_path = "/home/share/maziyang/HDRR/charades/charades.c3d.hdf5"
        '''2.22 ziyang modify input'''
        self.train_path = "/home/share/maziyang/HDRR/charades/train_semantic.pkl"
        self.val_path = "/home/share/maziyang/HDRR/charades/val_semantic.pkl"
        '''2.22 ziyang modify input'''
        self.model_load_path = None

        self.epochs = 15
        self.batch_size = 128
        self.lr = 0.001
        self.weight_decay = 0
        self.dropout = 0.5
        self.alpha = 1e-3

        self.frame_feature_dim = 4096
        self.word_feature_dim = 300
        self.node_dim = 512
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 24, 48, 72)
        self.max_srl_num = 3
        super().__init__(user_args)


class CharadesTwostream(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "two-stream"
        self.feature_path = "/home/share/maziyang/HDRR/charades/charades.two-stream.hdf5"
        self.train_path = "/home/share/maziyang/HDRR/charades/train_semantic.pkl"
        self.val_path = "/home/share/maziyang/HDRR/charades/val_semantic.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.0003
        self.weight_decay = 0.00003
        self.dropout = 0.5
        self.alpha = 1e-3

        self.frame_feature_dim = 8192
        self.word_feature_dim = 300
        self.node_dim = 512
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 24, 48, 72)
        self.max_srl_num = 3
        super().__init__(user_args)


class CharadesI3D(Config):
    def __init__(self, user_args):
        self.dataset = "charades"
        self.feature = "i3d"
        self.feature_path = "/home/share/maziyang/HDRR/charades/charades.i3d.hdf5"
        self.train_path = "/home/share/maziyang/HDRR/charades/train_semantic.pkl"
        self.val_path = "/home/share/maziyang/HDRR/charades/val_semantic.pkl"
        self.model_load_path = None

        self.epochs = 30
        self.batch_size = 128
        self.lr = 0.002
        self.weight_decay = 0.00003
        self.dropout = 0.5
        self.alpha = 1e-3

        self.frame_feature_dim = 1024
        self.word_feature_dim = 300
        self.node_dim = 128
        self.max_frames_num = 75
        self.max_words_num = 10
        self.gcn_layers_num = 1
        self.window_widths = (6, 12, 24, 48, 72)
        self.max_srl_num = 3
        super().__init__(user_args)


config = {
    ('activitynet', 'c3d'): ActivitynetC3D,
    ('charades', 'c3d'): CharadesC3D,
    ('charades', 'i3d'): CharadesI3D,
    ('charades', 'two-stream'): CharadesTwostream
}


def get_config(user_args):
    return config[(user_args.dataset, user_args.feature)](user_args.__dict__)
