import copy


base = dict(
    # dataset configs
    data = dict(
        root='/yours/dataset/path',
        datasets_base_to_new=['dtd', 'caltech101', 'eurosat', 'ucf101', 'oxford_flowers', 
                              'oxford_pets', 'stanford_cars', 'fgvc_aircraft', 'food101', 'sun397', 
                              'imagenet'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
    ),

    # mail configs
    mail = dict(
        username='yours@email.com',
        password='yours password here',
        host='yours.email.com',
        to='yours@email.com',
    ),
)

##########################################################
# Base-to-new Generalization of Baseline Methods

zsclip = dict(
    gpu_ids = [0],
    mode='b2n',
    
    # training configs
    train = dict(
        trainer='ZeroShotCLIP',      # trainer, please see trainers
        cfg='vit_b16',               # config, please see configs/
        seeds=[1],                   # seeds
        loadep=-1,                   # load epoch, -1 to load the last epoch
        shots=16,                    # num of shots
        opts=[],                     # extra opts, if you have, please add, such as [OPTIM.MAX_EPOCH, 10]
    ),
    
    # grid search configs, if enable=False, grid search will not be used
    grid_search = dict(enable=False),
    
    # output configs
    output = dict(
        root='outputs/zsclip',   # output root
        result='results/zsclip', # result root 
        remove_dirs=['root'],    # which directorys will be removed before training task starts
    ),
)

coop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='CoOp',
        cfg='vit_b16_ep10_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/coop',
        result='results/coop', 
        remove_dirs=['root'],
    ),
)

kgcoop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='KgCoOp',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/kgcoop',
        result='results/kgcoop', 
        remove_dirs=['root'],
    ),
)

prograd = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='ProGrad',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/prograd',
        result='results/prograd', 
        remove_dirs=['root'],
    ),
)

dapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='DAPT',
        cfg='vit_b16_ep10_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/dapt',
        result='results/dapt', 
        remove_dirs=['root'],
    ),
)

kgcoop_dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='KgCoOpDePT',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/dept',
        result='results/dept', 
        remove_dirs=['root'],
    ),
)

####################################################################
# Base-to-new Generalization with CAPT

coop_capt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='UCPCoOp',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/ucpcoop',
        result='results/ucpcoop', 
        remove_dirs=['root'],
    ),
)

kgcoop_capt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='UCPKgCoOp',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/ucpkgcoop',
        result='results/ucpkgcoop', 
        remove_dirs=['root'],
    ),
)

prograd_capt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='UCPProGrad',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/ucpprograd',
        result='results/ucpprograd', 
        remove_dirs=['root'],
    ),
)

dapt_capt = dict(
    gpu_ids = [0],
    mode='b2n',

    train = dict(
        trainer='UCPDAPT',
        cfg='vit_b16_ep10_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/ucpdapt',
        result='results/ucpdapt', 
        remove_dirs=['root'],
    ),
)

kgcoop_dept_capt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='UCPKgCoOpDePT',
        cfg='vit_b16_ep10_ctx_batch4',
        seeds=[1, 2, 3],
        loadep=-1,
        shots=16, 
        opts=[],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/ucpdept',
        result='results/ucpdept', 
        remove_dirs=['root'],
    ),
)

#####################################################


def get_config(name):
    base_cfg = copy.deepcopy(base)
    extend_cfg = copy.deepcopy(globals()[name])
    base_cfg.update(extend_cfg)
    return base_cfg
