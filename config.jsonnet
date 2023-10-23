//local batch = 128;
//local dim = 10;
//local lr = 0.001;
//local num_epochs = 300;
local dim = std.parseInt(std.extVar('dim'));
local batch = std.parseInt(std.extVar('batch'));
local lr = std.parseJson(std.extVar('lr'));
local num_epochs = std.parseInt(std.extVar('num_epochs'));
{
    "dataset_reader" : {
        "type": "modules.readers.ZH_POS_Reader",
        "max_instances": 1000
    },
    "train_data_path": "/media/sf_M1/S2/MethodeApprentissageAuto/_projet/data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu",
    "validation_data_path": "/media/sf_M1/S2/MethodeApprentissageAuto/_projet/data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.conllu",
    "model": {
        "type": "modules.models.POS_ZH_Model",
        "dim": dim
    },
    "data_loader": {
        "batch_size": batch,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
          lr: lr,
          type: 'adam',
        },
        "num_epochs": num_epochs,
        "patience": 20,
        "validation_metric":'+acc',
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "mode": "max",
            "threshold_mode": "abs",
            "factor":0.5,
            "patience": 3,
            "verbose":true,
            "threshold": 0.01
        }
    }
}
