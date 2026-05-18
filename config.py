
class SequenceConfig:
    def __init__(self, args):

        # Network 
        self.latent_dimension = 512
        self.num_transformer_blocks = 2
        self.num_heads = 16
        self.dropout = 0.0
        self.esm_emb_size = 960 

         # Environment  
        self.residue_vocabulary = {  
            "A":  {"allowed": True},
            "R":  {"allowed": True},
            "N":  {"allowed": True},
            "D":  {"allowed": True},
            "C":  {"allowed": True},
            "E":  {"allowed": True},
            "Q":  {"allowed": True},
            "G":  {"allowed": True},
            "H":  {"allowed": True},
            "I":  {"allowed": True},
            "L":  {"allowed": True},
            "K":  {"allowed": True},
            "M":  {"allowed": True},
            "F":  {"allowed": True},
            "P":  {"allowed": True},
            "S":  {"allowed": True},
            "T":  {"allowed": True},
            "W":  {"allowed": True},
            "Y":  {"allowed": True},
            "V":  {"allowed": True}
        }

        self.seed = args.seed # set a random seed number  
        self.training_device = args.device # set device either as cuda:gpu_num ('cuda:0') or cpu 
        self.noise_mode = args.noise_mode # set True to run the active learning under noisy proxy model
        self.noise_level = args.noise_level  # set to either -5, -15, -25 to reproduce results from the paper 
        self.low_data_setting = args.low_data_setting # set True to run active learning to train proxy under low data mode
        self.low_data_perc = args.low_data_perc # set to either 0.1, 0.2, 0.5 to use only 10%, 20%, or 50% of the available data for training proxy

        self.max_limit_pos_change = 1
        self.num_predictor_workers = 1 
        self.oracle_path = './objectives'
        self.active_learn_cycles = 10 # number of active learning rounds 
        self.multiplier = 5
        self.min_max_mutations= [1, 3]

        # Training for policy 
        self.num_dataloader_workers = 3  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.batch_size_training = 16
        self.num_batches_per_epoch = None  # Can be None, then we just do one pass through generated dataset

        # Optimizer for policy 
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 10,
                "decay_factor": 0.8
            }
        }

        self.log_to_file = True
        
        # Self-improvement sequence decoding
        self.self_improvement_learning = {
            "max_oracle_calls_per_round": 128,
            "num_trajectories_to_keep": 50, # Number of trajectories with the the highest objective function evaluation to keep for training
            "keep_intermediate_trajectories": False,
            "devices_for_workers": [f"{self.training_device}"] * 1,
            "batch_size_per_worker": 3, 
            "batch_size_per_cpu_worker": 3,
            "search_type": "wor",
            "beam_width": 32,
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # when True, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False, 
            "num_traj_test": 100, 
        }

        # Surrogate training arguments
        self.proxy_config = {
        "num_model_max_epochs": 3000,
        "ensemble_size": 3,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "patience": 10,
        "epochs_per_valid": 1,
        "proxy_batch_size": 256, 
        "dataset_path": './datasets'
        }

        # protein sequence related information
        AV2_WT= ("MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLD"
        "KGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQ"
        "AKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDAD"
        "SVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVI"
        "TTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLI"
        "NNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQG"
        "CLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPF"
        "HSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPG"
        "PCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVL"
        "IFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGV"
        "LPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTT"
        "FSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVY"
        "SEPRPIGTRYLTRNL")

        self.tasks_configs = {
            "task": args.task, 
            "wt_sequences": {"E4B": "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLPSGVTVDRSIILRHLLNSPTDPFTRQMLTESMLEPVPELKERIQAWMREKQSSDH",
                            "AMIE": "MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNARKIAEMIVGMKQGLPGMDLVVFPEYSLQGIMYDPAEMMETAVAIPGEETEIFSRACRKANVWGVFSLTGERHEEHPRKAPYNTLVLIDNNGEIVQKYRKIIPWCPIEGWYPGGQTYVSEGPKGMKISLIICDDGNYPEIWRDCAMKGAELIVRCQGYMYPAKDQQVMMAKAMAWANNCYVAVANAAGFDGVYSYFGHSAIIGFDGRTLGECGEEEMGIQYAQLSLSQIRDARANDQSQNHLFKILHRGYSGLQASGDGDRGLAECPFEFYRTWVTDAEKARENVERLTRSTTGVAQCPVGRLPEEG", 
                            "LGK": "MPIATWTGDNVLDFTVLGLNSGTSMDGIDCALCHFYQKTPDAPMEFELLEYGEVPLAQPIKQRVMRMILEDTTSPSELSEVNVILGEHFADAVRQFAAERNVDLSTIDAIASHGQTIWLLSMPEEGQVKSALTMAEGAILASRTGITSITDFRISDQAAGRQGAPLIAFFDALLLHHPTKLRACQNIGGIANVCFIPPDVDGRRTDEYYDFDTGPGNVFIDAVVRHFTNGEQEYDKDGAMGKRGKVDQELVDDFLKMPYFQLDPPKTTGREVFRDTLAHDLIRRAEAKGLSPDDIVATTTRITAQAIVDHYRRYAPSQEIDEIFMCGGGAYNPNIVEFIQQSYPNTKIMMLDEAGVPAGAKEAITFAWQGMEALVGRSIPVPTRVETRQHYVLGKVSPGLNYRSVMKKGMAFGGDAQQLPWVSEMIVKKKGKVITNNWALEHHHHHH", 
                            "Pab1": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIANDENGKSKGFGFVQFEEEGAAKEAIDALNGMLLNGQEIYVAP", 
                            "TEM": "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAERQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW", 
                            "UBE2I": "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILDEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPSY",
                            "GFP": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT""TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIE""LKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNT""PIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                            "AAV": AV2_WT[450:540]}
                            }


