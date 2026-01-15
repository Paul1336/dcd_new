class LearnabilitySampler(object):

    def __init__(self, 
                 venv,
                 learnability_alpha=0.5,
                 learnability_c=0.0,
                 top_k_to_sample_uniformly=-1,
                 staleness=0.1,
                 ued_algo=None
                 ):
        self.task_info_dict = {}
        self.learnability_alpha = learnability_alpha
        self.learnability_c = learnability_c
        self.top_k_to_sample_uniformly = top_k_to_sample_uniformly
        self.staleness = staleness
        self.ued_algo = ued_algo

        if ued_algo == 'old_sfl':
            self.env_names = []
        else:
            self.env_names = venv.remote_attr('subsampled_env_ids', index=[0])[0][0]

        print('Learnability Sampler: First 10 env_names: ', self.env_names[:10])
        print('Learnability Sampler: len(env_names): ', len(self.env_names))

        self.learnability_last_updated_global_step = -1
        self.task_info_dict = {
            env_id: {
                'zero_shot_success_rate': 0.0,
                'last_updated_global_step_for_learnability': 0
            }
            for env_id in self.env_names
        }

    def update_learnability(self, env_id, global_step, success_rate):
        if env_id not in self.task_info_dict:
            raise ValueError(f"Env {env_id} not found in learnability sampler")
        
        print('update learnability for env_id: ', env_id, 'with success_rate: ', success_rate)
        self.task_info_dict[env_id] = {
            'zero_shot_success_rate': success_rate,
            'last_updated_global_step_for_learnability': global_step
        }

        self.learnability_last_updated_global_step = global_step

    def update_env_names(self, env_names):
        self.env_names = env_names
        print('update env_names: ', self.env_names)

    def wrap_level_result(self, env_name):
        import json

        # print('sampled level success_rate : ', self.task_info_dict[level]['zero_shot_success_rate'])
        return  json.dumps(PARAS[env_name]) + '@@' + str(env_name)

    def sample(self):
        if self.ued_algo == 'old_sfl':
            return self.wrap_level_result(random.choice(self.env_names))
        
        epsilon = 1e-6
        if self.learnability_alpha is None:
            raise ValueError("learnability_alpha must be set for beta priority")

        # Only look for those learnability last updated is the previous global step
        sampled_env_ids = [
            env_id
            for env_id in self.env_names
            if self.task_info_dict[env_id][
                "last_updated_global_step_for_learnability"
            ]
            == self.learnability_last_updated_global_step
        ]

        task_priorities = [
            (
                self.task_info_dict[env_id]["zero_shot_success_rate"]
                + self.learnability_c
            )
            ** (self.learnability_alpha)
            * (
                1
                - self.task_info_dict[env_id]["zero_shot_success_rate"]
                + self.learnability_c
            )
            ** (1 - self.learnability_alpha)
            + epsilon
            for env_id in sampled_env_ids
        ]

        is_stale = random.random() < self.staleness
        if is_stale:
            return self.wrap_level_result(random.choice(self.env_names))
        else:
            if self.top_k_to_sample_uniformly > 0:
                top_k_env_ids = sorted(
                    sampled_env_ids,
                    key=lambda x: task_priorities[sampled_env_ids.index(x)],
                    reverse=True,
                )[: self.top_k_to_sample_uniformly]
                return self.wrap_level_result(random.choice(top_k_env_ids))
            else:
                return self.wrap_level_result(random.choices(sampled_env_ids, task_priorities, k=1)[0])
     