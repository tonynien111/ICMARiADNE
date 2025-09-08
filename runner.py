import torch
import ray
from model import PolicyNet
from worker import Worker
from parameter import *
from icm import ICMAgent


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)
        
        # Initialize local ICM agent if enabled
        self.local_icm_agent = None
        if USE_ICM:
            self.local_icm_agent = ICMAgent(NODE_INPUT_DIM, EMBEDDING_DIM, ICM_ACTION_DIM, self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)
    
    def set_icm_weights(self, icm_weights):
        if self.local_icm_agent is not None and icm_weights is not None:
            self.local_icm_agent.icm.load_state_dict(icm_weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        # save_img = True
        worker = Worker(self.meta_agent_id, self.local_network, episode_number, 
                       device=self.device, save_image=save_img, icm_agent=self.local_icm_agent)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])
        
        # set ICM weights if available
        if len(weights_set) > 1 and USE_ICM:
            self.set_icm_weights(weights_set[1])

        job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, info


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
