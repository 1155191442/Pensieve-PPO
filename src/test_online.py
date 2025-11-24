import os
import sys
import time
import numpy as np
import load_trace
#import a2c as network
import ppo2 as network
import fixed_env as env

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results_on/log_sim_ppo'
TEST_TRACES = './test/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time entropy adapt_ms reward
NN_MODEL = sys.argv[1]

ONLINE_ADAPTATION = True
ONLINE_PPO_STEPS = 1
ONLINE_ADAPT_LR = 1e-5
ONLINE_TRAINABLE_ACTOR = ['ctx', 'fc4_actor', 'pi_head']
ONLINE_TRAINABLE_CRITIC = ['ctx', 'fc4_actor', 'val_head']
# ONLINE_TRAINABLE_ACTOR = ['ctx']
# ONLINE_TRAINABLE_CRITIC = ['ctx']
    
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')


    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE,
        device=DEVICE)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        actor.load_model(NN_MODEL)
        print("Testing model restored.")

    base_model_state = actor.get_network_params()

    def reset_session_model():
        actor.set_network_params(base_model_state)
        if ONLINE_ADAPTATION:
            actor.configure_trainable_params(
                actor_modules=ONLINE_TRAINABLE_ACTOR,
                critic_modules=ONLINE_TRAINABLE_CRITIC,
                lr=ONLINE_ADAPT_LR)

    reset_session_model()

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0
    pending_state = None
    pending_action_vec = None
    pending_action_prob = None
    
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        current_state = np.array(state, copy=True)
        adapt_time_ms = 0.0
        if ONLINE_ADAPTATION and pending_state is not None:
            start_time = time.perf_counter()
            actor.online_adaptation_step(
                pending_state,
                pending_action_vec,
                pending_action_prob,
                reward,
                current_state,
                end_of_video,
                num_updates=ONLINE_PPO_STEPS)
            adapt_time_ms = (time.perf_counter() - start_time) * 1000.0

        # log time_stamp, bit_rate, buffer_size, reward, adaptation time
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(entropy_) + '\t' +
                        str(adapt_time_ms) + '\t' +
                        str(reward) + '\n')
        log_file.flush()

        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob) + noise)
        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        pending_state = np.array(state, copy=True)
        pending_action_vec = action_vec.copy()
        pending_action_prob = np.array(action_prob, copy=True)
        
        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            reset_session_model()
            pending_state = None
            pending_action_vec = None
            pending_action_prob = None

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
