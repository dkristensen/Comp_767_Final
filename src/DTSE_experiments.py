from __future__ import absolute_import,print_function

# from model import Model, copy_model_parameters
import DTSE_model as pm
from replay_memory import ReplayMemory, Transition
import torch
import state_getter as sg
import os
import sys
import optparse
import subprocess
import random
import numpy as np
from rewards import RewardsContainer
try:
    sys.path.append(os.path.join(os.path.dirname(__file__),"tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci


time_step = 1
time_modifier = 1/time_step

start_params = None
num_actions = 4

cfg_file = "scenarios/comp767.sumocfg"#genders_net.sumocfg"#"test.sumocfg"

VALIDATION_LENGTH = 2400#0
EVALUATION_LENGTH = 2400#0#1500
ERASE_LINE = '\x1b[2K'
TRAINING_LENGTH = 7200#3600

BATCH_SIZE,DISCOUNT_FACTOR = 64,0.95


def reset_sim(params):
    try:
        traci.close()
    except:
        print("Errored")
    traci.start(params)
    sg.set_departure_times()

    # if(params[0] == checkBinary('sumo')):
    print("\033[9A\033[0J\033[1A")
    
    return sg.get_DTSE()

def reset_all_sims():
    param = [checkBinary('sumo'),#-gui
                    "-c", cfg_file,#"tiny.sumocfg",#"genders_net.sumocfg",
                    "-S","-Q","-W"]
    state = reset_sim(param)
    return state

def optimize_model(q_estimator,replay_memory,optimizer,batch_size=BATCH_SIZE,discount_factor=DISCOUNT_FACTOR):
    if len(replay_memory) < BATCH_SIZE:
        return

    transitions = replay_memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    # Get minibatch for training
    try:
        a = np.array(batch.state).shape
    except:
        for val in batch.state:
            print(np.array(val).shape,np.array(val))
            
    next_states_batch = torch.FloatTensor(batch.next_state).squeeze()
    states_batch = torch.FloatTensor(batch.state).squeeze()

    
    action_batch = torch.LongTensor(np.array(batch.action).reshape(BATCH_SIZE,1))

    reward_batch = torch.FloatTensor(np.array(batch.reward).reshape(BATCH_SIZE,1))

    # DDQN Settings
    # Compute q-values
    # for x in next_states_batch: print(x)
    # assert all(x.shape == (pm.Model.input_length) for x in next_states_batch)
    # print(q_estimator.forward(states_batch).shape,q_estimator.forward(states_batch))
    q_state_values = q_estimator.forward(states_batch).gather(1,action_batch)
    
    
    # Compute Target values
    # q_values_next_target = torch.zeros(BATCH_SIZE,1)#q_estimator.forward(next_states_batch)


    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    q_next_values = q_estimator.forward(next_states_batch).gather(1,action_batch)
    # best_next_actions = np.argmax(q_next_values.detach().numpy(), axis=1)

    discounted_future = q_next_values * DISCOUNT_FACTOR
    # Compute the expected Q values
    expected_reward_batch = discounted_future + reward_batch
    # Compute Huber loss
    loss = torch.nn.functional.smooth_l1_loss(q_state_values, expected_reward_batch)#.unsqueeze(1)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in q_estimator.model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # pm.polyak_update(from_network = q_estimator,to_network = target_estimator)  
    return loss.item()




def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def validate(q_estimator):
    try:
        traci.close()
    except:
        1
    # q_estimator.save_model()
    # q_estimator.load_model()
    rewardsContainer = RewardsContainer()
    traci.start([checkBinary('sumo-gui'), "-c", cfg_file,
                             "--tripinfo-output", "tripinfo.xml",
                             "--step-length", str(time_step),
                             "--time-to-teleport", "99999",
                             "-S",
                             "-Q","-W"])
    sg.set_departure_times()
    step = 0
    total_rewards = []
    while traci.simulation.getMinExpectedNumber() > 0:
        if(step >= VALIDATION_LENGTH): 
            traci.close()
            break
        if((step%3) == 0):
            state = sg.get_DTSE()
        
            best_action = np.argmax(q_estimator.predict(state))
            sg.set_767_action(best_action)
            traci.simulationStep()
            sg.set_departure_times()
            rewards = rewardsContainer.get_all_rewards()
            total_rewards.append(rewards)
        else:
            traci.simulationStep()
            sg.set_departure_times()
        step+=1
    return total_reward/(VALIDATION_LENGTH/3)

def evaluate(q_estimator, policy):
    try:
        traci.close()
    except:
        1
    # q_estimator.save_model()
    # q_estimator = pm.load_model(q_estimator.filename)
        
    traci.start([checkBinary('sumo-gui'), "-c", cfg_file,#-gui
                             "--tripinfo-output", "tripinfo.xml",
                             "--step-length", str(time_step),
                             "--time-to-teleport", "99999",
                             "-S","-W",
                             "-Q"])
    sg.set_departure_times()
    rewards_object = RewardsContainer()
    total_rewards = []
    step = 0
    throughput = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        if(step >= EVALUATION_LENGTH): 
            traci.close()
            print("\033[4A\033[0J\033[1A")
            break
        if(traci.simulation.getTime()%3 == 0):
            
            state = sg.get_DTSE()
            state = torch.FloatTensor(state)
            prediction = q_estimator(state)
            # print(prediction)
            # best_action = np.argmax(prediction.detach().numpy())
            best_action = policy(state)
            sg.set_767_action(best_action)
            traci.simulationStep()
            sg.set_departure_times()
            rewards = rewards_object.get_all_rewards()
            total_rewards.append(rewards)

            print(ERASE_LINE+"\r Step {} ({}) \tBest Action: {}\tPrediction Value: {}\tRewards: {}".format(step,
                                                                                EVALUATION_LENGTH,#current_time,
                                                                                best_action,
                                                                                prediction[0][best_action],
                                                                                rewards[q_estimator.reward_index]),
                                                                                end="")
            sys.stdout.flush()
        else:
            traci.simulationStep()
            sg.set_departure_times()
        throughput += traci.simulation.getArrivedNumber()
        step+=1
    return np.concatenate((np.mean(np.array(total_rewards),axis=0), [throughput]))

def reset_all():
    states = reset_all_sims()
    for i in range(len(states)):
        states[i] = np.array(states[i])
    return states

def compute_reward(old_wait,new_wait,penalties):
    return old_wait - new_wait + penalties    

def do_sumo_step(policy, state, rewardsContainer):
    # Choose our action and attempt to implement it
    action = policy(torch.FloatTensor(state))#,epsilon)
    sg.set_767_action(action)
    # Step
    traci.simulationStep()
    sg.set_departure_times()

    # Get our post-action state
    next_state = sg.get_DTSE()

    # Update our cars locations and waiting times after the action
    rewards = rewardsContainer.get_all_rewards()

    return action, next_state, rewards
  

def do_train_epoch(q_estimator, replay_memory, optimizer, policy, n_steps, training_epoch, max_epochs):
    states = reset_all()
    state = sg.get_DTSE()
    sg.set_departure_times()
        
    old_past_state, old_new_state, old_action = None,None,None
    reward, loss, action = 0, 0, 0

    g_step = int(q_estimator.get_global_step())
    # epsilon = epsilon_params["values"][min(g_step, epsilon_params["steps"]-1)]
    rewardsContainer = RewardsContainer()
    for t in range(n_steps):#4800):#t in itertools.count():
        if(traci.simulation.getTime()%3 == 0):
            
            # If at the step, update the target estimator
            g_step = q_estimator.get_global_step()
            # if( g_step % target_update == 0):
                # pm.polyak_update(from_network=q_estimator,to_network=target_estimator)
                # target_estimator.global_step += 1
            
            # Print the current step
            print(ERASE_LINE+"\r Step {:0>4d} ({}) @ Episode {:0>4d}/{}, loss: {:06.3f}, Prev Reward: {:04.4f}, Last Action: {}".format(t,
                                                                        g_step,#current_time,
                                                                        training_epoch+1,
                                                                        max_epochs,
                                                                        loss,
                                                                        reward,
                                                                        action),
                                                                        end="")
            sys.stdout.flush()
            # Perform one step and return the values
            action, next_state, rewards = do_sumo_step(policy, state,rewardsContainer)
            reward = -rewards[q_estimator.reward_index]
            # reward+=penalty
            
            # Once we have history (t>0) we can add the experience to replay memory
            if(t>0):
                replay_memory.add_sample(old_past_state,old_action,old_new_state,-reward)
            
            old_past_state, old_new_state, old_action = states, next_state, action
            # prev_pen = penalty
            state = next_state
            q_estimator.global_step += 1

            # Run optimization step
            loss = optimize_model(q_estimator,replay_memory,optimizer)
        else:
            traci.simulationStep()
            sg.set_departure_times()
    return





def populate_replay_memory(replay_memory,init_size,start_params):
    state = reset_all()
    sg.set_departure_times()
    # state = sg.get_DTSE()
    rewardsContainer = RewardsContainer()
    state = state[0]

    old_past_state, old_new_state, old_action = None, None, None
    prev_pen = 0
    traffic_light_id = sg.get_all_traffic_light_ids()[0]
    action = None
    i = 0
    while(len(replay_memory) < init_size):
        if((i%3) == 0):
            # if(int(traci.simulation.getCurrentTime()/1000)%time_step == 0):
            print(ERASE_LINE+"\r Step {} ({})\t Last Action: {}".format(len(replay_memory), init_size, action), end="")
            sys.stdout.flush()
            # print(state)
            state = sg.get_DTSE()
            prev_rewards = rewardsContainer.get_all_rewards()
            reward = -prev_rewards[q_estimator.reward_index]

            traci.simulationStep()
            sg.set_departure_times()

            action = traci.trafficlight.getPhase(traffic_light_id)//2 
            action = action if action<3 else 3
            # sg.set_767_action(action)

            next_state = sg.get_DTSE()
            
            # if(old_past_state is not None):
            #     if(old_past_state is None or old_action is None or old_new_state is None or reward is None):
            #         print(old_past_state,old_action,old_new_state,reward)
            #     # print(old_past_state,old_action,old_new_state,reward)
            #     else:
            old_past_state, old_new_state, old_action = state, next_state, action
            replay_memory.add_sample(old_past_state,old_action,old_new_state,reward)        
        else:
            traci.simulationStep()
            sg.set_departure_times()
        i+=1
        if(traci.simulation.getMinExpectedNumber() is 0):
            state = reset_all()
            state = state[0]
            rewardsContainer = RewardsContainer()
            i = 0


            
    traci.close()
    


def make_epsilon_greedy_policy(estimator, num_actions):
    """
    Creates an epsilon-greedy policy using the Q-value approximator network
    """
    def policy_fn(observation, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions) # random action
        else:
            q_values = estimator(observation)
            return np.argmax(q_values.detach().numpy())
    return policy_fn

def make_softmax_exploration_policy(estimator, num_actions):
    def policy_fn(observation):
        with torch.no_grad():
            raw_out = estimator(observation).cpu().numpy()[0]
            sm_out = np.exp(raw_out - np.max(raw_out))
            sm_out = sm_out / sm_out.sum()
            # print(raw_out,sm_out)
            action = np.random.choice(num_actions,1,p=sm_out)
            return action[0]
    return policy_fn

def deep_q_learning(q_estimator,
                    optimizer,
                    num_episodes,
                    experiment_dir = "exp",
                    replay_memory_size = 5000,
                    replay_memory_init_size = 500,
                    discount_factor = 0.99,
                    epsilon_start = 1.0,
                    epsilon_end = 0.05,
                    epsilon_decay_steps = 500000,
                    batch_size = 32):
    
    Transition = ("state", "action", "reward", "next_state", "done")

    # Create the directories for the checkpoints, summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if(not os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    validate_every = 5
    


    # Set the epsilon decay schedule
    # epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy to follow
    policy = make_softmax_exploration_policy(q_estimator, num_actions)#len(traci.trafficlight.getIDList()))

    # set up the replay memory
    replay_memory = ReplayMemory(max_size = replay_memory_size)

    num_intersections = len(sg.get_all_traffic_light_ids())

    # polyak_averaging = pm.polyak_update(from_network=q_estimator, to_network=target_estimator)

    # Populate replay memory with the initial experience 
    print("Populating replay memory...")

    populate_replay_memory(replay_memory,replay_memory_init_size,start_params)
    
    # epsilon_params = {
    #     "start":epsilon_start,
    #     "end":epsilon_end,
    #     "steps":epsilon_decay_steps,
    #     "values":epsilons
    # }

    with open("results_{}.csv".format(q_estimator.reward_index), "w") as results:
        start = int(float(q_estimator.get_global_step())/TRAINING_LENGTH)
        for i_episode in range(start,num_episodes):
            if(i_episode%validate_every == 0):
                # validate(q_estimator, sess)
                values = evaluate(q_estimator, policy)
                value_string = ''
                for val in values: 
                    value_string+=", {}".format(val)
                print(values)
                results.write(str(i_episode))
                results.write(str(value_string)+"\n")
                results.flush()
                # print(value)

                # WHERE LOGGING NEEDS TO HAPPEN
    

            do_train_epoch(q_estimator = q_estimator,
                            policy = policy,
                            # epsilon_params = epsilon_params,
                            replay_memory = replay_memory,
                            optimizer = optimizer,
                            n_steps = TRAINING_LENGTH,
                            training_epoch = i_episode,
                            max_epochs = num_episodes
                            )
            
                

    return 




#10 lines below

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    # if options.nogui:
    #     sumoBinary = checkBinary('sumo')
    # else:
    #     sumoBinary = checkBinary('sumo-gui')
    # sumoBinary = checkBinary('sumo')
    sumoBinary = checkBinary('sumo')

    # first, generate the route file for this simulation
    # gen_routes_file()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    start_params = [sumoBinary,
                    "-c", cfg_file,
                    "--tripinfo-output", "tripinfo.xml",
                    "--step-length", str(time_step),
                    "-S",
                    "-Q"
                    ]
    reset_sim(start_params)


    num_intersections = len(traci.trafficlight.getIDList())
    # q_session = tf.Session()
    
    for reward_index in range(12):

        q_params = {
            "cuda":False,
            "filename":"q_network",
            "reward_index":reward_index
        }
        # pm.saveJson("{}/{}".format(pm.Model.model_path_prefix,q_params["filename"]), q_params)

        q_estimator = pm.Model(q_params)
        # if(os.path.isfile("{}/{}/model_weights.pt".format(pm.Model.model_path_prefix,q_params["filename"]))):
        #     q_estimator = pm.load_model(q_estimator.filename)
        optimizer = torch.optim.RMSprop(q_estimator.model.parameters(),lr=q_estimator.lr)
        print("through model")

        # q_estimator.load_model()
        # target_estimator.load_model()
        # print(evaluate(q_estimator))
        # sys.exit()
        deep_q_learning(q_estimator,
                        optimizer,
                        num_episodes=150,
                        experiment_dir="exp",
                        replay_memory_size=32000,
                        replay_memory_init_size=1000,#5000,#25000,
                        discount_factor = 0.7,
                        epsilon_start = 1.0,
                        epsilon_end=0.1,
                        epsilon_decay_steps=25000,#500*TRAINING_LENGTH,
                        batch_size = 64)
            # print("\nEpisode Reward: {}".format(stats.episode_reward[-1]))