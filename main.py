import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from ppo import PPOAgent
from env import *

result = {}

def train(args):

    env = make_env(args)
    agent = PPOAgent(env.observation_space, env.action_space, memory=Memory(), args=args)
    
    result['setting'] = {
        'max_ip': env.max_inventory_position,
        'period_num': env.max_period,
        'unit_ordering_cost': env.sku.suppliers[0].unit_cost,
        'fixed_ordering_cost': env.sku.suppliers[0].transport_cost,
        'unit_holding_cost': env.sku.holding_cost,
        'unit_disposal_cost': env.sku.disposal_cost,
        'unit_shortage_cost': env.sku.shortage_cost,
        'mode': 'Lost-Sales',
    }

    print('----------- Training -----------')
    cur_best = None
    running_reward = 0
    H = D = O = T = S = 0
    rl4ls = ibfa = ga = ql4ls = pg4ls = False
    target = 1

    for episode in range(1, args.max_episodes + 1):

        state = env.reset()
        while True:
            action = agent.select_action(state)
            state, reward, done, cost, _, _, _ = env.step(action)
            agent.memory.rewards.append(reward)
            agent.memory.done.append(done)
            running_reward += reward
            h, d, o, t, s = cost
            H += h
            D += d
            O += o
            T += t
            S += s
            if done:
                break

        if episode % args.update_per_episodes == 0:
            agent.update()
            agent.memory.clear()

        # if episode % args.save_per_episodes == 0:
        #     torch.save(agent.policy_old.state_dict(), './output/PPOAgent_{}.pth'.format(episode))

        if episode % args.log_per_episodes == 0:
            running_reward /= args.log_per_episodes
            running_reward *= -1
            H /= args.log_per_episodes
            D /= args.log_per_episodes
            T /= args.log_per_episodes
            O /= args.log_per_episodes
            S /= args.log_per_episodes

            if cur_best is None or running_reward < cur_best:
                cur_best = running_reward
                print('CurBest at Episode {}. Average reward = {:.3f}'.format(episode, cur_best))
                torch.save(agent.policy_old.state_dict(), './output/PPOAgent_best_' + str(args.max_period) + '.pth')

            print('Episode {} \t Avg reward: {:.3f}\t H = {:.0f}\t D = {:.0f}\t O = {:.0f}\t T = {:.0f}\t S = {:.0f}'.format(episode, running_reward, H, D, O, T, S))
            running_reward = 0
            H = 0
            D = 0
            T = 0
            O = 0
            S = 0



    with open('result.json', 'w') as f:
        json.dump(result, f, indent=4)



def eval(args):
    env = make_env(args)
    agent = PPOAgent(env.observation_space, env.action_space, memory=Memory(), args=args)
    agent.policy_old.load_state_dict(torch.load('./output/PPOAgent_best_' + str(args.max_period) + '.pth'))
    agent.policy_old.eval()

    print('\n----------- Evaluating -----------')
    H, D, O, T, S = 0, 0, 0, 0, 0
    for _ in range(args.eval_num):
        state = env.reset()
        acc_r = 0
        h, d, o, t, s = 0, 0, 0, 0, 0
        
        while True:
            act = agent.select_action(state)
            state, r, done, cost, _, _, _ = env.step(act)
            h_cost, d_cost, o_cost, t_cost, s_cost = cost
            acc_r += r
            h += h_cost
            d += d_cost
            o += o_cost 
            t += t_cost
            s += s_cost
            if done:
                break
        
        print('Reward: {:.3f}\t Hold = {:.0f}\t Dispose = {:.0f}\t Order = {:.0f}\t Trans = {:.0f}\t Short = {:.0f}'.format(acc_r, h, d, o, t, s))
        H += h
        O += o
        D += d
        T += t
        S += s

    H /= args.eval_num
    D /= args.eval_num
    O /= args.eval_num
    S /= args.eval_num
    T /= args.eval_num
    R = (H + D + O + T + S)

    print('\nTotal: {:.0f}\t Hold = {:.0f}\t Dispose = {:.0f}\t Order = {:.0f}\t Trans = {:.0f}\t Short = {:.0f}'.format(R, H, D, O, T, S))
    

def test(alg, lb, ub, args):

    env = make_env(args, 'test')
    agent = PPOAgent(env.observation_space, env.action_space, memory=Memory(), args=args)
    agent.policy_old.load_state_dict(torch.load('./output/PPOAgent_best.pth'))
    agent.policy_old.eval()

    print('\n----------- Testing -----------')

    result[alg] = {}
    period = 1
    running_reward = 0
    state = env.reset()
    result[alg]['Results'] = []
    while True:
        before = np.sum(env.inventory[1:], axis=0).squeeze(0)
        action = agent.select_action(state)
        state, reward, done, cost, unit, demand, filled_demand = env.step(action)
        after = np.sum(env.inventory[1:], axis=0).squeeze(0)
        h_cost, d_cost, o_cost, t_cost, s_cost = cost
        h_unit, d_unit, o_unit, s_unit, r_unit = unit
        # print(action * env.max_inventory_position, state, reward)
        result[alg]['Results'].append({
            '#': period,
            'inventory_level (before)': int(before),
            'ordering (qty)': o_unit,
            'ordering (cost)': o_cost + t_cost,
            'receive (qty)': r_unit,
            'demand': demand, 
            'filled_demand': filled_demand,
            'holding (qty)': h_unit,
            'holding (cost)': h_cost,
            'disposal (qty)': d_unit,
            'disposal (cost)': d_cost,
            'shortage (qty)': s_unit,
            'shortage (cost)': s_cost,
            'inventory_level (after)' : int(after),
        })
        agent.memory.rewards.append(reward)
        agent.memory.done.append(done)
        running_reward += reward
        period += 1
        if done:
            break
    print('Reward: {:.3f}'.format(running_reward))
    running_reward *= -1
    return running_reward > lb and running_reward < ub




def main():
    torch.manual_seed(111)
    parser = argparse.ArgumentParser(description='PyTorch on WP2.2')
    parser.add_argument('--max_episodes', default=1000, type=int)
    parser.add_argument('--update_per_episodes', default=10, type=int)
    parser.add_argument('--save_per_episodes', default=500, type=int)
    parser.add_argument('--log_per_episodes', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_updates', default=10, type=int)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--eval_num', default=100, type=int)
    parser.add_argument('--max_period', default=12, type=int)
    parser.add_argument('--device', default='0', type=str)
    args = parser.parse_args()

    train(args)
    eval(args)


if __name__ == '__main__':
    main()

