from textwrap import fill
import numpy as np
from utils import *

np.random.seed(111)

def make_env(args, file='dataset/data.xlsm', sku_code='SKU #26', mode='train'):
    sku_list = load_sku_data(file)
    selected_sku = choose_sku(sku_code, sku_list)
    return Environment(args, selected_sku, mode)


class Order(object):

    def __init__(self, quantity, lead_time, shelf_life):
        self.quantity = quantity
        self.lead_time = lead_time
        self.shelf_life = shelf_life

    def is_arrived(self):
        return self.lead_time == 0

    def step(self):
        assert self.lead_time > 0 and self.shelf_life > 0
        self.lead_time -= 1
        self.shelf_life -= 1


class Environment(object):

    def __init__(self, args, sku: SKU, mode='train'):
        self.mode = mode
        self.sku = sku  
        self.demand_space = 2
        self.observation_space = 3 + self.demand_space
        self.action_space = 1
        self.max_period = args.max_period
        self.min_demand = 1
        self.max_demand = 100
        self.min_lead_time = 0
        self.max_lead_time = 3
        self.max_inventory_position = self.ip_max()

        print("MIP = {}, unit holding = {}, unit ordering = {}, unit disposal = {}, unit shortage = {}, transport = {}".format(
            self.max_inventory_position, self.sku.holding_cost, self.sku.suppliers[0].unit_cost, self.sku.disposal_cost, self.sku.shortage_cost, self.sku.suppliers[0].transport_cost
        ))

        # Need to reset when a new episode begins
        self.inventory = None
        self.in_transit = None
        self.s_t = None
        self.current_period = 0
        self.unmet_demand = 0.

        self.test_case = []
        for i in range(0, self.max_period):
            self.test_case.append(np.random.randint(low=self.min_demand, high=self.max_demand))

    def ip_max(self):
        mean_lead_time = (self.max_lead_time + self.min_lead_time) / 2
        mean_demand = (self.max_demand + self.min_demand) / 2.
        return 2 * mean_lead_time * self.max_demand * \
            (self.max_lead_time * self.max_demand - mean_lead_time * mean_demand)

    def reset(self):
        self.current_period = 0
        self.unmet_demand = 0.
        self.inventory = np.zeros(shape=self.sku.shelf_life+1, dtype=np.float32)    # already in the inventory
        self.in_transit = []
        self.s_t = np.zeros(shape=self.observation_space, dtype=np.float32)
        return self.s_t

    def place_order(self, a_t):
        order_quantity = int(a_t * self.max_inventory_position)
        if order_quantity != 0:
            lead_time = np.random.randint(low=self.min_lead_time, high=self.max_lead_time)
            order = Order(order_quantity, lead_time, self.sku.shelf_life)
            self.in_transit.append(order)
        return order_quantity

    def receive_order(self):
        receive_quantity = 0
        for order in self.in_transit[:]:
            if order.is_arrived():
                self.inventory[order.shelf_life] += order.quantity
                receive_quantity += order.quantity
                self.in_transit.remove(order)
        return receive_quantity

    def meet_demand(self):
        if self.mode == 'train':
            demand = np.random.randint(low=self.min_demand, high=self.max_demand)
        else:
            demand = self.test_case[self.current_period]
        filled_demand = 0
        self.unmet_demand = demand
        for i in range(1, self.sku.shelf_life+1):
            if self.unmet_demand < self.inventory[i]:
                self.inventory[i] -= self.unmet_demand
                filled_demand += self.unmet_demand
                self.unmet_demand = 0.
                break
            else:
                self.unmet_demand -= self.inventory[i]
                filled_demand += self.inventory[i]
                self.inventory[i] = 0.
        # fill_rate = filled_demand / demand
        # fill_rate = 1 if fill_rate > 1 else fill_rate
        return demand, int(filled_demand)

    def _step(self, demand):
        self.inventory[:-1], self.inventory[-1] = self.inventory[1:], 0
        for order in self.in_transit:
            order.step()

        s_t = np.zeros(shape=self.observation_space, dtype=np.float32)
        for i in range(1, self.sku.shelf_life+1):
            s_t[0] += self.inventory[i]
            s_t[1] += i * self.inventory[i]

        for order in self.in_transit:
            s_t[0] += order.quantity
            s_t[1] += order.shelf_life * order.quantity

        s_t[0] = s_t[0] / self.max_inventory_position
        s_t[1] = s_t[1] / (self.sku.shelf_life * self.max_inventory_position)
        s_t[2] = -self.unmet_demand / self.max_inventory_position
        s_t[3:-1], s_t[-1] = self.s_t[4:], (demand - self.min_demand) / (self.max_demand - self.min_demand)

        self.current_period += 1
        done = 1 if self.current_period == self.max_period else 0

        return s_t, done

    def step(self, a_t):
        a_t = a_t[0]
        if a_t > 1 / self.max_inventory_position:
            order_quantity = self.place_order(a_t)
            order_cost = order_quantity * self.sku.suppliers[0].unit_cost
            transportation_cost = self.sku.suppliers[0].transport_cost
        else:
            order_quantity = 0
            order_cost = transportation_cost = 0.
        receive_unit = self.receive_order()
        demand, filled_demand = self.meet_demand()

        holding_unit = int(max(np.sum(self.inventory[1:], axis=0).squeeze(0), 0.0))
        disposal_unit = int(self.inventory[0])
        shortage_unit = int(self.unmet_demand)

        disposal_cost = disposal_unit * self.sku.disposal_cost
        holding_cost = holding_unit * self.sku.holding_cost
        shortage_cost = shortage_unit * self.sku.shortage_cost
        r = -(disposal_cost + order_cost + holding_cost + transportation_cost + shortage_cost) / 1e8

        self.s_t, done = self._step(demand)
        cost = [holding_cost, disposal_cost, order_cost, transportation_cost, shortage_cost]
        unit = [holding_unit, disposal_unit, order_quantity, shortage_unit, receive_unit]
        return self.s_t, r, done, cost, unit, demand, filled_demand
