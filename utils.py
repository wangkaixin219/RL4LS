from cmath import nan
import torch
import torch.nn as nn
from torch.autograd import Variable
import xlrd
from collections import namedtuple

SKU = namedtuple('SKU', ['code', 'suppliers', 'unit', 'holding_cost', 'disposal_cost', 'shortage_cost', 'shelf_life'])
Supplier = namedtuple('Supplier', ['sku_code', 'supplier_name', 'mode', 'min_lead_time', 'max_lead_time',
                                   'min_order', 'max_order', 'unit_cost', 'transport_cost'])

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.done = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.done[:]


def hard_update(target: nn.Module, src: nn.Module):
    for target_params, src_params in zip(target.parameters(), src.parameters()):
        target_params.data.copy_(src_params.data)


def soft_update(target: nn.Module, src: nn.Module, tau):
    for target_params, src_params in zip(target.parameters(), src.parameters()):
        target_params.data.copy_(target_params.data * (1.0 - tau) + src_params.data * tau)


def to_numpy(var: torch.Tensor):
    return var.data.numpy()


def to_tensor(var, requires_grad=False):
    return Variable(torch.from_numpy(var), requires_grad=requires_grad)


def choose_sku(sku_code, sku_list):
    selected_sku = None
    for sku in sku_list:
        if sku_code == sku.code:
            selected_sku = sku
            break
    return selected_sku


def load_sku_data(file):
    data = xlrd.open_workbook(file)

    # Read supplier attributions
    supply_lead_time_table = data.sheet_by_name('Supply Lead-time')
    supplier_list = []
    row_start_idx = 6
    row_end_idx = supply_lead_time_table.nrows
    for row in range(row_start_idx, row_end_idx):
        _, sku_code, _, supplier_name, min_order, mode, \
            unit_cost, transport_cost, _, min_lead_time, _, max_lead_time = supply_lead_time_table.row_values(row)
        supplier = Supplier(sku_code=sku_code, supplier_name=supplier_name, min_order=0, max_order=10*min_order,
                            mode=mode, min_lead_time=min_lead_time, max_lead_time=max_lead_time,
                            unit_cost=100, transport_cost=transport_cost)
        supplier_list.append(supplier)

    # Read sku attributions
    part_master_table = data.sheet_by_name('Part Master')
    sku_list = []
    row_start_idx = 6
    row_end_idx = part_master_table.nrows
    for row in range(row_start_idx, row_end_idx):
        _, code, _, _, _, _, holding_cost, shortage_cost, disposal_cost, unit, _ = part_master_table.row_values(row)
        sku = SKU(code=code, suppliers=[], unit=unit, shelf_life=6, holding_cost=100,
                  shortage_cost=50000, disposal_cost=disposal_cost)
        sku_list.append(sku)

    for sku in sku_list:
        for supplier in supplier_list:
            if sku.code == supplier.sku_code:
                sku.suppliers.append(supplier)

    return sku_list
