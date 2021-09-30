import math

'''
This is the implementation of Algorithm 4 in Box2 of QuantumFlow paper at NCOMM (Weiwen)
'''

def print_info():
    print("This is in qf_map")

def change_sign(sign, bin):
    one_positions = []
    try:
        beg_pos = 0
        while True:
            find_pos = bin.index("1", beg_pos)
            one_positions.append(find_pos)
            beg_pos = find_pos + 1
    except Exception as exception:
        # print("Not Found")
        pass
    for k, v in sign.items():
        change = True
        for pos in one_positions:
            if k[pos] == "0":
                change = False
                break
        if change:
            sign[k] = -1 * v


def find_start(affect_count_table, target_num):
    for k in list(affect_count_table.keys())[::-1]:
        if target_num <= k:
            return k

def recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates):
    if start_point == target_num:
        # print("recursive_change: STOP")
        return

    gap = int(math.fabs(start_point - target_num))
    step = find_start(affect_count_table, gap)
    change_sign(sign, affect_count_table[step])
    quantum_gates.append(affect_count_table[step])

    if direction == "r":
        # print("recursive_change: From",start_point,"Right(-):",step)
        start_point = start_point - step
        direction = "l"
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

    else:
        # print("recursive_change: From",start_point,"Left(+):",step)
        start_point = start_point + step
        direction = "r"
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

def Mapping_U_LYR(sign, target_num, digits):
    flag = "0" + str(digits) + "b"
    pre_num = 0
    affect_count_table = {}
    quantum_gates = []
    for i in range(digits):
        cur_num = pre_num + pow(2, i)
        pre_num = cur_num
        binstr_cur_num = format(cur_num, flag)
        affect_count_table[int(pow(2, binstr_cur_num.count("0")))] = binstr_cur_num

    if target_num in affect_count_table.keys():
        quantum_gates.append(affect_count_table[target_num])
        change_sign(sign, affect_count_table[target_num])

    else:
        direction = "r"
        start_point = find_start(affect_count_table, target_num)
        quantum_gates.append(affect_count_table[start_point])
        change_sign(sign, affect_count_table[start_point])
        recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

    return quantum_gates
