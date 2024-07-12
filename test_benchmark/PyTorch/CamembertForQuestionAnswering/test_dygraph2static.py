import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle
import pickle
paddle.enable_static()
exe = fluid.Executor(fluid.CPUPlace())
[prog, feed, fetchs] = fluid.io.load_inference_model(dirname="pd_model/inference_model/", 
                                                        executor=exe, 
                                                        model_filename="model.pdmodel",
                                                        params_filename="model.pdiparams")
def append_fetch_ops(program, fetch_target_names, fetch_holder_name='fetch'):
    """
    In this palce, we will add the fetch op
    """
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    print("the len of fetch_target_names:%d" % (len(fetch_target_names)))
    for i, name in enumerate(fetch_target_names):

        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})
        
def insert_fetch(program, fetchs, fetch_holder_name="fetch"):
    global_block = program.global_block()
    need_to_remove_op_index = list()
    for i, op in enumerate(global_block.ops):
        if op.type == 'fetch':
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    append_fetch_ops(program, fetchs, fetch_holder_name)
    
    
check_name = 'assign_0.tmp_0'
insert_fetch(prog, [check_name])
with open('../dataset/CamembertForQuestionAnswering/pytorch_input.pkl', 'rb') as inp:
    a = pickle.load(inp)
tmp_tensor = prog.global_block().var(check_name)
fetchs = [tmp_tensor]
result = exe.run(prog, feed={'x0': a["input_ids"], 'x1': a["attention_mask"]}, fetch_list=fetchs)
print(result[0])
