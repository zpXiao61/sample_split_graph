import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import sys

def convert_pbtxt_str_to_pb(sss,path,pb_name):
    graph_def = tf.GraphDef()
    text_format.Merge(sss, graph_def)
    tf.train.write_graph(graph_def, path, pb_name, as_text=False)
    return

def is_in_these_scopes(op,scopes=None,names=None,optypes=None):
    if names is not None:
        for n in names:
            if n == op.name:
                return True
    if scopes is not None:
        for s in scopes:
            if op.name.find(s) == 0:
                return True
    if optypes is not None:
        for t in optypes:
            if t == op.type:
                return True
    return False

def has(arr, ele):
    for tmp in arr:
        if (tmp == ele):
            return True
    return False

def node_def_add_node(sss):
    sss = sss.replace('\n', '\n\t')
    sss = "node {\n\t" + sss
    sss = sss[0:len(sss)-1] + '}\n'
    return sss
    
    sss = sss.strip()

    ssss = sss.split('\n')
    sssss = []
    for line in ssss:
        if line.find('input: "^') > -1:
            continue
        else:
            sssss.append(line)
    new_sss = ''
    for line in sssss:
        new_sss = new_sss+line+'\n'

    return new_sss

def is_op_name_in_ops(name, ops):
    for op in ops:
        if name == op.name:
            return True
    return False

def is_op_input_in_ops(op, ops):
    op_names = []
    
    #don't need control dependencies
    #for ci in op.control_inputs:
        #op_names.append(ci.name)
    for i in op.inputs:
        op_names.append(i.op.name)

    for name in op_names:
        for oo in ops:
            if (name == oo.name):
                return True

    return False

def ret_op_type_in_graph(op, ops):
    '''
    :param op:
    :param ops:
    :return:
     0: an output
     1: just control_dependency
     2: Not just control_dependency
    '''

    None

def is_op_an_input_of_ops(op,ops):
    for o in ops:
        for n in o.inputs:
            if n.op.name == op.name:
                return True
    return False

def is_a_const_or_sth(name, ops):
    types = ['Const']
    for op in ops:
        if name == op.name:
            for t in types:
                if op.type == t:
                    return True
                else:
                    return False
    return False

# return has?, output num, first outdtype
def is_opname_in_ops(name,ops):
    for op in ops:
        if op.name == name:
            if len(op.outputs) > 0:
                return True,len(op.outputs),op.outputs[0].dtype,op.type,op.node_def
            else:
                return True,len(op.outputs),None,op.type,op.node_def
    return False,None,None,None,None

'''
return placeholder names

path_input_pb: path of input pb file to be splited
output_folder: output folder to save splited pb1 and pb2
name_pb1: splited pb1
name_pb2: splited pb2

'''
def split_graph(path_input_pb,output_folder,name_pb1,name_pb2,scopes_pb2=None,optypes_pb2=None,opnames_pb2=None,savepbtxt=False): 
    
    savepbtxt=False # now always be false
    
    if scopes_pb2 is None and optypes_pb2 is None and opnames_pb2 is None:
        print('No split!!!')
        return None
    
    ops1 = []
    ops2 = []
    
######################################################################################################
    print('Load graph: ',path_input_pb)
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_input_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        ops = detection_graph.get_operations()
        
        
        # all unimplementing op in ops2, others in ops1
        for op in ops:
            if is_in_these_scopes(op,scopes=scopes_pb2,optypes=optypes_pb2,names=opnames_pb2):
                ops2.append(op)
            else:
                ops1.append(op)
        #################################################################################################
        print('ops1 and ops2 initialization done.')
        
        # move op in ops1 to ops2 which has inputs(dont need control dependencies) in ops2
        to_remove_in_ops1 = []
        new_add = True
        while(new_add):
            new_add = False
            for op in ops1:
                if (is_op_input_in_ops(op, ops2)):
                    ops2.append(op)
                    to_remove_in_ops1.append(op)
                    print(op.name)
                    new_add = True
            for op in to_remove_in_ops1:
                ops1.remove(op)
            to_remove_in_ops1 = []
    
        #ensure ops2's inputs ops which are in ops1 are all with len(op.outputs)==1
        need_add = True
        while(need_add):
            need_add = False
            to_add_ops = set()
            for oo in ops2:
                for ooi in oo.inputs:
                    has,num,outdtype,_3,_4 = is_opname_in_ops(ooi.op.name,ops1)
                    if has and num > 1:
                        to_add_ops.add(ooi.op.name)
                        print('Ensure step: '+ooi.op.name+' to move to ops2')
                        need_add = True
            to_remove = []
            for oo in ops1:
                for name in to_add_ops:
                    if name == oo.name:
                        ops2.append(oo)
                        to_remove.append(oo)
                        break
            for oo in to_remove:
                ops1.remove(oo)
                print('Ensure step: remove '+oo.name+' from ops1')
        #####################################################################################################
        print('ops1 and ops2 update done.')
        
        #for all op in ops2
        #if its inputs not in ops2,
        #  check it in ops1, and add to placeholders
        #if its control_inputs not in ops2,
        #  add to consts
        # 3 kind of types
        # not const, need a placeholder
        # a const ,can create a new same const
        # control like assert use a random const replace
        ops_placeholder_name = []
        ops_placeholder_dtype = []
        ops_const_name = []
        
        ops_true_const_node_def = []
        
        name_inputs = set()
        name_control_inputs = set()
        
        for op in ops2:
            for ci in op.control_inputs:
                name_control_inputs.add(ci.name)
            for t in op.inputs:
                name_inputs.add(t.op.name)
        
        ##########################################################################################
        print('Adding control_input as constant in pb2.')
        
        for opname in name_control_inputs:
            _0,_1,_2,_3,_4 = is_opname_in_ops(opname,ops2)
            if _0:
                continue
            else:
                ops_const_name.append(opname)
        
        ############################################################################################
        
        print('Adding inputs for pb2: copy constants for true const inputs and create pleceholders for none-const inputs.')
        
        for opname in name_inputs:
            _0,_1,_2,_3,_4 = is_opname_in_ops(opname,ops2)
    
            if _0:
                continue
            else:
                ophas,outnum,outdtype,optype,op_def = is_opname_in_ops(opname,ops1)
                # perhaps a const output to both ops1 and ops2, so dont't move ops2' const inputs(in ops1) to ops2
                if outnum != 1:
                    print(opname+' in ops1, outnum > 1!')
                    print('outnum: '+str(outnum))
                    import sys
                    sys.exit(-1)   
                if ophas:
                    if optype == 'Const':
                        ops_true_const_node_def.append(op_def)
                        print('Add true const: '+opname)
                    else:
                        ops_placeholder_name.append(opname)
                        ops_placeholder_dtype.append(outdtype)
                        print('Add placeholder: '+opname)
                else:
                    print(opname+' not in ops1!')
                    import sys
                    sys.exit(-1)
        #########################################################################
    
        #for all op in ops1
        #if its inputs not in ops1,
        #  exit(-1)
        #if its control_inputs not in ops1,
        #  add to consts

        ops_const_name_1 = []
        
        name_inputs_1 = set()
        name_control_inputs_1 = set()
        
        for op in ops1:
            for ci in op.control_inputs:
                name_control_inputs_1.add(ci.name)
            for t in op.inputs:
                name_inputs_1.add(t.op.name)
        for opname in name_inputs_1:
            _0,_1,_2,_3,_4 = is_opname_in_ops(opname,ops1)
            if _0 is False:
                print("wrong: " + str(opname) + " is not in ops1.\nexit.")
                sys.exit(-1)
        ##########################################################################################
        print('Adding control_input as constant in pb1.')
        
        for opname in name_control_inputs_1:
            _0,_1,_2,_3,_4 = is_opname_in_ops(opname,ops1)
            if _0:
                continue
            else:
                ops_const_name_1.append(opname)
        
############################################################################################
    
    # add consts for ops1
    # control like assert use a random const replace
    with tf.Graph().as_default():
        for cs in ops_const_name_1:
            #controls
            ops1.append(tf.constant(1.0,dtype=tf.float32,name=cs).op)
            print('constant added for ops1: ')
            print('\tscope name: '+cs)
    
    # add placeholders and consts for ops2
    # 3 kind of types
    # not const, need a placeholder
    # a const ,can create a new same const
    # control like assert use a random const replace
    with tf.Graph().as_default():
        for tmptype,tmpname in zip(ops_placeholder_dtype,ops_placeholder_name):
            ops2.append(tf.placeholder(tmptype, shape=None, name=tmpname).op)
            print('placeholder added: ')
            print('\tdata type: '+str(tmptype))
            print('\tscope name: '+tmpname)
        for cs in ops_const_name:
            #controls
            has = False
            for nn in ops_placeholder_name:
                  if cs == nn:
                      has = True
            if has:
                print('In add const: opname '+cs+' already in placeholders.')
                continue
            else:
                ops2.append(tf.constant(1.0,dtype=tf.float32,name=cs).op)
                print('constant added for ops2: ')
                print('\tscope name: '+cs)
    
    
    
    #########################################################################
    
    
    print('split done, follwed with saving.')
    '''
    with open(PATH_PLACEHOLDERS, 'w+') as plhfile:
        for name in ops_placeholder_name:
            plhfile.write(name+'\n')
    
    
    with open(PATH_OPS1, 'w+') as opsfile1:
        for o1 in ops1:
            #print(o1.name)
            sss = node_def_add_node(str(o1.node_def))
            opsfile1.write(sss)
    
    with open(PATH_OPS2, 'w+') as opsfile2:
        for true_const_def in ops_true_const_node_def:
            sss = node_def_add_node(str(true_const_def))
            opsfile2.write(sss)
        for o2 in ops2:
            sss = node_def_add_node(str(o2.node_def))
            opsfile2.write(sss)
    '''
    sss1 = ''
    for o1 in ops1:
        sss = node_def_add_node(str(o1.node_def))
        sss1 = sss1 + sss
    sss2 = ''
    for true_const_def in ops_true_const_node_def:
        sss = node_def_add_node(str(true_const_def))
        sss2 = sss2 + sss
    for o2 in ops2:
        sss = node_def_add_node(str(o2.node_def))
        sss2 = sss2 + sss
    
    if savepbtxt:
        import os
        name_pbtxt1 = name_pb1 + 'txt'
        name_pbtxt2 = name_pb2 + 'txt'
        with open(os.path.join(output_folder+name_pbtxt1), 'w+') as opsfile1:
            opsfile1.write(sss1)
            print('save pbtxt1: ', os.path.join(output_folder+name_pbtxt1))
        with open(os.path.join(output_folder+name_pbtxt2), 'w+') as opsfile2:
            opsfile2.write(sss2)
            print('save pbtxt2: ', os.path.join(output_folder+name_pbtxt2))
    ##########################################################################
    print('start saving pb1 and pb2.')
    
    convert_pbtxt_str_to_pb(sss1,output_folder,name_pb1)
    convert_pbtxt_str_to_pb(sss2,output_folder,name_pb2)
    
    print('saving pb1 and pb2 done.')
    
    return ops_placeholder_name