��	
��
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
�
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:	*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:	*
dtype0
�
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:	*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:	*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:	*
dtype0
�
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:	*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
�
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
SGD/conv3d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv3d/kernel/momentum
�
.SGD/conv3d/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d/kernel/momentum**
_output_shapes
:*
dtype0
�
SGD/conv3d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameSGD/conv3d/bias/momentum
�
,SGD/conv3d/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/conv3d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameSGD/conv3d_1/kernel/momentum
�
0SGD/conv3d_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_1/kernel/momentum**
_output_shapes
:	*
dtype0
�
SGD/conv3d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameSGD/conv3d_1/bias/momentum
�
.SGD/conv3d_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_1/bias/momentum*
_output_shapes
:	*
dtype0
�
SGD/conv3d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameSGD/conv3d_2/kernel/momentum
�
0SGD/conv3d_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_2/kernel/momentum**
_output_shapes
:	*
dtype0
�
SGD/conv3d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv3d_2/bias/momentum
�
.SGD/conv3d_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_2/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/conv3d_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameSGD/conv3d_3/kernel/momentum
�
0SGD/conv3d_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_3/kernel/momentum**
_output_shapes
:	*
dtype0
�
SGD/conv3d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameSGD/conv3d_3/bias/momentum
�
.SGD/conv3d_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_3/bias/momentum*
_output_shapes
:	*
dtype0
�
SGD/conv3d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameSGD/conv3d_4/kernel/momentum
�
0SGD/conv3d_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_4/kernel/momentum**
_output_shapes
:	*
dtype0
�
SGD/conv3d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv3d_4/bias/momentum
�
.SGD/conv3d_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_4/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/conv3d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameSGD/conv3d_5/kernel/momentum
�
0SGD/conv3d_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_5/kernel/momentum**
_output_shapes
:*
dtype0
�
SGD/conv3d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/conv3d_5/bias/momentum
�
.SGD/conv3d_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv3d_5/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
�4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�3
value�3B�3 B�3
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�
7iter
	8decay
9learning_rate
:momentummomentumhmomentumimomentumjmomentumkmomentumlmomentumm%momentumn&momentumo+momentump,momentumq1momentumr2momentums
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics

	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
!	variables
"trainable_variables
#regularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
'	variables
(trainable_variables
)regularization_losses
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
-	variables
.trainable_variables
/regularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
3	variables
4trainable_variables
5regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

c0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	dtotal
	ecount
f	variables
g	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

f	variables
��
VARIABLE_VALUESGD/conv3d/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_1/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_1/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_2/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_2/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_3/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_3/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_4/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_4/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_5/kernel/momentumYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv3d_5/bias/momentumWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*3
_output_shapes!
:���������   *
dtype0*(
shape:���������   
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_129917
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.SGD/conv3d/kernel/momentum/Read/ReadVariableOp,SGD/conv3d/bias/momentum/Read/ReadVariableOp0SGD/conv3d_1/kernel/momentum/Read/ReadVariableOp.SGD/conv3d_1/bias/momentum/Read/ReadVariableOp0SGD/conv3d_2/kernel/momentum/Read/ReadVariableOp.SGD/conv3d_2/bias/momentum/Read/ReadVariableOp0SGD/conv3d_3/kernel/momentum/Read/ReadVariableOp.SGD/conv3d_3/bias/momentum/Read/ReadVariableOp0SGD/conv3d_4/kernel/momentum/Read/ReadVariableOp.SGD/conv3d_4/bias/momentum/Read/ReadVariableOp0SGD/conv3d_5/kernel/momentum/Read/ReadVariableOp.SGD/conv3d_5/bias/momentum/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_130361
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcountSGD/conv3d/kernel/momentumSGD/conv3d/bias/momentumSGD/conv3d_1/kernel/momentumSGD/conv3d_1/bias/momentumSGD/conv3d_2/kernel/momentumSGD/conv3d_2/bias/momentumSGD/conv3d_3/kernel/momentumSGD/conv3d_3/bias/momentumSGD/conv3d_4/kernel/momentumSGD/conv3d_4/bias/momentumSGD/conv3d_5/kernel/momentumSGD/conv3d_5/bias/momentum**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_130461��
�

�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_130240

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
)__inference_conv3d_2_layer_call_fn_130146

inputs%
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�	
�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130174

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129480�
PartitionedCall_1PartitionedCallPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129614

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_5_layer_call_fn_130230

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�M
�	
A__inference_model_layer_call_and_return_conditional_losses_130037

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:	6
(conv3d_3_biasadd_readvariableop_resource:	E
'conv3d_4_conv3d_readvariableop_resource:	6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:
identity

identity_1��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_2/Relu:activations:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   t
IdentityIdentityconv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
B__inference_conv3d_layer_call_and_return_conditional_losses_130117

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_130221

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�S
�

!__inference__wrapped_model_129403
input_1I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:	<
.model_conv3d_1_biasadd_readvariableop_resource:	K
-model_conv3d_2_conv3d_readvariableop_resource:	<
.model_conv3d_2_biasadd_readvariableop_resource:K
-model_conv3d_3_conv3d_readvariableop_resource:	<
.model_conv3d_3_biasadd_readvariableop_resource:	K
-model_conv3d_4_conv3d_readvariableop_resource:	<
.model_conv3d_4_biasadd_readvariableop_resource:K
-model_conv3d_5_conv3d_readvariableop_resource:<
.model_conv3d_5_biasadd_readvariableop_resource:
identity��#model/conv3d/BiasAdd/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_2/Conv3DConv3D!model/conv3d_1/Relu:activations:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   |
7model/activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
5model/activity_regularization/ActivityRegularizer/AbsAbs!model/conv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
9model/activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
5model/activity_regularization/ActivityRegularizer/SumSum9model/activity_regularization/ActivityRegularizer/Abs:y:0Bmodel/activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: |
7model/activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5model/activity_regularization/ActivityRegularizer/mulMul@model/activity_regularization/ActivityRegularizer/mul/x:output:0>model/activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
5model/activity_regularization/ActivityRegularizer/addAddV2@model/activity_regularization/ActivityRegularizer/Const:output:09model/activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: �
7model/activity_regularization/ActivityRegularizer/ShapeShape!model/conv3d_2/Relu:activations:0*
T0*
_output_shapes
:�
Emodel/activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?model/activity_regularization/ActivityRegularizer/strided_sliceStridedSlice@model/activity_regularization/ActivityRegularizer/Shape:output:0Nmodel/activity_regularization/ActivityRegularizer/strided_slice/stack:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6model/activity_regularization/ActivityRegularizer/CastCastHmodel/activity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
9model/activity_regularization/ActivityRegularizer/truedivRealDiv9model/activity_regularization/ActivityRegularizer/add:z:0:model/activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	z
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_4/Conv3DConv3D!model/conv3d_3/Relu:activations:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
IdentityIdentitymodel/conv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   �
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�	
�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130181

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129614�
PartitionedCall_1PartitionedCallPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�4
�
A__inference_model_layer_call_and_return_conditional_losses_129838
input_1+
conv3d_129797:
conv3d_129799:-
conv3d_1_129802:	
conv3d_1_129804:	-
conv3d_2_129807:	
conv3d_2_129809:-
conv3d_3_129821:	
conv3d_3_129823:	-
conv3d_4_129826:	
conv3d_4_129828:-
conv3d_5_129831:
conv3d_5_129833:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_129797conv3d_129799*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_129436�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_129802conv3d_1_129804*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_129807conv3d_2_129809*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129480�
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_129821conv3d_3_129823*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_129826conv3d_4_129828*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_129831conv3d_5_129833*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534�
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
'__inference_conv3d_layer_call_fn_130106

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_129436{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�4
�
A__inference_model_layer_call_and_return_conditional_losses_129736

inputs+
conv3d_129695:
conv3d_129697:-
conv3d_1_129700:	
conv3d_1_129702:	-
conv3d_2_129705:	
conv3d_2_129707:-
conv3d_3_129719:	
conv3d_3_129721:	-
conv3d_4_129724:	
conv3d_4_129726:-
conv3d_5_129729:
conv3d_5_129731:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_129695conv3d_129697*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_129436�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_129700conv3d_1_129702*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_129705conv3d_2_129707*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129614�
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_129719conv3d_3_129721*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_129724conv3d_4_129726*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_129729conv3d_5_129731*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534�
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_130157

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_129917
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_129403{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
&__inference_model_layer_call_fn_129570
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_129542{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�

�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
V
?__inference_activity_regularization_activity_regularizer_129418
x
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: F
addAddV2Const:output:0mul:z:0*
T0*
_output_shapes
: >
IdentityIdentityadd:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
�4
�
A__inference_model_layer_call_and_return_conditional_losses_129542

inputs+
conv3d_129437:
conv3d_129439:-
conv3d_1_129454:	
conv3d_1_129456:	-
conv3d_2_129471:	
conv3d_2_129473:-
conv3d_3_129502:	
conv3d_3_129504:	-
conv3d_4_129519:	
conv3d_4_129521:-
conv3d_5_129535:
conv3d_5_129537:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_129437conv3d_129439*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_129436�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_129454conv3d_1_129456*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_129471conv3d_2_129473*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129480�
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_129502conv3d_3_129504*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_129519conv3d_4_129521*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_129535conv3d_5_129537*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534�
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_130137

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_129947

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_129542{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_130201

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
T
8__inference_activity_regularization_layer_call_fn_130167

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129614l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�4
�
A__inference_model_layer_call_and_return_conditional_losses_129882
input_1+
conv3d_129841:
conv3d_129843:-
conv3d_1_129846:	
conv3d_1_129848:	-
conv3d_2_129851:	
conv3d_2_129853:-
conv3d_3_129865:	
conv3d_3_129867:	-
conv3d_4_129870:	
conv3d_4_129872:-
conv3d_5_129875:
conv3d_5_129877:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_129841conv3d_129843*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_129436�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_129846conv3d_1_129848*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_129851conv3d_2_129853*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_129470�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129614�
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_129418�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_129865conv3d_3_129867*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_129870conv3d_4_129872*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_129875conv3d_5_129877*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_129534�
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
)__inference_conv3d_1_layer_call_fn_130126

inputs%
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_129453{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_3_layer_call_fn_130190

inputs%
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_129501{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129480

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130244

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
T
8__inference_activity_regularization_layer_call_fn_130162

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_129480l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_129977

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_129736{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�M
�	
A__inference_model_layer_call_and_return_conditional_losses_130097

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:	6
(conv3d_3_biasadd_readvariableop_resource:	E
'conv3d_4_conv3d_readvariableop_resource:	6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:
identity

identity_1��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_2/Relu:activations:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   t
IdentityIdentityconv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
B__inference_conv3d_layer_call_and_return_conditional_losses_129436

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130248

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_4_layer_call_fn_130210

inputs%
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_129518{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�D
�
__inference__traced_save_130361
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_sgd_conv3d_kernel_momentum_read_readvariableop7
3savev2_sgd_conv3d_bias_momentum_read_readvariableop;
7savev2_sgd_conv3d_1_kernel_momentum_read_readvariableop9
5savev2_sgd_conv3d_1_bias_momentum_read_readvariableop;
7savev2_sgd_conv3d_2_kernel_momentum_read_readvariableop9
5savev2_sgd_conv3d_2_bias_momentum_read_readvariableop;
7savev2_sgd_conv3d_3_kernel_momentum_read_readvariableop9
5savev2_sgd_conv3d_3_bias_momentum_read_readvariableop;
7savev2_sgd_conv3d_4_kernel_momentum_read_readvariableop9
5savev2_sgd_conv3d_4_bias_momentum_read_readvariableop;
7savev2_sgd_conv3d_5_kernel_momentum_read_readvariableop9
5savev2_sgd_conv3d_5_bias_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_sgd_conv3d_kernel_momentum_read_readvariableop3savev2_sgd_conv3d_bias_momentum_read_readvariableop7savev2_sgd_conv3d_1_kernel_momentum_read_readvariableop5savev2_sgd_conv3d_1_bias_momentum_read_readvariableop7savev2_sgd_conv3d_2_kernel_momentum_read_readvariableop5savev2_sgd_conv3d_2_bias_momentum_read_readvariableop7savev2_sgd_conv3d_3_kernel_momentum_read_readvariableop5savev2_sgd_conv3d_3_bias_momentum_read_readvariableop7savev2_sgd_conv3d_4_kernel_momentum_read_readvariableop5savev2_sgd_conv3d_4_bias_momentum_read_readvariableop7savev2_sgd_conv3d_5_kernel_momentum_read_readvariableop5savev2_sgd_conv3d_5_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::	:	:	::	:	:	:::: : : : : : :::	:	:	::	:	:	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0	,
*
_output_shapes
:	: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�z
�
"__inference__traced_restore_130461
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias:@
"assignvariableop_2_conv3d_1_kernel:	.
 assignvariableop_3_conv3d_1_bias:	@
"assignvariableop_4_conv3d_2_kernel:	.
 assignvariableop_5_conv3d_2_bias:@
"assignvariableop_6_conv3d_3_kernel:	.
 assignvariableop_7_conv3d_3_bias:	@
"assignvariableop_8_conv3d_4_kernel:	.
 assignvariableop_9_conv3d_4_bias:A
#assignvariableop_10_conv3d_5_kernel:/
!assignvariableop_11_conv3d_5_bias:&
assignvariableop_12_sgd_iter:	 '
assignvariableop_13_sgd_decay: /
%assignvariableop_14_sgd_learning_rate: *
 assignvariableop_15_sgd_momentum: #
assignvariableop_16_total: #
assignvariableop_17_count: L
.assignvariableop_18_sgd_conv3d_kernel_momentum::
,assignvariableop_19_sgd_conv3d_bias_momentum:N
0assignvariableop_20_sgd_conv3d_1_kernel_momentum:	<
.assignvariableop_21_sgd_conv3d_1_bias_momentum:	N
0assignvariableop_22_sgd_conv3d_2_kernel_momentum:	<
.assignvariableop_23_sgd_conv3d_2_bias_momentum:N
0assignvariableop_24_sgd_conv3d_3_kernel_momentum:	<
.assignvariableop_25_sgd_conv3d_3_bias_momentum:	N
0assignvariableop_26_sgd_conv3d_4_kernel_momentum:	<
.assignvariableop_27_sgd_conv3d_4_bias_momentum:N
0assignvariableop_28_sgd_conv3d_5_kernel_momentum:<
.assignvariableop_29_sgd_conv3d_5_bias_momentum:
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_sgd_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_sgd_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_sgd_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_sgd_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sgd_conv3d_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_sgd_conv3d_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_sgd_conv3d_1_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_sgd_conv3d_1_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp0assignvariableop_22_sgd_conv3d_2_kernel_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_sgd_conv3d_2_bias_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp0assignvariableop_24_sgd_conv3d_3_kernel_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_sgd_conv3d_3_bias_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_sgd_conv3d_4_kernel_momentumIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_sgd_conv3d_4_bias_momentumIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp0assignvariableop_28_sgd_conv3d_5_kernel_momentumIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_sgd_conv3d_5_bias_momentumIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
&__inference_model_layer_call_fn_129794
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_129736{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_1<
serving_default_input_1:0���������   H
conv3d_5<
StatefulPartitionedCall:0���������   tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
t__call__
*u&call_and_return_all_conditional_losses
v_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7iter
	8decay
9learning_rate
:momentummomentumhmomentumimomentumjmomentumkmomentumlmomentumm%momentumn&momentumo+momentump,momentumq1momentumr2momentums"
	optimizer
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics

	variables
trainable_variables
regularization_losses
t__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
+:)2conv3d/kernel
:2conv3d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_1/kernel
:	2conv3d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_2/kernel
:2conv3d_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
!	variables
"trainable_variables
#regularization_losses
}__call__
�activity_regularizer_fn
*~&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_3/kernel
:	2conv3d_3/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
'	variables
(trainable_variables
)regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_4/kernel
:2conv3d_4/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
-	variables
.trainable_variables
/regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
3	variables
4trainable_variables
5regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	dtotal
	ecount
f	variables
g	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
6:42SGD/conv3d/kernel/momentum
$:"2SGD/conv3d/bias/momentum
8:6	2SGD/conv3d_1/kernel/momentum
&:$	2SGD/conv3d_1/bias/momentum
8:6	2SGD/conv3d_2/kernel/momentum
&:$2SGD/conv3d_2/bias/momentum
8:6	2SGD/conv3d_3/kernel/momentum
&:$	2SGD/conv3d_3/bias/momentum
8:6	2SGD/conv3d_4/kernel/momentum
&:$2SGD/conv3d_4/bias/momentum
8:62SGD/conv3d_5/kernel/momentum
&:$2SGD/conv3d_5/bias/momentum
�2�
&__inference_model_layer_call_fn_129570
&__inference_model_layer_call_fn_129947
&__inference_model_layer_call_fn_129977
&__inference_model_layer_call_fn_129794�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_130037
A__inference_model_layer_call_and_return_conditional_losses_130097
A__inference_model_layer_call_and_return_conditional_losses_129838
A__inference_model_layer_call_and_return_conditional_losses_129882�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_129403input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_layer_call_fn_130106�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv3d_layer_call_and_return_conditional_losses_130117�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_1_layer_call_fn_130126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_130137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_2_layer_call_fn_130146�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_130157�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_activity_regularization_layer_call_fn_130162
8__inference_activity_regularization_layer_call_fn_130167�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130174
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130181�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
)__inference_conv3d_3_layer_call_fn_130190�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_130201�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_4_layer_call_fn_130210�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_130221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_5_layer_call_fn_130230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_130240�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_129917input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_activity_regularization_activity_regularizer_129418�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130244
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130248�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_129403�%&+,12<�9
2�/
-�*
input_1���������   
� "?�<
:
conv3d_5.�+
conv3d_5���������   i
?__inference_activity_regularization_activity_regularizer_129418&�
�
�	
x
� "� �
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130174�K�H
1�.
,�)
inputs���������   
�

trainingp "?�<
'�$
0���������   
�
�	
1/0 �
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_130181�K�H
1�.
,�)
inputs���������   
�

trainingp"?�<
'�$
0���������   
�
�	
1/0 �
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130244�K�H
1�.
,�)
inputs���������   
�

trainingp "1�.
'�$
0���������   
� �
S__inference_activity_regularization_layer_call_and_return_conditional_losses_130248�K�H
1�.
,�)
inputs���������   
�

trainingp"1�.
'�$
0���������   
� �
8__inference_activity_regularization_layer_call_fn_130162sK�H
1�.
,�)
inputs���������   
�

trainingp "$�!���������   �
8__inference_activity_regularization_layer_call_fn_130167sK�H
1�.
,�)
inputs���������   
�

trainingp"$�!���������   �
D__inference_conv3d_1_layer_call_and_return_conditional_losses_130137t;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   	
� �
)__inference_conv3d_1_layer_call_fn_130126g;�8
1�.
,�)
inputs���������   
� "$�!���������   	�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_130157t;�8
1�.
,�)
inputs���������   	
� "1�.
'�$
0���������   
� �
)__inference_conv3d_2_layer_call_fn_130146g;�8
1�.
,�)
inputs���������   	
� "$�!���������   �
D__inference_conv3d_3_layer_call_and_return_conditional_losses_130201t%&;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   	
� �
)__inference_conv3d_3_layer_call_fn_130190g%&;�8
1�.
,�)
inputs���������   
� "$�!���������   	�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_130221t+,;�8
1�.
,�)
inputs���������   	
� "1�.
'�$
0���������   
� �
)__inference_conv3d_4_layer_call_fn_130210g+,;�8
1�.
,�)
inputs���������   	
� "$�!���������   �
D__inference_conv3d_5_layer_call_and_return_conditional_losses_130240t12;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_5_layer_call_fn_130230g12;�8
1�.
,�)
inputs���������   
� "$�!���������   �
B__inference_conv3d_layer_call_and_return_conditional_losses_130117t;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
'__inference_conv3d_layer_call_fn_130106g;�8
1�.
,�)
inputs���������   
� "$�!���������   �
A__inference_model_layer_call_and_return_conditional_losses_129838�%&+,12D�A
:�7
-�*
input_1���������   
p 

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_129882�%&+,12D�A
:�7
-�*
input_1���������   
p

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_130037�%&+,12C�@
9�6
,�)
inputs���������   
p 

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_130097�%&+,12C�@
9�6
,�)
inputs���������   
p

 
� "?�<
'�$
0���������   
�
�	
1/0 �
&__inference_model_layer_call_fn_129570z%&+,12D�A
:�7
-�*
input_1���������   
p 

 
� "$�!���������   �
&__inference_model_layer_call_fn_129794z%&+,12D�A
:�7
-�*
input_1���������   
p

 
� "$�!���������   �
&__inference_model_layer_call_fn_129947y%&+,12C�@
9�6
,�)
inputs���������   
p 

 
� "$�!���������   �
&__inference_model_layer_call_fn_129977y%&+,12C�@
9�6
,�)
inputs���������   
p

 
� "$�!���������   �
$__inference_signature_wrapper_129917�%&+,12G�D
� 
=�:
8
input_1-�*
input_1���������   "?�<
:
conv3d_5.�+
conv3d_5���������   